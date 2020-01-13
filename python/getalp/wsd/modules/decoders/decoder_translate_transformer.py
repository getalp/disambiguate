from torch.nn import Module, Linear, ModuleList, LayerNorm, Dropout, Embedding
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from getalp.wsd.modules import PositionalEncoding
from getalp.wsd.torch_fix import *
from getalp.wsd.data_config import DataConfig
from getalp.wsd.model_config import ModelConfig
from onmt.decoders.transformer import TransformerDecoderLayer
from onmt.translate.beam_search import BeamSearch
from onmt.translate.beam import GNMTGlobalScorer
from onmt.utils.misc import tile
from getalp.wsd.torch_utils import default_device
from getalp.wsd.common import bos_token_index, eos_token_index, pad_token_index
import math


class DecoderTranslateTransformer(Module):

    def __init__(self, config: ModelConfig, data_config: DataConfig, encoder_embeddings):
        super().__init__()

        self.embeddings = Embedding(num_embeddings=data_config.output_translation_vocabulary_sizes[0][0],
                                    embedding_dim=config.encoder_output_size,
                                    padding_idx=pad_token_index)

        self.positional_encoding = PositionalEncoding(config.encoder_output_size)

        if config.decoder_translation_scale_embeddings:
            self.embeddings_scale = math.sqrt(float(config.encoder_output_size))
        else:
            self.embeddings_scale = None

        self.dropout = Dropout(config.decoder_translation_transformer_dropout)

        if config.decoder_translation_share_encoder_embeddings:
            self.embeddings.weight = encoder_embeddings.get_lut_embeddings().weight

        self.transformer_layers = ModuleList([TransformerDecoderLayer(d_model=config.encoder_output_size,
                                                                      heads=config.decoder_translation_transformer_heads,
                                                                      d_ff=config.decoder_translation_transformer_hidden_size,
                                                                      dropout=config.decoder_translation_transformer_dropout,
                                                                      attention_dropout=config.decoder_translation_transformer_dropout)
                                              for _ in range(config.decoder_translation_transformer_layers)])

        self.layer_norm = LayerNorm(config.encoder_output_size, eps=1e-6)

        self.linear: Linear = Linear(in_features=config.encoder_output_size, out_features=data_config.output_translation_vocabulary_sizes[0][0])

        if config.decoder_translation_share_embeddings:
            self.linear.weight = self.embeddings.weight

        self.max_seq_out_len = 150
        self.beam_size = 1
        self.state = {}

    # input:
    #   - encoder_output:         FloatTensor - batch x seq_in x hidden
    #   - pad_mask:               LongTensor  - batch x seq_in
    #   - true_output (training): LongTensor  - batch x seq_out
    # output:
    #   - output:                 FloatTensor - batch x seq_out x vocab_out
    def forward(self, encoder_output: torch.Tensor, pad_mask: torch.Tensor, true_output: torch.Tensor):
        pad_mask = pad_mask.transpose(0, 1)  # seq_in x batch
        encoder_output = encoder_output.transpose(0, 1)  # seq_in x batch x hidden
        if true_output is None:
            output = self.forward_dev(encoder_output, pad_mask)
        else:
            true_output = true_output.transpose(0, 1)    # seq_out x batch
            output = self.forward_train(encoder_output, pad_mask, true_output)
        return output

    def forward_step(self, src, tgt, memory_bank, step=None):
        if step == 0:
            self.state["cache"] = {}
            for i, layer in enumerate(self.transformer_layers):
                layer_cache = {"memory_keys": None, "memory_values": None, "self_keys": None, "self_values": None}
                self.state["cache"]["layer_{}".format(i)] = layer_cache

        src = src.transpose(0, 1)  # batch x seq_in
        tgt = tgt.transpose(0, 1)  # batch x seq_out
        memory_bank = memory_bank.transpose(0, 1)  # batch x seq_in x hidden

        output = self.embeddings(tgt)  # batch x seq_out x hidden
        if self.embeddings_scale is not None:
            output = output * self.embeddings_scale
        if self.positional_encoding is not None:
            if step is None:
                output = output + self.positional_encoding(output.size(1), full=True)
            else:
                output = output + self.positional_encoding(step, full=False)
        output = self.dropout(output)

        src_pad_mask = src.data.eq(pad_token_index).unsqueeze(1)  # [B, 1, T_src]
        tgt_pad_mask = tgt.data.eq(pad_token_index).unsqueeze(1)  # [B, 1, T_tgt]

        attn = None
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] if step is not None else None
            output, attn = layer(output, memory_bank, src_pad_mask, tgt_pad_mask, layer_cache=layer_cache, step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1)
        attn = attn.transpose(0, 1)

        return dec_outs, attn

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def forward_train(self, encoder_output, pad_mask, true_output):
        bos_true_output = self.make_bos_token(encoder_output.size(1))  # 1 x batch
        true_output = true_output[:-1]  # seq-1 x batch
        true_output = torch_cat((bos_true_output, true_output), dim=0)  # seq x batch
        output, _ = self.forward_step(src=pad_mask, tgt=true_output, memory_bank=encoder_output, step=None)  # seq_out x batch x hidden
        output = self.linear(output)  # seq_out x batch x vocab_out
        output = output.transpose(0, 1)  # batch x seq_out x vocab_out
        return output

    @torch.no_grad()
    def forward_dev(self, encoder_output, pad_mask):
        if self.beam_size == 1:
            return self.forward_dev_greedy(encoder_output, pad_mask)
        else:
            return self.forward_dev_beam_search(encoder_output, pad_mask)

    @torch.no_grad()
    def forward_dev_greedy(self, encoder_output, pad_mask):
        batch_size = encoder_output.size(1)
        last_output = self.make_bos_token(batch_size)  # 1 x batch
        outputs = []  # List[1 x batch]
        eos_count = torch_zeros(batch_size, dtype=torch_uint8, device=default_device)  # batch
        for i in range(self.max_seq_out_len):
            last_output, _ = self.forward_step(src=pad_mask, tgt=last_output, memory_bank=encoder_output, step=i)  # 1 x batch x hidden
            last_output = self.linear(last_output)  # 1 x batch x vocab_out
            last_output = torch_argmax(last_output, dim=2)  # 1 x batch
            outputs.append(last_output)
            eos_count = eos_count.add(last_output.eq(eos_token_index))
            if eos_count.gt(0).sum() >= batch_size:
                break
        return torch_cat(outputs, dim=0).transpose(0, 1)  # batch x seq

    @torch.no_grad()
    def forward_dev_beam_search(self, encoder_output: torch.Tensor, pad_mask):
        batch_size = encoder_output.size(1)

        self.state["cache"] = None
        memory_lengths = pad_mask.ne(pad_token_index).sum(dim=0)

        self.map_state(lambda state, dim: tile(state, self.beam_size, dim=dim))
        encoder_output = tile(encoder_output, self.beam_size, dim=1)
        pad_mask = tile(pad_mask, self.beam_size, dim=1)
        memory_lengths = tile(memory_lengths, self.beam_size, dim=0)

        beam = BeamSearch(beam_size=self.beam_size, n_best=1, batch_size=batch_size, mb_device=default_device,
                          global_scorer=GNMTGlobalScorer(alpha=0.2, beta=0.2, coverage_penalty="none", length_penalty="avg"),
                          pad=pad_token_index, eos=eos_token_index, bos=bos_token_index, min_length=1, max_length=100,
                          return_attention=False, stepwise_penalty=False, block_ngram_repeat=0, exclusion_tokens=set(),
                          memory_lengths=memory_lengths, ratio=-1)

        for i in range(self.max_seq_out_len):
            inp = beam.current_predictions.view(1, -1)

            out, attn = self.forward_step(src=pad_mask, tgt=inp, memory_bank=encoder_output, step=i)  # 1 x batch*beam x hidden
            out = self.linear(out)  # 1 x batch*beam x vocab_out
            out = log_softmax(out, dim=2)  # 1 x batch*beam x vocab_out

            out = out.squeeze(0)  # batch*beam x vocab_out
            # attn = attn.squeeze(0)  # batch*beam x vocab_out
            # out = out.view(batch_size, self.beam_size, -1)  # batch x beam x vocab_out
            # attn = attn.view(batch_size, self.beam_size, -1)
            # TODO: fix attn and use coverage_penalty="summary"

            beam.advance(out, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                encoder_output = encoder_output.index_select(1, select_indices)
                pad_mask = pad_mask.index_select(1, select_indices)
                memory_lengths = memory_lengths.index_select(0, select_indices)

            self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        outputs = beam.predictions
        outputs = [x[0] for x in outputs]
        outputs = pad_sequence(outputs, batch_first=True)
        return outputs

    # output : 1 x batch
    @staticmethod
    def make_bos_token(batch_size):
        return torch_full(size=(1, batch_size), fill_value=bos_token_index, device=default_device, dtype=torch_long)
