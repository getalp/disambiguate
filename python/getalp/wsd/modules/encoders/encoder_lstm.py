from torch.nn import Module, LSTM, Dropout
from getalp.wsd.model_config import ModelConfig
from getalp.wsd.modules.encoders.encoder_base import EncoderBase


class EncoderLSTM(Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.base = EncoderBase(config)

        if config.encoder_lstm_layers > 0:
            self.lstm = LSTM(input_size=self.base.resulting_embeddings_size, hidden_size=config.encoder_lstm_hidden_size,
                             num_layers=config.encoder_lstm_layers, bidirectional=True, batch_first=True)
            config.encoder_output_size = config.encoder_lstm_hidden_size * 2
        else:
            self.lstm = None
            config.encoder_output_size = self.base.resulting_embeddings_size

        if config.encoder_lstm_dropout is not None:
            self.dropout = Dropout(p=config.encoder_lstm_dropout)
        else:
            self.dropout = None

    # input:
    #   - embeddings     List[FloatTensor] - features x batch x seq x hidden
    #   - pad_mask       LongTensor        - batch x seq
    # output:
    #   - output         FloatTensor       - batch x seq x hidden
    def forward(self, embeddings, pad_mask):
        embeddings = self.base(embeddings, pad_mask)
        if self.lstm is not None:
            embeddings, (_, _) = self.lstm(embeddings)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return embeddings
