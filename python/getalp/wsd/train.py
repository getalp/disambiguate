from getalp.wsd.trainer import Trainer
from getalp.common.common import str2bool
import argparse
import pprint


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', required=True, type=str, help=" ")
    parser.add_argument('--model_path', required=True, type=str, help=" ")
    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help=" ")
    parser.add_argument('--token_per_batch', nargs="?", type=int, default=8000, help=" ")
    parser.add_argument('--ensemble_count', nargs="?", type=int, default=8, help=" ")
    parser.add_argument('--epoch_count', nargs="?", type=int, default=40, help=" ")
    parser.add_argument('--eval_frequency', nargs="?", type=int, default=4000, help=" ")
    parser.add_argument('--update_frequency', nargs="?", type=int, default=1, help=" ")
    parser.add_argument('--warmup_batch_count', nargs="?", type=int, default=10, help=" ")
    parser.add_argument('--input_embeddings_size', nargs="+", type=int, default=None, help=" ")
    parser.add_argument('--input_embeddings_tokenize_model', nargs="+", type=str, default=None, help=" ")
    parser.add_argument('--input_elmo_model', nargs="+", type=str, default=None, help=" ")
    parser.add_argument('--input_bert_model', nargs="+", type=str, default=None, help=" ")
    parser.add_argument('--input_auto_model', nargs="+", type=str, default=None, help=" ")
    parser.add_argument('--input_auto_path', nargs="+", type=str, default=None, help=" ")
    parser.add_argument('--input_word_dropout_rate', nargs="?", type=float, default=None, help=" ")
    parser.add_argument('--input_resize', nargs="+", type=int, default=None, help=" ")
    parser.add_argument('--input_linear_size', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--input_dropout_rate', nargs="?", type=float, default=None, help=" ")
    parser.add_argument('--encoder_type', nargs="?", type=str, default=None, help=" ")
    parser.add_argument('--encoder_lstm_hidden_size', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--encoder_lstm_layers', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--encoder_lstm_dropout', nargs="?", type=float, default=None, help=" ")
    parser.add_argument('--encoder_transformer_hidden_size', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--encoder_transformer_layers', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--encoder_transformer_heads', nargs="?", type=int, default=None, help=" ")
    parser.add_argument('--encoder_transformer_dropout', nargs="?", type=float, default=None, help=" ")
    parser.add_argument('--encoder_transformer_positional_encoding', nargs="?", type=str2bool, default=None, help=" ")
    parser.add_argument('--encoder_transformer_scale_embeddings', nargs="?", type=str2bool, default=None, help=" ")
    parser.add_argument('--optimizer', nargs="?", type=str, default="adam", help=" ", choices=["adam"])
    parser.add_argument('--adam_beta1', nargs="?", type=float, default=0.9, help=" ")
    parser.add_argument('--adam_beta2', nargs="?", type=float, default=0.999, help=" ")
    parser.add_argument('--adam_eps', nargs="?", type=float, default=1e-8, help=" ")
    parser.add_argument('--lr_scheduler', nargs="?", type=str, default="fixed", help=" ", choices=("fixed", "noam"))
    parser.add_argument('--lr_scheduler_fixed_lr', nargs="?", type=float, default=0.0001, help=" ")
    parser.add_argument('--lr_scheduler_noam_warmup', nargs="?", type=int, default=6000, help=" ")
    parser.add_argument('--lr_scheduler_noam_model_size', nargs="?", type=int, default=512, help=" ")
    parser.add_argument('--reset', action="store_true", help=" ")
    parser.add_argument('--save_best_loss', action="store_true", help=" ")
    parser.add_argument('--save_every_epoch', action="store_true", help=" ")
    args = parser.parse_args()
    print("Command line arguments:")
    pprint.pprint(vars(args))

    trainer = Trainer()

    trainer.data_path = args.data_path
    trainer.model_path = args.model_path
    trainer.batch_size = args.batch_size
    trainer.token_per_batch = args.token_per_batch
    trainer.eval_frequency = args.eval_frequency
    trainer.update_every_batch = args.update_frequency
    trainer.stop_after_epoch = args.epoch_count
    trainer.ensemble_size = args.ensemble_count
    trainer.save_best_loss = args.save_best_loss
    trainer.save_end_of_epoch = args.save_every_epoch
    trainer.shuffle_train_on_init = True
    trainer.warmup_batch_count = args.warmup_batch_count
    trainer.input_embeddings_size = args.input_embeddings_size
    trainer.input_embeddings_tokenize_model = args.input_embeddings_tokenize_model
    trainer.input_elmo_model = args.input_elmo_model
    trainer.input_bert_model = args.input_bert_model
    trainer.input_auto_model = args.input_auto_model
    trainer.input_auto_path = args.input_auto_path
    trainer.input_word_dropout_rate = args.input_word_dropout_rate
    trainer.input_resize = args.input_resize
    trainer.input_linear_size = args.input_linear_size
    trainer.input_dropout_rate = args.input_dropout_rate
    trainer.encoder_type = args.encoder_type
    trainer.encoder_lstm_layers = args.encoder_lstm_layers
    trainer.encoder_lstm_hidden_size = args.encoder_lstm_hidden_size
    trainer.encoder_lstm_dropout = args.encoder_lstm_dropout
    trainer.encoder_transformer_hidden_size = args.encoder_transformer_hidden_size
    trainer.encoder_transformer_layers = args.encoder_transformer_layers
    trainer.encoder_transformer_heads = args.encoder_transformer_heads
    trainer.encoder_transformer_dropout = args.encoder_transformer_dropout
    trainer.encoder_transformer_positional_encoding = args.encoder_transformer_positional_encoding
    trainer.encoder_transformer_scale_embeddings = args.encoder_transformer_scale_embeddings
    trainer.optimizer = args.optimizer
    trainer.adam_beta1 = args.adam_beta1
    trainer.adam_beta2 = args.adam_beta2
    trainer.adam_eps = args.adam_eps
    trainer.lr_scheduler = args.lr_scheduler
    trainer.lr_scheduler_fixed_lr = args.lr_scheduler_fixed_lr
    trainer.lr_scheduler_noam_warmup = args.lr_scheduler_noam_warmup
    trainer.lr_scheduler_noam_model_size = args.lr_scheduler_noam_model_size
    trainer.reset = args.reset

    trainer.train()


if __name__ == "__main__":
    main()
