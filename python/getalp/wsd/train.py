from getalp.wsd.trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', required=True, type=str, help=" ")
    parser.add_argument('--model_path', required=True, type=str, help=" ")
    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help=" ")
    parser.add_argument('--ensemble_count', nargs="?", type=int, default=8, help=" ")
    parser.add_argument('--epoch_count', nargs="?", type=int, default=100, help=" ")
    parser.add_argument('--eval_frequency', nargs="?", type=int, default=4000, help=" ")
    parser.add_argument('--update_frequency', nargs="?", type=int, default=1, help=" ")
    parser.add_argument('--lr', nargs="?", type=float, default=0.0001, help=" ")
    parser.add_argument('--warmup_sample_size', nargs="?", type=int, default=80, help=" ")
    parser.add_argument('--reset', action="store_true", help=" ")
    args = parser.parse_args()
    print(args)

    trainer = Trainer()
    trainer.data_path = args.data_path
    trainer.model_path = args.model_path
    trainer.batch_size = args.batch_size
    trainer.test_every_batch = args.eval_frequency
    trainer.update_every_batch = args.update_frequency
    trainer.stop_after_epoch = args.epoch_count
    trainer.ensemble_size = args.ensemble_count
    trainer.save_best_loss = False
    trainer.save_end_of_epoch = False
    trainer.shuffle_train_on_init = True
    trainer.warmup_sample_size = args.warmup_sample_size
    trainer.reset = args.reset
    trainer.learning_rate = args.lr

    trainer.train()


if __name__ == "__main__":
    main()
