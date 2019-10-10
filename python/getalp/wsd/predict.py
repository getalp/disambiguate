from getalp.wsd.predicter import Predicter
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--weights', nargs="+", type=str)
    parser.add_argument('--clear_text', action="store_true", help=" ")
    parser.add_argument('--batch_size', nargs="?", type=int, default=1, help=" ")
    parser.add_argument('--disambiguate', action="store_true", help=" ")
    parser.add_argument('--beam_size', nargs="?", type=int, default=1, help=" ")
    parser.add_argument('--output_all_features', action="store_true", help=" ")
    parser.add_argument('--write_log', action="store_true", help=" ")
    args = parser.parse_args()

    predicter = Predicter()
    predicter.training_root_path = args.data_path
    predicter.ensemble_weights_path = args.weights
    predicter.clear_text = args.clear_text
    predicter.batch_size = args.batch_size
    predicter.disambiguate = args.disambiguate
    predicter.beam_size = args.beam_size
    predicter.output_all_features = args.output_all_features
    predicter.write_log = args.write_log

    predicter.predict()


if __name__ == "__main__":
    main()
