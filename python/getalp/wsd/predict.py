from getalp.wsd.predicter import Predicter
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--weights', nargs="+", type=str)
    args = parser.parse_args()

    predicter = Predicter()
    predicter.training_root_path = args.data_path
    predicter.ensemble_weights_path = args.weights

    predicter.predict()


if __name__ == "__main__":
    main()
