# disambiguate: Neural Word Sense Disambiguation Toolkit

This repository contains a set of easy-to-use tools for training, evaluating and using neural WSD models. 
This is the implementation used in the article [Improving the Coverage and the Generalization Ability of Neural Word Sense Disambiguation through Hypernymy and Hyponymy Relationships](https://arxiv.org/abs/1811.00960), written by Loïc Vial, Benjamin Lecouteux and Didier Schwab.

## Dependencies
- Python (version 3.6 or higher) - <https://python.org>
- Java (version 8 or higher) - <https://java.com>
- Maven - <https://maven.apache.org>
- PyTorch (version 0.4.0 or higher) - <https://pytorch.org>
- UFSAC - <https://github.com/getalp/UFSAC>

To install Python, Java and Maven, you can use your favorite package manager (apt-get, pacman...).

To install PyTorch, please follow [this page](https://pytorch.org/get-started).

To install UFSAC, simply:
- download the sources from the [UFSAC repository](https://github.com/getalp/UFSAC)
- go into the `java` folder 
- run `mvn install`

## Compilation

Once the dependencies are installed, please run `./java/compile.sh` to compile the Java code. 

## Use pre-trained models

At the moment we are only providing one of our best model trained on the SemCor and the WordNet Gloss Tagged, with the vocabulary reduction applied, as described in [our article](https://arxiv.org/abs/1811.00960).

Here is the link to the data: <https://drive.google.com/open?id=14SCzyPiw9oMqdE1_9WlOMAAsYEPcOux0>

Once the data are downloaded and extracted, you can use the following commands (replace `$DATADIR` with the path of the appropriate folder):
- `./decode.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd`

  This script allows to disambiguate raw text from the standard input to the standard output

- `./evaluate.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd --corpus [UFSAC corpus]...` 

  This script evaluates a WSD model by computing its coverage, precision, recall and F1 scores on sense annotated corpora in the UFSAC format, with and without first sense backoff.  

Notes on the arguments:
- `--data_path` is the path to the directory containing the `input_vocabulary`, `output_vocabulary` and `config.json` files describing the model architecture
- `--weights` is a list of model weights: if multiple weights are given, an ensemble of these weights is used in `decode.sh`, and both the evaluation of the ensemble of weights and the evaluation of each individual weight is performed in `evaluate.sh`
- These two scripts also have the options `--lowercase` (default true) if you want to disable lowercasing of input, and `--sense_reduction` (default true) if you want to disable the sense vocabulary reduction method.
- UFSAC corpora are available in the [UFSAC repository](https://github.com/getalp/UFSAC), and notably if you want to reproduce our results, please download UFSAC 2.1 and you will find the SemCor (file `semcor.xml`, the WordNet Gloss Tagged (file `wngt.xml`) and all the SemEval/SensEval evaluation corpora.

## Train a WSD model

This part is still being written, please come back later :)
