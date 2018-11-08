# disambiguate: Neural Word Sense Disambiguation Toolkit

This repository contains a set of easy-to-use tools for training, evaluating and using neural WSD models. 
This is the implementation used in the article [Improving the Coverage and the Generalization Ability of Neural Word Sense Disambiguation through Hypernymy and Hyponymy Relationships](https://arxiv.org/abs/1811.00960), written by Loïc Vial, Benjamin Lecouteux and Didier Schwab.

## Dependencies
- Python (version 3.6 or higher) - <https://python.org>
- Java (version 8 or higher) - <https://java.com>
- Maven - <https://maven.apache.org>
- PyTorch (version 0.4.0 or higher) - <https://pytorch.org>
- UFSAC - <https://github.com/getalp/UFSAC>

To install **Python**, **Java** and **Maven**, you can use the package manager of your distribution (apt-get, pacman...).

To install **PyTorch**, please follow [this page](https://pytorch.org/get-started).

To install **UFSAC**, simply:
- download the sources from the [UFSAC repository](https://github.com/getalp/UFSAC)
- go into the `java` folder 
- run `mvn install`

## Compilation

Once the dependencies are installed, please run `./java/compile.sh` to compile the Java code. 

## Use pre-trained models

At the moment we are only providing one of our best model trained on the SemCor and the WordNet Gloss Tagged, with the vocabulary reduction applied, as described in [our article](https://arxiv.org/abs/1811.00960).

Here is the link to the data: <https://drive.google.com/open?id=1_-CxENMkmUSGkcmb6xcFBhJR114A4GsY>

Once the data are downloaded and extracted, you can use the following commands (replace `$DATADIR` with the path of the appropriate folder):
- `./decode.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd`

  This script allows to disambiguate raw text from the standard input to the standard output

- `./evaluate.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd --corpus [UFSAC corpus]...` 

  This script evaluates a WSD model by computing its coverage, precision, recall and F1 scores on sense annotated corpora in the UFSAC format, with and without first sense backoff.  

Description of the arguments:
- `--data_path [DIR]` is the path to the directory containing the files needed for describing the model architecture (files `config.json`, `input_vocabularyX` and `output_vocabularyX`) 
- `--weights [FILE]...` is a list of model weights: if multiple weights are given, an ensemble of these weights is used in `decode.sh`, and both the evaluation of the ensemble of weights and the evaluation of each individual weight is performed in `evaluate.sh`
- `--corpus [FILE]...` (`evaluate.sh` only) is the list of UFSAC corpora used for evaluating the WSD model

Optional arguments: 
- `--lowercase [true|false]` (default true) if you want to enable/disable lowercasing of input
- `--sense_reduction [true|false]` (default true) if you want to enable/disable the sense vocabulary reduction method.

UFSAC corpora are available in the [UFSAC repository](https://github.com/getalp/UFSAC). If you want to reproduce our results, please download UFSAC 2.1 and you will find the SemCor (file `semcor.xml`, the WordNet Gloss Tagged (file `wngt.xml`) and all the SemEval/SensEval evaluation corpora that we used.

## Train a WSD model

To train a model, first call the `./prepare_data.sh` script with the following arguments:
- `--data_path [DIR]` is the path to the directory that will contain the description of the model (files `config.json`, `input_vocabularyX` and `output_vocabularyX`) and the processed training data (files `train` and `dev`)
- `--train [FILE]...` is the list of corpora in UFSAC format used for training
- `--dev [FILE]...` (optional) is the list of corpora in UFSAC format used for development
- `--input_features [FEATURE]...` (default surface\_form) is the list of input features used, as UFSAC attributes. Possible values are, but not limited to, *surface_form*, *lemma*, *pos*, *wn30\_key*...
- `--output_features [FEATURE]...` (default wn30\_key) is the list of output features to predict by the model, as UFSAC attributes. Possible values are the same as input features
- `--lowercase [true|false]` (default true) if you want to enable/disable lowercasing of input
- `--sense_reduction [true|false]` (default true) if you want to enable/disable the sense vocabulary reduction method.
- `--add_monosemics [true|false]` (default false) if you want to consider all monosemic words annotated with their unique sense tag (even if they are not initially annotated) 
- `--remove_monosemics [true|false]` (default false) if you want to remove the tag of all monosemic words
- `--remove_duplicates [true|false]` (default true) if you want to remove duplicate sentences from the training set (output features are merged)





This section is still being written, please come back later for more :)
