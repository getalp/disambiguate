# disambiguate: Neural Word Sense Disambiguation Toolkit


This repository contains a set of easy-to-use tools for training, evaluating and using neural WSD models.

This is the implementation used in the article [Sense Vocabulary Compression through the Semantic Knowledge of WordNet for Neural Word Sense Disambiguation](https://arxiv.org/abs/1905.05677), written by Loïc Vial, Benjamin Lecouteux and Didier Schwab.

## Dependencies
- Python (version 3.6 or higher) - <https://python.org>
- Java (version 8 or higher) - <https://java.com>
- Maven - <https://maven.apache.org>
- PyTorch (version 1.0.0 or higher) - <https://pytorch.org>
- (optional, for using ELMo) AllenNLP - <https://allennlp.org>
- (optional, for using BERT) huggingface's pytorch-pretrained-BERT - <https://github.com/huggingface/pytorch-pretrained-BERT>
- UFSAC - <https://github.com/getalp/UFSAC>

To install **Python**, **Java** and **Maven**, you can use the package manager of your distribution (apt-get, pacman...).

To install **PyTorch**, please follow the instructions on [this page](https://pytorch.org/get-started).

To install **AllenNLP** (necessary if using ELMo), please follow the instructions on [this page](https://allennlp.org/tutorials).

To install **huggingface's pytorch-pretrained-BERT** (necessary if using BERT), please follow the instructions on [this page](https://github.com/huggingface/pytorch-pretrained-BERT).

To install **UFSAC**, simply:

- download the content of the [UFSAC repository](https://github.com/getalp/UFSAC)
- go into the `java` folder
- run `mvn install`

## Compilation

Once the dependencies are installed, please run `./java/compile.sh` to compile the Java code.

## Sense mappings

We provide the two sense mappings used in our paper as standalone files in the directory `sense_mappings`.

The files consist of 117659 lines (one line by synset): the left-hand ID is the original synset ID, and the right-hand is the ID of the associated group of synsets.

The file `hypernyms_mapping.txt` results from the sense compression method through hypernyms. The exact algorithm that was used is located in the method `getSenseCompressionThroughHypernymsClusters()` of the file `java/src/main/java/getalp/wsd/utils/WordnetUtils.java`.

The file `all_relations_mapping.txt` results from the method through all relationships. The exact algorithm that was used is located in the method `getSenseCompressionThroughAllRelationsClusters()` of the file `java/src/main/java/getalp/wsd/utils/WordnetUtils.java`.

## Using pre-trained models

We are currently providing one of our best model trained on the SemCor and the WordNet Gloss Tagged, using BERT embeddings, with the vocabulary compression through the hypernymy/hyponymy relationships applied, as described in [our article](https://arxiv.org/abs/1905.05677).

Here is the link to the data: <https://drive.google.com/file/d/14OmLqKsbV4M50WN8DvqN76uJl5E96iTo>

Once the data are downloaded and extracted, you can use the following commands (replace `$DATADIR` with the path of the appropriate folder):

### Disambiguating raw text

- `./decode.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd0`

  This script allows to disambiguate raw text from the standard input to the standard output

### Evaluating a model

- `./evaluate.sh --data_path $DATADIR --weights $DATADIR/model_weights_wsd0 --corpus [UFSAC corpus]...` 

  This script evaluates a WSD model by computing its coverage, precision, recall and F1 scores on sense annotated corpora in the UFSAC format, with and without first sense backoff.

Description of the arguments:

- `--data_path [DIR]` is the path to the directory containing the files needed for describing the model architecture (files `config.json`, `input_vocabularyX` and `output_vocabularyX`)
- `--weights [FILE]...` is a list of model weights: if multiple weights are given, an ensemble of these weights is used in `decode.sh`, and both the evaluation of the ensemble of weights and the evaluation of each individual weight is performed in `evaluate.sh`
- `--corpus [FILE]...` (`evaluate.sh` only) is the list of UFSAC corpora used for evaluating the WSD model

Optional arguments:

- `--lowercase [true|false]` (default `false`) if you want to enable/disable lowercasing of input
- `--batch_size [n]` (default `1`) is the batch size.
- `--sense_compression_hypernyms [true|false]` (default `true`) must be `true` if the model was trained using the sense vocabulary compression through the hypernym/hyponym relationships, or `false` otherwise.
- `--sense_compression_file [FILE]` must indicate the path of the sense mapping file used for training the model if any, and if different from the hypernyms mapping.

UFSAC corpora are available in the [UFSAC repository](https://github.com/getalp/UFSAC). If you want to reproduce our results, please download UFSAC 2.1 and you will find the SemCor (file `semcor.xml`, the WordNet Gloss Tagged (file `wngt.xml`) and all the SemEval/SensEval evaluation corpora that we used (files `raganato_*.xml`).

## Training new WSD models

### Preparing data

Call the `./prepare_data.sh` script with the following main arguments:

- `--data_path [DIR]` is the path to the directory that will contain the description of the model (files `config.json`, `input_vocabularyX` and `output_vocabularyX`) and the processed training data (files `train` and `dev`)
- `--train [FILE]...` is the list of corpora in UFSAC format used for the training set
- `--dev [FILE]...` (optional) is the list of corpora in UFSAC format used for the development set
- `--dev_from_train [N]` (default `0`) randomly extracts `N` sentences from the training corpus and use it as development corpus
- `--input_features [FEATURE]...` (default `surface_form`) is the list of input features used, as UFSAC attributes. Possible values are, but not limited to, `surface_form`, `lemma`, `pos`, `wn30_key`...
- `--input_embeddings [FILE]...` (default `null`) is the list of pre-trained embeddings to use for each input feature. Must be the same number of arguments as `input_features`, use special value `null` if you want to train embeddings as part of the model
- `--input_clear_text [true|false]...` (default `false`) is a list of true/false values (one value for each input feature) indicating if the feature must be used as clear text (e.g. with ELMo/BERT) or as integer values (with classic embeddings). Must be the same number of arguments as `input_features`
- `--output_features [FEATURE]...` (default `wn30_key`) is the list of output features to predict by the model, as UFSAC attributes. Possible values are the same as input features
- `--lowercase [true|false]` (default `true`) if you want to enable/disable lowercasing of input
- `--sense_compression_hypernyms [true|false]` (default `true`) if you want to enable/disable the sense vocabulary compression through the hypernym/hyponym relationships.
- `--sense_compression_file [FILE]` if you want to use another sense vocabulary compression mapping.
- `--add_monosemics [true|false]` (default `false`) if you want to consider all monosemic words annotated with their unique sense tag (even if they are not initially annotated)
- `--remove_monosemics [true|false]` (default `false`) if you want to remove the tag of all monosemic words
- `--remove_duplicates [true|false]` (default `true`) if you want to remove duplicate sentences from the training set (output features are merged)

### Training a model (or an ensemble of models)

Call the `./train.sh` script with the following main arguments:

- `--data_path [DIR]` is the path to the directory generated by `prepare_data.sh` (must contains the files describing the model and the processed training data)
- `--model_path [DIR]` is the path where the trained model weights and the training info will be saved
- `--batch_size [N]` (default `100`) is the batch size
- `--ensemble_count [N]` (default `8`) is the number of different model to train
- `--epoch_count [N]` (default `100`) is the number of epoch
- `--eval_frequency [N]` (default `4000`) is the number of batch to process before evaluating the model on the development set. The count resets every epoch, and an eveluation is also performed at the end of every epoch
- `--update_frequency [N]` (default `1`) is the number of batch to accumulate before backpropagating (if you want to accumulate the gradient of several batches)
- `--lr [N]` (default `0.0001`) is the initial learning rate of the optimizer (Adam)
- `--input_embeddings_size [N]` (default `300`) is the size of input embeddings (if not using pre-trained embeddings, BERT nor ELMo)
- `--input_elmo_model [MODEL]` is the name of the ELMo model to use (one of `small`, `medium` or `original`), it will be downloaded automatically.
- `--input_bert_model [MODEL]` is the name of the BERT model to use (of the form `bert-{base,large}-(multilingual-)(un)cased`), it will be downloaded automatically.
- `--encoder_type [ENCODER]` (default `lstm`) is one of `lstm` or `transformer`.
- `--encoder_lstm_hidden_size [N]` (default `1000`)
- `--encoder_lstm_layers [N]` (default `1`)
- `--encoder_lstm_dropout [N]` (default `0.5`)
- `--encoder_transformer_hidden_size [N]` (default `512`)
- `--encoder_transformer_layers [N]` (default `6`)
- `--encoder_transformer_heads [N]` (default `8`)
- `--encoder_transformer_positional_encoding [true|false]` (default `true`)
- `--encoder_transformer_dropout [N]` (default `0.1`)
- `--reset [true|false]` (default `false`) if you do not want to resume a previous training. Be careful as it will effectively resets the training state and the model weights saved in the `--model_path`

