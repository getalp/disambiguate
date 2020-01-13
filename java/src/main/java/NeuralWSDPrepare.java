import java.util.*;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.neural.NeuralDataPreparator;
import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

public class NeuralWSDPrepare
{
    public static void main(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("data_path");
        parser.addArgumentList("train");
        parser.addArgumentList("dev", Collections.emptyList());
        parser.addArgument("dev_from_train", "0");
        parser.addArgument("corpus_format", "xml");
        parser.addArgumentList("txt_corpus_features", Collections.singletonList("null"));
        parser.addArgumentList("input_features", Collections.singletonList("surface_form"));
        parser.addArgumentList("input_embeddings", Collections.singletonList("null"));
        parser.addArgumentList("input_vocabulary", Collections.singletonList("null"));
        parser.addArgument("input_vocabulary_limit", "-1");
        parser.addArgumentList("input_clear_text", Collections.singletonList("false"));
        parser.addArgumentList("output_features", Collections.singletonList("wn30_key"));
        parser.addArgument("output_feature_vocabulary_limit", "-1");
        parser.addArgument("truncate_line_length", "80");
        parser.addArgument("exclude_line_length", "150");
        parser.addArgument("line_length_tokenizer", "null");
        parser.addArgument("lowercase", "false");
        parser.addArgument("filter_lemma", "true");
        parser.addArgument("uniform_dash", "false");
        parser.addArgument("sense_compression_hypernyms", "true");
        parser.addArgument("sense_compression_instance_hypernyms", "false");
        parser.addArgument("sense_compression_antonyms", "false");
        parser.addArgument("sense_compression_file", "");
        parser.addArgument("add_wordkey_from_sensekey", "false");
        parser.addArgument("add_monosemics", "false");
        parser.addArgument("remove_monosemics", "false");
        parser.addArgument("remove_duplicates", "true");
        if (!parser.parse(args, true)) return;

        String dataPath = parser.getArgValue("data_path");
        List<String> trainingCorpusPaths = parser.getArgValueList("train");
        List<String> devCorpusPaths = parser.getArgValueList("dev");
        int devFromTrain = parser.getArgValueInteger("dev_from_train");
        String corpusFormat = parser.getArgValue("corpus_format");
        List<String> txtCorpusFeatures = parser.getArgValueList("txt_corpus_features");
        List<String> inputFeatures = parser.getArgValueList("input_features");
        List<String> inputEmbeddings = parser.getArgValueList("input_embeddings");
        List<String> inputVocabulary = parser.getArgValueList("input_vocabulary");
        int inputVocabularyLimit = parser.getArgValueInteger("input_vocabulary_limit");
        List<Boolean> inputClearText = parser.getArgValueBooleanList("input_clear_text");
        List<String> outputFeatures = parser.getArgValueList("output_features");
        int outputFeatureVocabularyLimit = parser.getArgValueInteger("output_feature_vocabulary_limit");
        int maxLineLength = parser.getArgValueInteger("truncate_line_length");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean filterLemma = parser.getArgValueBoolean("filter_lemma");
        boolean uniformDash = parser.getArgValueBoolean("uniform_dash");
        boolean senseCompressionHypernyms = parser.getArgValueBoolean("sense_compression_hypernyms");
        boolean senseCompressionInstanceHypernyms = parser.getArgValueBoolean("sense_compression_instance_hypernyms");
        boolean senseCompressionAntonyms = parser.getArgValueBoolean("sense_compression_antonyms");
        String senseCompressionFile = parser.getArgValue("sense_compression_file");
        boolean addWordKeyFromSenseKey = parser.getArgValueBoolean("add_wordkey_from_sensekey");
        boolean addMonosemics = parser.getArgValueBoolean("add_monosemics");
        boolean removeMonosemics = parser.getArgValueBoolean("remove_monosemics");
        boolean removeDuplicateSentences = parser.getArgValueBoolean("remove_duplicates");

        Map<String, String> senseCompressionClusters = null;
        if (senseCompressionHypernyms || senseCompressionAntonyms)
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClusters(WordnetHelper.wn30(), senseCompressionHypernyms, senseCompressionInstanceHypernyms, senseCompressionAntonyms);
        }
        if (!senseCompressionFile.isEmpty())
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClustersFromFile(senseCompressionFile);
        }

        inputEmbeddings = padList(inputEmbeddings, inputFeatures.size(), "null");
        inputVocabulary = padList(inputVocabulary, inputFeatures.size(), "null");
        inputClearText = padList(inputClearText, inputFeatures.size(), false);

        NeuralDataPreparator preparator = new NeuralDataPreparator();

        preparator.addWordKeyFromSenseKey = addWordKeyFromSenseKey;

        if (txtCorpusFeatures.size() == 1 && txtCorpusFeatures.get(0).equals("null"))
        {
            txtCorpusFeatures = Collections.emptyList();
        }
        preparator.txtCorpusFeatures = txtCorpusFeatures;

        preparator.setOutputDirectoryPath(dataPath);

        for (String corpusPath : trainingCorpusPaths)
        {
            preparator.addTrainingCorpus(corpusPath);
        }

        for (String corpusPath : devCorpusPaths)
        {
            preparator.addDevelopmentCorpus(corpusPath);
        }

        for (int i = 0; i < inputFeatures.size(); i++)
        {
            String inputFeatureAnnotationName = inputFeatures.get(i);
            String inputFeatureEmbeddings = inputEmbeddings.get(i).equals("null") ? null : inputEmbeddings.get(i);
            String inputFeatureVocabulary = inputVocabulary.get(i).equals("null") ? null : inputVocabulary.get(i);
            preparator.addInputFeature(inputFeatureAnnotationName, inputFeatureEmbeddings, inputFeatureVocabulary);
        }

        if (outputFeatures.size() == 1 && outputFeatures.get(0).equals("null"))
        {
            outputFeatures.clear();
        }

        for (int i = 0; i < outputFeatures.size(); i++)
        {
            preparator.addOutputFeature(outputFeatures.get(i), null);
        }

        preparator.setCorpusFormat(corpusFormat);
        preparator.setInputVocabularyLimit(inputVocabularyLimit);
        preparator.setInputClearText(inputClearText);
        preparator.setOutputFeatureVocabularyLimit(outputFeatureVocabularyLimit);

        preparator.maxLineLength = maxLineLength;
        preparator.lowercaseWords = lowercase;
        preparator.filterLemma = filterLemma;
        preparator.uniformDash = uniformDash;
        preparator.multisenses = false;
        preparator.removeAllCoarseGrained = true;
        preparator.addMonosemics = addMonosemics;
        preparator.removeMonosemics = removeMonosemics;
        preparator.reducedOutputVocabulary = senseCompressionClusters;
        preparator.additionalDevFromTrainSize = devFromTrain;
        preparator.removeDuplicateSentences = removeDuplicateSentences;

        preparator.prepareTrainingFile();
    }

    private static <T> List<T> padList(List<T> list, int padSize, T padValue)
    {
        List<T> newList = new ArrayList<>(list);
        while (newList.size() < padSize)
        {
            newList.add(padValue);
        }
        return newList;
    }
}

