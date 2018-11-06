import java.util.Collections;
import java.util.List;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.neural.NeuralDataPreparator;
import getalp.wsd.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

public class NeuralWSDPrepare
{
    public static void main(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("data_path");
        parser.addArgumentList("train");
        parser.addArgumentList("dev");
        parser.addArgumentList("input_features", Collections.singletonList("surface_form"));
        parser.addArgumentList("output_features", Collections.singletonList("wn30_key"));
        parser.addArgument("lowercase", "true");
        parser.addArgument("uniform_dash", "false");
        parser.addArgument("dev_from_train", "0");
        parser.addArgument("sense_reduction", "true");
        parser.addArgument("add_monosemics", "false");
        parser.addArgument("remove_monosemics", "false");
        parser.addArgument("remove_duplicates", "true");
        if (!parser.parse(args, true)) return;

        String dataPath = parser.getArgValue("data_path");
        List<String> trainingCorpusPaths = parser.getArgValueList("train");
        List<String> devCorpusPaths = parser.getArgValueList("dev");
        List<String> inputFeatures = parser.getArgValueList("input_features");
        List<String> outputFeatures = parser.getArgValueList("output_features");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean uniformDash = parser.getArgValueBoolean("uniform_dash");
        int devFromTrain = parser.getArgValueInteger("dev_from_train");
        boolean senseReduction = parser.getArgValueBoolean("sense_reduction");
        boolean addMonosemics = parser.getArgValueBoolean("add_monosemics");
        boolean removeMonosemics = parser.getArgValueBoolean("remove_monosemics");
        boolean removeDuplicateSentences = parser.getArgValueBoolean("remove_duplicates");

        NeuralDataPreparator preparator = new NeuralDataPreparator();

        preparator.setOutputDirectoryPath(dataPath);

        for (String corpusPath : trainingCorpusPaths)
        {
            preparator.addTrainingCorpus(corpusPath);
        }

        for (String corpusPath : devCorpusPaths)
        {
            preparator.addDevelopmentCorpus(corpusPath);
        }

        for (int i = 0 ; i < inputFeatures.size() ; i += 2)
        {
            String inputFeatureAnnotationName = inputFeatures.get(i);
            String inputFeatureEmbeddings = "null";
            if (i + 1 < inputFeatures.size())
            {
                inputFeatureEmbeddings = inputFeatures.get(i + 1);
            }
            if (inputFeatureEmbeddings.equals("null"))
            {
                preparator.addInputFeature(inputFeatureAnnotationName);
            }
            else
            {
                preparator.addInputFeature(inputFeatureAnnotationName, inputFeatureEmbeddings);
            }
        }

        for (int i = 0 ; i < outputFeatures.size() ; i += 2)
        {
            String outputFeatureAnnotationName = outputFeatures.get(i);
            String outputFeatureVocabulary = "null";
            if (i + 1 < outputFeatures.size())
            {
                outputFeatureVocabulary = outputFeatures.get(i + 1);
            }
            if (outputFeatureVocabulary.equals("null"))
            {
                preparator.addOutputFeature(outputFeatureAnnotationName);
            }
            else
            {
                preparator.addOutputFeature(outputFeatureAnnotationName, outputFeatureVocabulary);
            }
        }

        preparator.maxLineLength = 80;
        preparator.lowercaseWords = lowercase;
        preparator.uniformDash = uniformDash;
        preparator.multisenses = false;
        preparator.removeAllCoarseGrained = true;
        preparator.addMonosemics = addMonosemics;
        preparator.removeMonosemics = removeMonosemics;
        if (senseReduction) preparator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
        else preparator.reducedOutputVocabulary = null;
        preparator.additionalDevFromTrainSize = devFromTrain;
        preparator.removeDuplicateSentences = removeDuplicateSentences;

        preparator.prepareTrainingFile();
    }
}

