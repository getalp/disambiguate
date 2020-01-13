import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.FirstSenseDisambiguator;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.streaming.modifier.StreamingCorpusModifierSentence;
import getalp.wsd.ufsac.utils.CorpusPOSTaggerAndLemmatizer;
import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

import java.util.List;

public class NeuralWSDDecodeUFSAC
{
    public static void main(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgument("input");
        parser.addArgument("output");
        parser.addArgument("lowercase", "false");
        parser.addArgument("sense_reduction", "true");
        parser.addArgument("clear_text", "true");
        parser.addArgument("batch_size", "1");
        parser.addArgument("mfs_backoff", "true");
        if (!parser.parse(args)) return;

        String pythonPath = parser.getArgValue("python_path");
        String dataPath = parser.getArgValue("data_path");
        List<String> weights = parser.getArgValueList("weights");
        String inputPath = parser.getArgValue("input");
        String outputPath = parser.getArgValue("output");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseReduction = parser.getArgValueBoolean("sense_reduction");
        boolean clearText = parser.getArgValueBoolean("clear_text");
        int batchSize = parser.getArgValueInteger("batch_size");
        boolean mfsBackoff = parser.getArgValueBoolean("mfs_backoff");

        CorpusPOSTaggerAndLemmatizer tagger = new CorpusPOSTaggerAndLemmatizer();
        Disambiguator firstSenseDisambiguator = new FirstSenseDisambiguator(WordnetHelper.wn30());
        NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize);
        neuralDisambiguator.lowercaseWords = lowercase;
        if (senseReduction) neuralDisambiguator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
        else neuralDisambiguator.reducedOutputVocabulary = null;

        StreamingCorpusModifierSentence modifier = new StreamingCorpusModifierSentence()
        {
            public void modifySentence(Sentence sentence)
            {
                tagger.tag(sentence.getWords());
                neuralDisambiguator.disambiguate(sentence, "wsd");
                if (mfsBackoff)
                {
                    firstSenseDisambiguator.disambiguate(sentence, "wsd");
                }
            }
        };

        modifier.load(inputPath, outputPath);
        neuralDisambiguator.close();
    }
}

