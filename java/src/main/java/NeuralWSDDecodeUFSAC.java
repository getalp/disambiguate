import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.streaming.modifier.StreamingCorpusModifierSentence;
import getalp.wsd.ufsac.streaming.reader.StreamingCorpusReaderSentence;
import getalp.wsd.ufsac.streaming.writer.StreamingCorpusWriterSentence;
import getalp.wsd.ufsac.utils.CorpusPOSTaggerAndLemmatizer;
import getalp.wsd.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;
import getalp.wsd.common.utils.Wrapper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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
        parser.addArgument("lowercase", "true");
        parser.addArgument("sense_reduction", "true");
        parser.addArgument("lemma_pos_tagged", "false");
        if (!parser.parse(args)) return;

        String pythonPath = parser.getArgValue("python_path");
        String dataPath = parser.getArgValue("data_path");
        List<String> weights = parser.getArgValueList("weights");
        String inputPath = parser.getArgValue("input");
        String outputPath = parser.getArgValue("output");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseReduction = parser.getArgValueBoolean("sense_reduction");
        boolean lemmaPOSTagged = parser.getArgValueBoolean("lemma_pos_tagged");

        Wrapper<CorpusPOSTaggerAndLemmatizer> lemmaPOSTagger = new Wrapper<>(null);
        if (!lemmaPOSTagged)
        {
            lemmaPOSTagger.obj = new CorpusPOSTaggerAndLemmatizer();
        }
        NeuralDisambiguator disambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights);
        disambiguator.lowercaseWords = lowercase;
        if (senseReduction) disambiguator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
        else disambiguator.reducedOutputVocabulary = null;

        StreamingCorpusModifierSentence modifier = new StreamingCorpusModifierSentence()
        {
            public void modifySentence(Sentence sentence)
            {
                if (lemmaPOSTagger.obj != null)
                {
                    lemmaPOSTagger.obj.tag(sentence.getWords());
                }
                disambiguator.disambiguate(sentence, "wsd");
            }
        };

        modifier.load(inputPath, outputPath);
        disambiguator.close();
    }
}

