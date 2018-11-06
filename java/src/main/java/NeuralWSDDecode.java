import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.utils.CorpusPOSTaggerAndLemmatizer;
import getalp.wsd.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;

public class NeuralWSDDecode
{
    public static void main(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgument("lowercase", "true");
        parser.addArgument("sense_reduction", "true");
        if (!parser.parse(args)) return;

        String pythonPath = parser.getArgValue("python_path");
        String dataPath = parser.getArgValue("data_path");
        List<String> weights = parser.getArgValueList("weights");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseReduction = parser.getArgValueBoolean("sense_reduction");

        CorpusPOSTaggerAndLemmatizer tagger = new CorpusPOSTaggerAndLemmatizer();
        NeuralDisambiguator disambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights);
        disambiguator.lowercaseWords = lowercase;
        if (senseReduction) disambiguator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
        else disambiguator.reducedOutputVocabulary = null;

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out));
        for (String line = reader.readLine() ; line != null ; line = reader.readLine())
        {
            Sentence sentence = new Sentence(line);
            tagger.tag(sentence.getWords());
            disambiguator.disambiguate(sentence, "wsd");
            for (Word word : sentence.getWords())
            {
                writer.write(word.getValue().replace("|", ""));
                if (word.hasAnnotation("lemma") && word.hasAnnotation("pos") && word.hasAnnotation("wsd"))
                {
                    writer.write("|" + word.getAnnotationValue("wsd"));
                }
                writer.write(" ");
            }
            writer.newLine();
            writer.flush();
        }
        writer.close();
        reader.close();
        disambiguator.close();
    }
}

