import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.evaluation.WSDEvaluator;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.FirstSenseDisambiguator;
import getalp.wsd.method.MonosemicDisambiguator;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.method.result.DisambiguationResult;
import getalp.wsd.method.result.MultipleDisambiguationResult;
import getalp.wsd.ufsac.core.*;
import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

public class NeuralWSDTest
{
    private String pythonPath;

    private String dataPath;

    private List<String> weights;

    private List<String> testCorpusPaths;

    private boolean lowercase;

    private Map<String, String> senseCompressionClusters;

    private boolean filterLemma;

    private boolean clearText;

    private int batchSize;

    private Disambiguator monosemicDisambiguator;

    private Disambiguator firstSenseDisambiguator;

    private WSDEvaluator evaluator;

    private void test(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgumentList("corpus", Arrays.asList(
                "ufsac-public-2.1/raganato_senseval2.xml",
                "ufsac-public-2.1/raganato_senseval3.xml",
                "ufsac-public-2.1/raganato_semeval2007.xml",
                "ufsac-public-2.1/raganato_semeval2013.xml",
                "ufsac-public-2.1/raganato_semeval2015.xml",
                "ufsac-public-2.1/raganato_ALL.xml",
                "ufsac-public-2.1/semeval2007task7.xml"));
        parser.addArgument("lowercase", "false");
        parser.addArgument("sense_compression_hypernyms", "true");
        parser.addArgument("sense_compression_instance_hypernyms", "false");
        parser.addArgument("sense_compression_antonyms", "false");
        parser.addArgument("sense_compression_file", "");
        parser.addArgument("filter_lemma", "true");
        parser.addArgument("clear_text", "false");
        parser.addArgument("batch_size", "1");
        if (!parser.parse(args, true)) return;

        pythonPath = parser.getArgValue("python_path");
        dataPath = parser.getArgValue("data_path");
        weights = parser.getArgValueList("weights");
        testCorpusPaths = parser.getArgValueList("corpus");
        lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseCompressionHypernyms = parser.getArgValueBoolean("sense_compression_hypernyms");
        boolean senseCompressionInstanceHypernyms = parser.getArgValueBoolean("sense_compression_instance_hypernyms");
        boolean senseCompressionAntonyms = parser.getArgValueBoolean("sense_compression_antonyms");
        String senseCompressionFile = parser.getArgValue("sense_compression_file");
        filterLemma = parser.getArgValueBoolean("filter_lemma");
        clearText = parser.getArgValueBoolean("clear_text");
        batchSize = parser.getArgValueInteger("batch_size");

        senseCompressionClusters = null;
        if (senseCompressionHypernyms || senseCompressionAntonyms)
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClusters(WordnetHelper.wn30(), senseCompressionHypernyms, senseCompressionInstanceHypernyms, senseCompressionAntonyms);
        }
        if (!senseCompressionFile.isEmpty())
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClustersFromFile(senseCompressionFile);
        }

        monosemicDisambiguator = new MonosemicDisambiguator(WordnetHelper.wn30());
        firstSenseDisambiguator = new FirstSenseDisambiguator(WordnetHelper.wn30());

        evaluator = new WSDEvaluator();

        System.out.println();
        System.out.println("------ Evaluate the score of an ensemble of models");
        System.out.println();

        evaluate_ensemble();

        System.out.println();
        System.out.println("------ Evaluate the scores of individual models");
        System.out.println();

        evaluate_mean_scores();
    }

    private void evaluate_ensemble() throws Exception
    {
        NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize);
        neuralDisambiguator.lowercaseWords = lowercase;
        neuralDisambiguator.filterLemma = filterLemma;
        neuralDisambiguator.reducedOutputVocabulary = senseCompressionClusters;
        for (String testCorpusPath : testCorpusPaths)
        {
            System.out.println("Evaluate on corpus " + testCorpusPath);
            Corpus testCorpus = Corpus.loadFromXML(testCorpusPath);
            System.out.println("Evaluate without backoff");
            evaluator.evaluate(neuralDisambiguator, testCorpus, "wn30_key");
            System.out.println("Evaluate with monosemics");
            evaluator.evaluate(monosemicDisambiguator, testCorpus, "wn30_key");
            System.out.println("Evaluate with backoff first sense");
            /* DisambiguationResult result = */evaluator.evaluate(firstSenseDisambiguator, testCorpus, "wn30_key");
            System.out.println();
        }
        neuralDisambiguator.close();
    }

    private void evaluate_mean_scores() throws Exception
    {
        List<NeuralDisambiguator> neuralDisambiguators = new ArrayList<>();
        for (String weight : weights)
        {
            NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weight, clearText, batchSize);
            neuralDisambiguator.lowercaseWords = lowercase;
            neuralDisambiguator.filterLemma = filterLemma;
            neuralDisambiguator.reducedOutputVocabulary = senseCompressionClusters;
            neuralDisambiguators.add(neuralDisambiguator);
        }
        for (String testCorpusPath : testCorpusPaths)
        {
            System.out.println("Evaluate on corpus " + testCorpusPath);
            MultipleDisambiguationResult resultsBackoffZero = new MultipleDisambiguationResult();
            MultipleDisambiguationResult resultsBackoffMonosemics = new MultipleDisambiguationResult();
            MultipleDisambiguationResult resultsBackoffFirstSense = new MultipleDisambiguationResult();
            for (int i = 0; i < weights.size(); i++)
            {
                NeuralDisambiguator neuralDisambiguator = neuralDisambiguators.get(i);
                Corpus testCorpus = Corpus.loadFromXML(testCorpusPath);
                System.out.println("" + i + " Evaluate without backoff");
                DisambiguationResult resultBackoffZero = evaluator.evaluate(neuralDisambiguator, testCorpus, "wn30_key");
                System.out.println("" + i + " Evaluate with monosemics");
                DisambiguationResult resultBackoffMonosemics = evaluator.evaluate(monosemicDisambiguator, testCorpus, "wn30_key");
                System.out.println("" + i + " Evaluate with backoff first sense");
                DisambiguationResult resultBackoffFirstSense = evaluator.evaluate(firstSenseDisambiguator, testCorpus, "wn30_key");
                resultsBackoffZero.addDisambiguationResult(resultBackoffZero);
                resultsBackoffMonosemics.addDisambiguationResult(resultBackoffMonosemics);
                resultsBackoffFirstSense.addDisambiguationResult(resultBackoffFirstSense);
            }
            System.out.println();
            System.out.println("Mean of scores without backoff: " + resultsBackoffZero.scoreMean());
            System.out.println("Standard deviation without backoff: " + resultsBackoffZero.scoreStandardDeviation());
            System.out.println("Mean of scores with monosemics: " + resultsBackoffMonosemics.scoreMean());
            System.out.println("Standard deviation with monosemics: " + resultsBackoffMonosemics.scoreStandardDeviation());
            System.out.println("Mean of scores with backoff first sense: " + resultsBackoffFirstSense.scoreMean());
            System.out.println("Standard deviation with backoff first sense: " + resultsBackoffFirstSense.scoreStandardDeviation());
            System.out.println();
        }
        for (int i = 0; i < weights.size(); i++)
        {
            neuralDisambiguators.get(i).close();
        }
    }

    public static void main(String[] args) throws Exception
    {
        new NeuralWSDTest().test(args);
    }
}
