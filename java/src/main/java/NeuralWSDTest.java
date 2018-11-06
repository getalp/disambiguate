import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.evaluation.WSDEvaluator;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.FirstSenseDisambiguator;
import getalp.wsd.method.MonosemicDisambiguator;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.method.result.DisambiguationResult;
import getalp.wsd.method.result.MultipleDisambiguationResult;
import getalp.wsd.ufsac.core.*;
import getalp.wsd.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

public class NeuralWSDTest
{
    private String pythonPath;

    private String dataPath;

    private List<String> weights;

    private List<String> testCorpusPaths;

    private boolean lowercase;

    private boolean senseReduction;

    private Disambiguator monosemicDisambiguator;

    private Disambiguator firstSenseDisambiguator;

    private WSDEvaluator evaluator;

    public void test(String[] args) throws Exception
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
        parser.addArgument("lowercase", "true");
        parser.addArgument("sense_reduction", "true");
        if (!parser.parse(args, true)) return;

        pythonPath = parser.getArgValue("python_path");
        dataPath = parser.getArgValue("data_path");
        weights = parser.getArgValueList("weights");
        testCorpusPaths = parser.getArgValueList("corpus");
        lowercase = parser.getArgValueBoolean("lowercase");
        senseReduction = parser.getArgValueBoolean("sense_reduction");

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
        NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights);
        neuralDisambiguator.lowercaseWords = lowercase;
        if (senseReduction)
        {
            neuralDisambiguator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
        }
        else
        {
            neuralDisambiguator.reducedOutputVocabulary = null;
        }
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
        for (int i = 0; i < weights.size(); i++)
        {
            neuralDisambiguators.add(new NeuralDisambiguator(pythonPath, dataPath, weights.get(i)));
        }
        for (String testCorpusPath : testCorpusPaths)
        {
            System.out.println("Evaluate on corpus " + testCorpusPath);
            MultipleDisambiguationResult results = new MultipleDisambiguationResult();
            for (int i = 0; i < weights.size(); i++)
            {
                NeuralDisambiguator neuralDisambiguator = neuralDisambiguators.get(i);
                neuralDisambiguator.lowercaseWords = lowercase;
                if (senseReduction)
                {
                    neuralDisambiguator.reducedOutputVocabulary = WordnetUtils.getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
                }
                else
                {
                    neuralDisambiguator.reducedOutputVocabulary = null;
                }
                Corpus testCorpus = Corpus.loadFromXML(testCorpusPath);
                System.out.println("" + i + " Evaluate without backoff");
                evaluator.evaluate(neuralDisambiguator, testCorpus, "wn30_key");
                System.out.println("" + i + " Evaluate with monosemics");
                evaluator.evaluate(monosemicDisambiguator, testCorpus, "wn30_key");
                System.out.println("" + i + " Evaluate with backoff first sense");
                DisambiguationResult result = evaluator.evaluate(firstSenseDisambiguator, testCorpus, "wn30_key");
                results.addDisambiguationResult(result);
            }
            System.out.println("Mean Scores : " + results.scoreMean());
            System.out.println("Standard Deviation Scores : " + results.scoreStandardDeviation());
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
