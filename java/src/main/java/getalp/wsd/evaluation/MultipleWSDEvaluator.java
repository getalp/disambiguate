package getalp.wsd.evaluation;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.result.MultipleDisambiguationResult;
import getalp.wsd.utils.ObjectUsingSystemOutALot;

public class MultipleWSDEvaluator extends ObjectUsingSystemOutALot
{
    private List<Disambiguator> inputs;
    
    private String testCorpusPath;
    
    private String senseAnnotationTag;
    
    private WSDEvaluator singleEvaluator;

    public MultipleWSDEvaluator()
    {
        inputs = new ArrayList<>();
        testCorpusPath = "";
        senseAnnotationTag = "";
        singleEvaluator = new WSDEvaluator();
    }
    
    public void setPrintFailed(boolean printFailed)
    {
        singleEvaluator.setPrintFailed(printFailed);
    }
    
    public void setTestCorpus(String corpusPath, String senseAnnotationTag)
    {
        this.testCorpusPath = corpusPath;
        this.senseAnnotationTag = senseAnnotationTag;
    }

    public void addDisambiguator(Disambiguator wsd)
    {
        inputs.add(wsd);
    }

    public void evaluate(int n)
    {
        MultipleDisambiguationResult[] res = new MultipleDisambiguationResult[inputs.size()];
        for (int i = 0 ; i < res.length ; i++)
        {
            res[i] = singleEvaluator.evaluate(inputs.get(i), testCorpusPath, senseAnnotationTag, n);
        }
        println("Recap :");
        for (int i = 0 ; i < res.length ; i++)
        {
            println("Test " + i + " (" + inputs.get(i) + ")");
            println("Mean Scores : " + res[i].scoreMean());
            println("Standard Deviation Scores : " + res[i].scoreStandardDeviation());
            println("Mean Times : " + res[i].timeMean());
            println();
        }
        MannWhitneyUTest mannTest = new MannWhitneyUTest();
        for (int i = 0 ; i < res.length ; i++)
        {
            for (int j = 0 ; j < res.length ; j++)
            {
                println("MWUTest " + i + " (" + inputs.get(i) + ") / " + j + " (" + inputs.get(j) + ") : " + mannTest.mannWhitneyUTest(res[i].allScores(), res[j].allScores()));
            }
        }
    }
}
