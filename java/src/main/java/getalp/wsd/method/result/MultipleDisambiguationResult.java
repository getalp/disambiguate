package getalp.wsd.method.result;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import com.google.common.math.DoubleMath;

public class MultipleDisambiguationResult
{
    private List<DisambiguationResult> results = new ArrayList<>();
    
    public MultipleDisambiguationResult()
    {
        
    }
    
    public void addDisambiguationResult(DisambiguationResult result)
    {
        results.add(result);
    }
    
    public double scoreMean()
    {
        return DoubleMath.mean(allScores());
    }
    
    public double scoreStandardDeviation()
    {
        return new StandardDeviation().evaluate(allScores(), scoreMean());
    }
    
    public double timeMean()
    {
        return DoubleMath.mean(allTimes());
    }
    
    public double[] allScores()
    {
        return results.stream().mapToDouble((DisambiguationResult r) -> {return r.scoreF1();}).toArray();
    }
    
    public double[] allTimes()
    {
        return results.stream().mapToDouble((DisambiguationResult r) -> {return r.time;}).toArray();
    }
}