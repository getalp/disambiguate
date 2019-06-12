package getalp.wsd.method.result;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DisambiguationResult
{
    public int total;
    
    public int good;
    
    public int bad;

    public double time;

    public Map<String, Integer> totalPerPOS = initMapPerPos();

    public Map<String, Integer> goodPerPOS = initMapPerPos();

    public Map<String, Integer> badPerPOS = initMapPerPos();

    public DisambiguationResult()
    {
        this(0, 0, 0);
    }

    public DisambiguationResult(int total, int good, int bad)
    {
        this(total, good, bad, initMapPerPos(), initMapPerPos(), initMapPerPos());
    }

    public DisambiguationResult(int total, int good, int bad, Map<String, Integer> totalPerPOS, Map<String, Integer> goodPerPOS, Map<String, Integer> badPerPOS)
    {
        this.total = total;
        this.good = good;
        this.bad = bad;
        this.totalPerPOS = totalPerPOS;
        this.goodPerPOS = goodPerPOS;
        this.badPerPOS = badPerPOS;
    }

    public void concatenateResult(DisambiguationResult other)
    {
        total += other.total;
        good += other.good;
        bad += other.bad;
        for (String pos : Arrays.asList("n", "v", "a", "r", "x"))
        {
            totalPerPOS.put(pos, totalPerPOS.get(pos) + other.totalPerPOS.get(pos));
            goodPerPOS.put(pos, goodPerPOS.get(pos) + other.goodPerPOS.get(pos));
            badPerPOS.put(pos, badPerPOS.get(pos) + other.badPerPOS.get(pos));
        }
    }

    public int attempted()
    {
        return good + bad;
    }

    public int missed()
    {
        return total - attempted();
    }

    public double coverage()
    {
        return ratioPercent(attempted(), total);
    }

    public double scoreRecall()
    {
        return ratioPercent(good, total);
    }

    public double scorePrecision()
    {
        return ratioPercent(good, attempted());
    }

    public double scoreF1()
    {
        double r = scoreRecall();
        double p = scorePrecision();
        return 2.0 * ((p * r) / (p + r));
    }

    public int attemptedPerPOS(String pos)
    {
        return goodPerPOS.get(pos) + badPerPOS.get(pos);
    }

    public int missedPerPOS(String pos)
    {
        return totalPerPOS.get(pos) - attemptedPerPOS(pos);
    }

    public double coveragePerPOS(String pos)
    {
        return ratioPercent(attemptedPerPOS(pos), totalPerPOS.get(pos));
    }

    public double scoreRecallPerPOS(String pos)
    {
        return ratioPercent(goodPerPOS.get(pos), totalPerPOS.get(pos));
    }

    public double scorePrecisionPerPOS(String pos)
    {
        return ratioPercent(goodPerPOS.get(pos), attemptedPerPOS(pos));
    }

    public double scoreF1PerPOS(String pos)
    {
        double r = scoreRecallPerPOS(pos);
        double p = scorePrecisionPerPOS(pos);
        return 2.0 * ((p * r) / (p + r));
    }

    private double ratioPercent(double num, double den)
    {
        return (num / den) * 100;
    }

    private static Map<String, Integer> initMapPerPos()
    {
        Map<String, Integer> map = new HashMap<>();
        for (String pos : Arrays.asList("n", "v", "a", "r", "x"))
        {
            map.put(pos, 0);
        }
        return map;
    }
}