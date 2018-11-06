package getalp.wsd.method.result;

public class DisambiguationResult
{
    public int total;
    
    public int good;
    
    public int bad;

    public int attempted;
    
    public int missed;
    
    public double time;
    
    public DisambiguationResult()
    {
        this(0, 0, 0);
    }

    public DisambiguationResult(int total, int good, int bad)
    {
        this.total = total;
        this.good = good;
        this.bad = bad;
        this.attempted = good + bad;
        this.missed = total - attempted;
    }

    public void concatenateResult(DisambiguationResult other)
    {
        total += other.total;
        good += other.good;
        bad += other.bad;
        attempted += other.attempted;
        missed += other.missed;
    }
    
    public double coverage()
    {
        return ratioPercent(total - missed, total);
    }

    public double scoreRecall()
    {
        return ratioPercent(good, total);
    }

    public double scorePrecision()
    {
        return ratioPercent(good, total - missed);
    }

    public double scoreF1()
    {
        double r = scoreRecall();
        double p = scorePrecision();
        return 2.0 * ((p * r) / (p + r));
    }

    private double ratioPercent(double num, double den)
    {
        return (num / den) * 100;
    }
}