package getalp.wsd.utils.tuples;


public class Triplet<FirstType, SecondType, ThirdType>
{
    public FirstType first;
    
    public SecondType second;
    
    public ThirdType third;
    
    public Triplet(FirstType first, SecondType second, ThirdType third)
    {
        this.first = first;
        this.second = second;
        this.third = third;
    }
}
