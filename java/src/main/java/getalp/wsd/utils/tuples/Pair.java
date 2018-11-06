package getalp.wsd.utils.tuples;

public class Pair<FirstType, SecondType>
{
    public FirstType first;
    
    public SecondType second;
    
    public Pair(FirstType first, SecondType second)
    {
        this.first = first;
        this.second = second;
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if (this == obj) return true;
        if (obj == null) return false;
        if (!(obj instanceof Pair<?, ?>)) return false;
        Pair<?, ?> other = (Pair<?, ?>) obj;
        if (first == null && other.first != null) return false;
        if (first != null && !first.equals(other.first)) return false;
        if (second == null && other.second != null) return false;
        if (second != null && !second.equals(other.second)) return false;
        return true;
    }
}
