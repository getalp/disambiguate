package getalp.wsd.utils;

public class ObjectUsingSystemOutALot
{
    protected void print(String str)
    {
        System.out.print(str);
        System.out.flush();
    }
    
    protected void println(String str)
    {
        System.out.println(str);
    }
    
    protected void println()
    {
        System.out.println();
    }
}
