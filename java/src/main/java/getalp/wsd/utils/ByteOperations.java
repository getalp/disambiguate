package getalp.wsd.utils;

import java.io.IOException;
import java.io.InputStream;

public class ByteOperations
{
    public static float readFloat(InputStream is) throws IOException
    {
        byte[] bytes = new byte[4];
        is.read(bytes);
        return getFloat(bytes);
    }
    
    public static float getFloat(byte[] b)
    {
        return getFloat(b, 0);
    }
    
    public static float getFloat(byte[] b, int offset)
    {
        int accum = 0;
        accum = accum | (b[offset + 0] & 0xff) << 0;
        accum = accum | (b[offset + 1] & 0xff) << 8;
        accum = accum | (b[offset + 2] & 0xff) << 16;
        accum = accum | (b[offset + 3] & 0xff) << 24;
        return Float.intBitsToFloat(accum);
    }
    
    public static float[][] getFloatMatrix(byte[] b, int rowCount, int colCount)
    {
        float[][] ret = new float[rowCount][colCount];
        for (int i = 0 ; i < rowCount ; i++)
        {
            for (int j = 0 ; j < colCount ; j++)
            {
                ret[i][j] = ByteOperations.getFloat(b, ((i * colCount) + j) * 4);
            }
        }
        return ret;
    }
}
