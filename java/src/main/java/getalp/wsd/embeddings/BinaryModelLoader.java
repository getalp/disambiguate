package getalp.wsd.embeddings;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import getalp.wsd.utils.ByteOperations;

public class BinaryModelLoader
{
    private boolean skipEOLChar;
    
    private boolean verbose;
    
    public BinaryModelLoader()
    {
        this(true);
    }

    public BinaryModelLoader(boolean verbose)
    {
        this(false, verbose);
    }

    public BinaryModelLoader(boolean skipEndOfLineChar, boolean verbose)
    {
        this.skipEOLChar = skipEndOfLineChar;
        this.verbose = verbose;
    }
    
    public WordVectors load(String modelPath)
    {
        try
        {
            return loadNoCatch(modelPath);
        } 
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    private WordVectors loadNoCatch(String modelPath) throws IOException
    {
        FileInputStream fis = new FileInputStream(modelPath);
        BufferedInputStream bis = new BufferedInputStream(fis);
        DataInputStream dis = new DataInputStream(bis);
        int vectorCount = Integer.parseInt(readString(dis));
        int vectorSize = Integer.parseInt(readString(dis));
        double[][] vectors = new double[vectorCount][vectorSize];
        String[] words = new String[vectorCount];
        Map<String, Integer> wordsIndexes = new HashMap<>();
        int lastPercentage = 0;
        if (verbose)
        {
            System.out.println("Loading " + vectorCount + " vectors of size " + vectorSize + " from " + modelPath);
        }
        for (int i = 0 ; i < vectorCount ; i++) 
        {
            if (verbose)
            {
                int currentPercentage = ((int) ((((double) (i + 1)) / ((double) (vectorCount))) * 100.0));
                if (currentPercentage > lastPercentage) System.out.print("Loading vectors... (" + currentPercentage + "%)\r");
                lastPercentage = currentPercentage;
            }
            words[i] = readString(dis);
            if (words[i].isEmpty())
            {
                for (int j = 0 ; j < vectorSize ; j++)
                {
                    vectors[i][j] = ByteOperations.readFloat(dis);
                    vectors[i][j] = 0;
                }
                if (skipEOLChar)
                {
                    dis.readByte();
                }
            }
            else
            {
                wordsIndexes.put(words[i], i);
                for (int j = 0 ; j < vectorSize ; j++)
                {
                    vectors[i][j] = ByteOperations.readFloat(dis);
                }
                if (skipEOLChar)
                {
                    dis.readByte();
                }
                if (VectorOperation.norm(vectors[i]) != 0)
                {
                    vectors[i] = VectorOperation.normalize(vectors[i]);
                }
            }
        }
        if (verbose)
        {
            System.out.println();
        }
        return new WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes);
    }

    private static String readString(DataInputStream dis) throws IOException
    {
        final int buffer_size = 50;
        byte[] bytes = new byte[buffer_size];
        byte b = dis.readByte();
        int i = -1;
        StringBuilder sb = new StringBuilder();
        while (b != ' ' && b != '\n' && b != '\t' && b != '\0') 
        {
            i++;
            bytes[i] = b;
            b = dis.readByte();
            if (i == buffer_size - 1) 
            {
                sb.append(new String(bytes));
                i = -1;
                bytes = new byte[buffer_size];
            }
        }
        sb.append(new String(bytes, 0, i + 1));
        return sb.toString();
    }
}
