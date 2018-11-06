package getalp.wsd.embeddings;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class TextualModelSaver
{
    public void save(WordVectors wordVectors, String modelPath)
    {
        try
        {
            saveNoCatch(wordVectors, modelPath);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }
    
    private void saveNoCatch(WordVectors wordVectors, String modelPath) throws IOException
    {
        PrintWriter pw = new PrintWriter(new FileWriter(modelPath));
        for (String word : wordVectors.getVocabulary())
        {
            double[] vector = wordVectors.getWordVector(word);
            pw.print(word);
            for (double scalar : vector)
            {
                pw.print(" " + scalar);
            }
            pw.println();
        }
        pw.close();
    }
}
