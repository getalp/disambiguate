package getalp.wsd.embeddings;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import getalp.wsd.common.utils.RegExp;
import getalp.wsd.common.utils.Wrapper;

public class TextualModelLoader
{
    private boolean verbose;
    
    private boolean skipFirstLine;
    
    public TextualModelLoader()
    {
        this(false);
    }
    
    public TextualModelLoader(boolean verbose)
    {
        this(false, verbose);
    }
    
    public TextualModelLoader(boolean skipFirstLine, boolean verbose)
    {
        this.verbose = verbose;
        this.skipFirstLine = skipFirstLine;
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

    public WordVectors loadVocabularyOnly(String modelPath)
    {
        try
        {
            return loadVocabularyOnlyNoCatch(modelPath);
        } 
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    private WordVectors loadNoCatch(String modelPath) throws IOException
    {
        Wrapper<Integer> i = new Wrapper<>(0);
        Wrapper<Integer> j = new Wrapper<>(0);
        
        BufferedReader reader = Files.newBufferedReader(Paths.get(modelPath));
        
        if (skipFirstLine)
        {
            reader.readLine();
        }
        
        reader.lines().forEach(line ->
        {
            if (verbose) System.out.print("Loading Word Vectors... (count " + (i.obj + 1) + ")\r");
            String[] lineSplitted = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
            j.obj = lineSplitted.length - 1;
            i.obj++;
        });
        
        reader.close();
        
        if (verbose) System.out.println();
        int vectorCount = i.obj;
        int vectorSize = j.obj;
        double[][] vectors = new double[vectorCount][vectorSize];
        String[] words = new String[vectorCount];
        Map<String, Integer> wordsIndexes = new HashMap<>();
        i.obj = 0;
        
        reader = Files.newBufferedReader(Paths.get(modelPath));
        
        if (skipFirstLine)
        {
            reader.readLine();
        }
        
        reader.lines().forEach(line ->
        {
            if (verbose) System.out.print("Loading Word Vectors... (" + (i.obj + 1) + "/" + vectorCount + ")\r");
            String[] lineSplitted = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
            words[i.obj] = lineSplitted[0];
            wordsIndexes.put(words[i.obj], i.obj);
            for (int k = 1 ; k < lineSplitted.length ; k++)
            {
                vectors[i.obj][k - 1] = Double.valueOf(lineSplitted[k]);
            }
            i.obj++;
        });
        
        reader.close();
        
        if (verbose) System.out.println();
        return new WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes);
    }

    private WordVectors loadVocabularyOnlyNoCatch(String modelPath) throws IOException
    {
        Wrapper<Integer> i = new Wrapper<>(0);
        
        BufferedReader reader = Files.newBufferedReader(Paths.get(modelPath));
        reader.lines().forEach(line ->
        {
            if (verbose)  System.out.print("Loading Word Vectors... (count " + i.obj + ")\r");
            i.obj++;
        });
        reader.close();
        
        if (verbose) System.out.println();
        int vectorCount = i.obj;
        int vectorSize = 0;
        double[][] vectors = new double[vectorCount][vectorSize];
        String[] words = new String[vectorCount];
        Map<String, Integer> wordsIndexes = new HashMap<>();
        i.obj = 0;
        
        reader = Files.newBufferedReader(Paths.get(modelPath));
        reader.lines().forEach(line ->
        {
            if (verbose) System.out.print("Loading Word Vectors... (" + i.obj + "/" + vectorCount + ")\r");
            words[i.obj] = RegExp.anyWhiteSpaceGrouped.splitAsStream(line).findFirst().get();
            wordsIndexes.put(words[i.obj], i.obj);
            i.obj++;
        });
        reader.close();
        
        if (verbose) System.out.println();
        return new WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes);
    }
}
