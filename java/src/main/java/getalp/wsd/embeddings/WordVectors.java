package getalp.wsd.embeddings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import getalp.wsd.utils.tuples.Pair;

public class WordVectors
{
    private int vectorCount;
    
    private int vectorSize;
    
    private double[][] vectors;
    
    private String[] words;
    
    private Map<String, Integer> wordsIndexes;

    public WordVectors(List<Pair<String, double[]>> vectors)
    {
        this.vectorCount = vectors.size();
        this.vectorSize = vectors.get(0).second.length;
        this.vectors = new double[vectorCount][vectorSize];
        this.words = new String[vectorCount];
		this.wordsIndexes = new HashMap<>();
    	for (int i = 0 ; i < vectors.size() ; i++)
    	{
    		this.vectors[i] = vectors.get(i).second;
    		this.words[i] = vectors.get(i).first;
    		this.wordsIndexes.put(this.words[i], i);
    	}
    }

    public WordVectors(int vectorCount, int vectorSize, double[][] vectors, String[] words, Map<String, Integer> wordsIndexes)
    {
        this.vectorCount = vectorCount;
        this.vectorSize = vectorSize;
        this.vectors = vectors;
        this.words = words;
        this.wordsIndexes = wordsIndexes;
    }

    public int getVectorSize()
    {
        return vectorSize;
    }
    
    public String[] getVocabulary()
    {
        return words;
    }
    
    public boolean hasWordVector(String word)
    {
        return wordsIndexes.containsKey(word);
    }
    
    public double[] getWordVector(String word)
    {
        return vectors[wordsIndexes.get(word)];
    }
    
    public int getWordVectorIndex(String word)
    {
        return wordsIndexes.get(word);
    }

    public List<String> getMostSimilarWords(String word, double threshold) 
    {
        if (!wordsIndexes.containsKey(word)) return new ArrayList<>();
        return getMostSimilarWords(vectors[wordsIndexes.get(word)], threshold);
    }

    public List<String> getMostSimilarWords(String word, int topN) 
    {
        if (!wordsIndexes.containsKey(word)) return new ArrayList<>();
        return getMostSimilarWords(vectors[wordsIndexes.get(word)], topN);
    }

    public List<String> getMostSimilarWords(double[] word, int topN) 
    {
        Stuff[] zenearests = new Stuff[topN];
        for (int i = 0 ; i < topN ; i++) zenearests[i] = new Stuff(-Double.MAX_VALUE, 0);
        for (int j = 0 ; j < vectorCount ; j++) 
        {
            double[] v = vectors[j];
            double sim = VectorOperation.dot_product(word, v);
            if (sim > zenearests[0].sim) 
            {
                zenearests[0].sim = sim; 
                zenearests[0].index = j;
                Arrays.sort(zenearests);
            }
        }
        List<String> zenearestsstr = new ArrayList<>(topN);
        for (int i = topN - 1 ; i >= 0 ; i--) 
        {
            zenearestsstr.add(words[zenearests[i].index]);
        }
        return zenearestsstr;
    }
    
    public List<String> getMostSimilarWords(double[] word) 
    {
        Integer[] indexes = getMostSimilarWordIndexes(word);
        List<String> zenearestsstr = new ArrayList<>(vectorCount);
        for (int i = 0 ; i < vectorCount ; ++i)
        {
            zenearestsstr.add(words[indexes[i]]);
        }
        return zenearestsstr;
    }
    
    public Integer[] getMostSimilarWordIndexes(double[] word) 
    {
        Integer[] indexes = new Integer[vectorCount];
        double[] sims = new double[vectorCount];
        for (int i = 0 ; i < vectorCount ; ++i)
        {
            indexes[i] = i;
            sims[i] = VectorOperation.dot_product(word, vectors[i]);
        }
        Arrays.parallelSort(indexes, (a, b) -> Double.compare(sims[b], sims[a]));
        return indexes;
    }

    public List<String> getMostSimilarWords(double[] word, double threshold) 
    {
        List<String> similarWords = new ArrayList<>();
        for (int i = 0 ; i < vectorCount ; i++) 
        {
            double sim = VectorOperation.dot_product(word, vectors[i]);
            if (sim > threshold) 
            {
                similarWords.add(words[i]);
            }
        }
        return similarWords;
    }

    public List<Pair<String, Double>> getSimilarWordsAndSimilarity(String word) 
    {
        return getSimilarWordsAndSimilarity(vectors[wordsIndexes.get(word)]);
    }

    public List<Pair<String, Double>> getSimilarWordsAndSimilarity(double[] word) 
    {
        List<Pair<String, Double>> similarWords = new ArrayList<>();
        for (int i = 0 ; i < vectorCount ; i++) 
        {
            similarWords.add(new Pair<>(words[i], VectorOperation.dot_product(word, vectors[i])));
        }
        return similarWords;
    }

    public List<Pair<String, Double>> getMostSimilarWordsAndSimilarity(double[] word, double threshold) 
    {
        List<Pair<String, Double>> similarWords = new ArrayList<>();
        for (int i = 0 ; i < vectorCount ; i++) 
        {
            double sim = VectorOperation.dot_product(word, vectors[i]);
            if (sim > threshold) 
            {
                similarWords.add(new Pair<>(words[i], sim));
            }
        }
        return similarWords;
    }

    private static class Stuff implements Comparable<Stuff> 
    {
        public Double sim;
        public Integer index;
        public Stuff(double sim, int index) 
        {
            this.sim = sim;
            this.index = index;
        }
        public int compareTo(Stuff o) 
        {
            return sim.compareTo(o.sim);
        }
    }
    
}
