package getalp.wsd.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.utils.RegExp;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.streaming.reader.StreamingCorpusReaderWord;

public class Util
{
    /**
     * Trims, Lowercases, and Removes all non-ASCII and non-letters characters
     */
    public static String normalize(String str)
    {
        str = str.trim().toLowerCase();
        str = RegExp.nonLetterPattern.matcher(str).replaceAll("");
        return str;
    }
    
    public static String getWordKeyOfSenseKey(String senseKey)
    {
        String lemma = senseKey.substring(0, senseKey.indexOf("%"));
        int pos = Integer.valueOf(senseKey.substring(senseKey.indexOf("%") + 1, senseKey.indexOf("%") + 2));
        return lemma + "%" + POSConverter.toWNPOS(pos);
    }

    public static void writeFifo(String fifoPath) throws IOException
    {
        Files.deleteIfExists(Paths.get(fifoPath));
        Runtime.getRuntime().exec("mkfifo " + fifoPath);
    }

    public static List<String> getVocabularyOfCorpus(List<String> corpusPaths, WordnetHelper wn)
    {
        Set<String> vocab = new HashSet<>();
        
        StreamingCorpusReaderWord reader = new StreamingCorpusReaderWord()
        {
            @Override
            public void readWord(Word w)
            {
                List<String> allPOS = new ArrayList<>();
                if (w.hasAnnotation("pos"))
                {
                    String pos = POSConverter.toWNPOS(w.getAnnotationValue("pos"));
                    allPOS.add(pos);
                }
                else
                {
                    allPOS.addAll(Arrays.asList("n", "v", "a", "r"));
                }
                for (String lemma : w.getAnnotationValues("lemma", ";"))
                {
                    for (String pos : allPOS)
                    {
                        String wordKey = lemma + "%" + pos;
                        if (!vocab.contains(wordKey) && wn.getVocabulary().contains(wordKey))
                        {
                            vocab.add(wordKey);
                        }
                    }
                }
            }
        };

        for (String corpusPath : corpusPaths)
        {
            reader.load(corpusPath);
        }
        
        return new ArrayList<>(vocab);
    }
}
