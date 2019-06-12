package getalp.wsd.utils;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.streaming.reader.StreamingCorpusReaderWord;

public class WordnetUtils
{
    public static String extractLemmaFromSenseKey(String senseKey)
    {
        return senseKey.substring(0, senseKey.indexOf("%"));
    }

    public static String extractPOSFromSenseKey(String senseKey)
    {
        return POSConverter.toWNPOS(Integer.valueOf(senseKey.substring(senseKey.indexOf("%") + 1, senseKey.indexOf("%") + 2)));
    }

    public static Set<String> getUniqueSynsetKeysFromSenseKeys(WordnetHelper wn, List<String> senseKeys)
    {
        Set<String> synsetKeys = new HashSet<>();
        for (String senseKey : senseKeys)
        {
            synsetKeys.add(wn.getSynsetKeyFromSenseKey(senseKey));
        }
        return synsetKeys;
    }

    public static void getHypernymHierarchy(WordnetHelper wn, String synsetKey, List<String> hypernymyHierarchy)
    {
        if (hypernymyHierarchy.contains(synsetKey)) return;
        hypernymyHierarchy.add(synsetKey);
        List<String> hypernymSynsetKeys = wn.getHypernymSynsetKeysFromSynsetKey(synsetKey);
        if (!hypernymSynsetKeys.isEmpty())
        {
            getHypernymHierarchy(wn, hypernymSynsetKeys.get(0), hypernymyHierarchy);
        }
    }

    public static List<String> getHypernymHierarchy(WordnetHelper wn, String synsetKey)
    {
        List<String> hypernymyHierarchy = new ArrayList<>();
        getHypernymHierarchy(wn, synsetKey, hypernymyHierarchy);
        return hypernymyHierarchy;
    }

    public static void getHypernymHierarchyIncludeInstanceHypernyms(WordnetHelper wn, String synsetKey, List<String> hypernymyHierarchy)
    {
        if (hypernymyHierarchy.contains(synsetKey)) return;
        hypernymyHierarchy.add(synsetKey);
        List<String> hypernymSynsetKeys = wn.getHypernymSynsetKeysFromSynsetKey(synsetKey);
        hypernymSynsetKeys.addAll(wn.getInstanceHypernymSynsetKeysFromSynsetKey(synsetKey));
        if (!hypernymSynsetKeys.isEmpty())
        {
            getHypernymHierarchy(wn, hypernymSynsetKeys.get(0), hypernymyHierarchy);
        }
    }

    public static List<String> getHypernymHierarchyIncludeInstanceHypernyms(WordnetHelper wn, String synsetKey)
    {
        List<String> hypernymyHierarchy = new ArrayList<>();
        getHypernymHierarchyIncludeInstanceHypernyms(wn, synsetKey, hypernymyHierarchy);
        return hypernymyHierarchy;
    }

    public static Map<String, String> getReducedSynsetKeysWithHypernyms1(WordnetHelper wn, String[] corpora, boolean removeMonosemics, boolean removeCoarseGrained)
    {
        String senseTag = "wn" + wn.getVersion() + "_key";
        
        Map<String, Set<String>> allVocabulary = new HashMap<>();
        Map<String, List<String>> allHypernymHierarchy = new HashMap<>();
        
        StreamingCorpusReaderWord reader = new StreamingCorpusReaderWord()
        {
            public void readWord(Word word)
            {
                if (word.hasAnnotation(senseTag))
                {
                    String wordKey = word.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(word.getAnnotationValue("pos"));
                    if (removeMonosemics && wn.getSenseKeyListFromWordKey(wordKey).size() == 1) return;
                    List<String> senseKeys = word.getAnnotationValues(senseTag, ";");
                    Set<String> synsetKeys = getUniqueSynsetKeysFromSenseKeys(wn, senseKeys);
                    if (removeCoarseGrained && synsetKeys.size() > 1) return;
                    for (String synsetKey : synsetKeys)
                    {
                        allVocabulary.putIfAbsent(wordKey, new HashSet<>());
                        allVocabulary.get(wordKey).add(synsetKey);
                        allHypernymHierarchy.putIfAbsent(synsetKey, getHypernymHierarchy(wn, synsetKey));
                    }
                }
                /*
                else
                {
                    if (word.hasAnnotation("lemma") && word.hasAnnotation("pos"))
                    {
                        String wordKey = word.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(word.getAnnotationValue("pos"));
                        List<String> senseKeys = wn.getSenseKeyListFromWordKey(wordKey);
                        if (senseKeys != null && senseKeys.size() == 1)
                        {
                            wordKeysOriginal.add(wordKey);
                            senseKeysOriginal.add(senseKeys.get(0));
                            allSenseKeysOriginal.putIfAbsent(wordKey, new HashSet<>());
                            allSenseKeysOriginal.get(wordKey).add(senseKeys.get(0));
                        }
                    }
                }
                */
            }
        };
        
        for (String corpus : corpora)
        {
            reader.load(corpus);
        }

        Set<String> necessarySynsetKeys = new HashSet<>();

        for (String wordKey : allVocabulary.keySet())
        {
            for (String synsetKey : allVocabulary.get(wordKey))
            {
                List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
                int whereToStop = hypernymHierarchy.size();
                boolean found = false;
                for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
                {
                    if (found) break;
                    for (String synsetKey2 : allVocabulary.get(wordKey))
                    {
                        if (synsetKey.equals(synsetKey2)) continue;
                        if (found) break;
                        List<String> hypernymHierarchy2 = allHypernymHierarchy.get(synsetKey2);
                        for (int j = 0 ; j < hypernymHierarchy2.size() ; j++)
                        {
                            if (hypernymHierarchy.get(i).equals(hypernymHierarchy2.get(j)))
                            {
                                whereToStop = i;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (whereToStop == 0)
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(0));
                }
                else // > 0
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(whereToStop - 1));
                }
            }
        }

        Map<String, String> synsetKeysToSimpleSynsetKey = new HashMap<>();
        
        for (String synsetKey : allHypernymHierarchy.keySet())
        {
            List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
            for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
            {
                if (necessarySynsetKeys.contains(hypernymHierarchy.get(i)))
                {
                    synsetKeysToSimpleSynsetKey.put(synsetKey, hypernymHierarchy.get(i));
                    break;
                }
            }
        }

        return synsetKeysToSimpleSynsetKey;
    }

    public static Map<String, String> getReducedSynsetKeysWithHypernyms2(WordnetHelper wn, String[] corpora, boolean removeMonosemics, boolean removeCoarseGrained)
    {
        String senseTag = "wn" + wn.getVersion() + "_key";
        
        Map<String, Set<String>> allVocabulary = new HashMap<>();
        Map<String, List<String>> allHypernymHierarchy = new HashMap<>();
        
        StreamingCorpusReaderWord reader = new StreamingCorpusReaderWord()
        {
            public void readWord(Word word)
            {
                if (word.hasAnnotation(senseTag))
                {
                    String wordKey = word.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(word.getAnnotationValue("pos"));
                    if (removeMonosemics && wn.getSenseKeyListFromWordKey(wordKey).size() == 1) return;
                    List<String> senseKeys = word.getAnnotationValues(senseTag, ";");
                    Set<String> synsetKeys = getUniqueSynsetKeysFromSenseKeys(wn, senseKeys);
                    if (removeCoarseGrained && synsetKeys.size() > 1) return;
                    for (String synsetKey : synsetKeys)
                    {
                        allVocabulary.putIfAbsent(wordKey, new HashSet<>());
                        allVocabulary.get(wordKey).add(synsetKey);
                        allHypernymHierarchy.putIfAbsent(synsetKey, getHypernymHierarchy(wn, synsetKey));
                    }
                }
            }
        };
        
        for (String corpus : corpora)
        {
            reader.load(corpus);
        }

        Map<String, String> synsetKeysToSimpleSynsetKey = new HashMap<>();
        
        for (String wordKey : allVocabulary.keySet())
        {
            for (String synsetKey : allVocabulary.get(wordKey))
            {
                List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
                int whereToStop = hypernymHierarchy.size();
                boolean found = false;
                for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
                {
                    if (found) break;
                    for (String synsetKey2 : allVocabulary.get(wordKey))
                    {
                        if (synsetKey.equals(synsetKey2)) continue;
                        if (found) break;
                        List<String> hypernymHierarchy2 = allHypernymHierarchy.get(synsetKey2);
                        for (int j = 0 ; j < hypernymHierarchy2.size() ; j++)
                        {
                            if (hypernymHierarchy.get(i).equals(hypernymHierarchy2.get(j)))
                            {
                                whereToStop = i;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (whereToStop == 0)
                {
                    synsetKeysToSimpleSynsetKey.put(wordKey + synsetKey, hypernymHierarchy.get(0));
                }
                else // > 0
                {
                    synsetKeysToSimpleSynsetKey.put(wordKey + synsetKey, hypernymHierarchy.get(whereToStop - 1));
                }
            }
        }

        return synsetKeysToSimpleSynsetKey;
    }
    
    public static Map<String, String> getReducedSynsetKeysWithHypernyms3(WordnetHelper wn)
    {
        Map<String, Set<String>> allVocabulary = new HashMap<>();
        Map<String, List<String>> allHypernymHierarchy = new HashMap<>();

        for (String wordKey : wn.getVocabulary())
        {
            allVocabulary.putIfAbsent(wordKey, new HashSet<>());
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                allVocabulary.get(wordKey).add(synsetKey);
                allHypernymHierarchy.putIfAbsent(synsetKey, getHypernymHierarchy(wn, synsetKey));
            }
        }
        
        Set<String> necessarySynsetKeys = new HashSet<>();

        for (String wordKey : allVocabulary.keySet())
        {
            for (String synsetKey : allVocabulary.get(wordKey))
            {
                List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
                int whereToStop = hypernymHierarchy.size();
                boolean found = false;
                for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
                {
                    if (found) break;
                    for (String synsetKey2 : allVocabulary.get(wordKey))
                    {
                        if (synsetKey.equals(synsetKey2)) continue;
                        if (found) break;
                        List<String> hypernymHierarchy2 = allHypernymHierarchy.get(synsetKey2);
                        for (int j = 0 ; j < hypernymHierarchy2.size() ; j++)
                        {
                            if (hypernymHierarchy.get(i).equals(hypernymHierarchy2.get(j)))
                            {
                                whereToStop = i;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (whereToStop == 0)
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(0));
                }
                else // > 0
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(whereToStop - 1));
                }
            }
        }

        Map<String, String> synsetKeysToSimpleSynsetKey = new HashMap<>();
        
        for (String synsetKey : allHypernymHierarchy.keySet())
        {
            List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
            for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
            {
                if (necessarySynsetKeys.contains(hypernymHierarchy.get(i)))
                {
                    synsetKeysToSimpleSynsetKey.put(synsetKey, hypernymHierarchy.get(i));
                    break;
                }
            }
        }

        return synsetKeysToSimpleSynsetKey;
    }

    public static Map<String, String> getReducedSynsetKeysWithHypernyms4(WordnetHelper wn)
    {
        Map<String, Set<String>> allVocabulary = new HashMap<>();
        Map<String, List<String>> allHypernymHierarchy = new HashMap<>();

        for (String wordKey : wn.getVocabulary())
        {
            allVocabulary.putIfAbsent(wordKey, new HashSet<>());
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                allVocabulary.get(wordKey).add(synsetKey);
                allHypernymHierarchy.putIfAbsent(synsetKey, getHypernymHierarchy(wn, synsetKey));
            }
        }

        Map<String, String> synsetKeysToSimpleSynsetKey = new HashMap<>();

        for (String wordKey : allVocabulary.keySet())
        {
            for (String synsetKey : allVocabulary.get(wordKey))
            {
                List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
                int whereToStop = hypernymHierarchy.size();
                boolean found = false;
                for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
                {
                    if (found) break;
                    for (String synsetKey2 : allVocabulary.get(wordKey))
                    {
                        if (synsetKey.equals(synsetKey2)) continue;
                        if (found) break;
                        List<String> hypernymHierarchy2 = allHypernymHierarchy.get(synsetKey2);
                        for (int j = 0 ; j < hypernymHierarchy2.size() ; j++)
                        {
                            if (hypernymHierarchy.get(i).equals(hypernymHierarchy2.get(j)))
                            {
                                whereToStop = i;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (whereToStop == 0)
                {
                    synsetKeysToSimpleSynsetKey.put(wordKey + synsetKey, hypernymHierarchy.get(0));
                }
                else // > 0
                {
                    synsetKeysToSimpleSynsetKey.put(wordKey + synsetKey, hypernymHierarchy.get(whereToStop - 1));
                }
            }
        }

        return synsetKeysToSimpleSynsetKey;
    }

    public static Map<String, String> getSenseCompressionThroughHypernymsAndInstanceHypernymsClusters(WordnetHelper wn, Map<String, String> currentClusters)
    {
        Map<String, Set<String>> allVocabulary = new HashMap<>();
        Map<String, List<String>> allHypernymHierarchy = new HashMap<>();

        for (String wordKey : wn.getVocabulary())
        {
            allVocabulary.putIfAbsent(wordKey, new HashSet<>());
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                allVocabulary.get(wordKey).add(synsetKey);
                allHypernymHierarchy.putIfAbsent(synsetKey, getHypernymHierarchyIncludeInstanceHypernyms(wn, synsetKey));
            }
        }

        Set<String> necessarySynsetKeys = new HashSet<>();

        for (String wordKey : allVocabulary.keySet())
        {
            for (String synsetKey : allVocabulary.get(wordKey))
            {
                List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
                int whereToStop = hypernymHierarchy.size();
                boolean found = false;
                for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
                {
                    if (found) break;
                    for (String synsetKey2 : allVocabulary.get(wordKey))
                    {
                        if (synsetKey.equals(synsetKey2)) continue;
                        if (found) break;
                        List<String> hypernymHierarchy2 = allHypernymHierarchy.get(synsetKey2);
                        for (int j = 0 ; j < hypernymHierarchy2.size() ; j++)
                        {
                            if (hypernymHierarchy.get(i).equals(hypernymHierarchy2.get(j)))
                            {
                                whereToStop = i;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (whereToStop == 0)
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(0));
                }
                else // > 0
                {
                    necessarySynsetKeys.add(hypernymHierarchy.get(whereToStop - 1));
                }
            }
        }

        Map<String, String> synsetKeysToSimpleSynsetKey = new HashMap<>();

        for (String synsetKey : allHypernymHierarchy.keySet())
        {
            List<String> hypernymHierarchy = allHypernymHierarchy.get(synsetKey);
            for (int i = 0 ; i < hypernymHierarchy.size() ; i++)
            {
                if (necessarySynsetKeys.contains(hypernymHierarchy.get(i)))
                {
                    synsetKeysToSimpleSynsetKey.put(synsetKey, hypernymHierarchy.get(i));
                    break;
                }
            }
        }

        return synsetKeysToSimpleSynsetKey;
    }

    public static Map<String, String> getSenseCompressionThroughHypernymsClusters(WordnetHelper wn, Map<String, String> currentClusters)
    {
        return getReducedSynsetKeysWithHypernyms3(wn);
    }

    public static Map<String, String> getSenseCompressionThroughAntonymsClusters(WordnetHelper wn, Map<String, String> currentClusters)
    {
        Map<String, String> antonymClusters = new HashMap<>();
        for (String synsetKey : currentClusters.values())
        {
            if (antonymClusters.containsKey(synsetKey)) continue;
            List<String> antonymSynsetKeys = wn.getAntonymSynsetKeysFromSynsetKey(synsetKey);
            for (String antonymSynsetKey : antonymSynsetKeys)
            {
                antonymClusters.put(antonymSynsetKey, synsetKey);
            }
            antonymClusters.put(synsetKey, synsetKey);
        }
        Map<String, String> newClusters = new HashMap<>();
        for (String synsetKey : currentClusters.keySet())
        {
            newClusters.put(synsetKey, antonymClusters.get(currentClusters.get(synsetKey)));
        }
        return newClusters;
    }

    public static Map<String, String> getSenseCompressionClusters(WordnetHelper wn, boolean hypernyms, boolean instanceHypernyms, boolean antonyms)
    {
        Map<String, String> clusters = new HashMap<>();
        for (String wordKey : wn.getVocabulary())
        {
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                clusters.putIfAbsent(synsetKey, synsetKey);
            }
        }

        if (hypernyms)
        {
            if (instanceHypernyms)
            {
                clusters = getSenseCompressionThroughHypernymsAndInstanceHypernymsClusters(wn, clusters);
            }
            else
            {
                clusters = getSenseCompressionThroughHypernymsClusters(wn, clusters);
            }
        }

        if (antonyms)
        {
            clusters = getSenseCompressionThroughAntonymsClusters(wn, clusters);
        }

        return clusters;
    }

    public static Map<String, String> getSenseCompressionClustersFromFile(String filePath)
    {
        try
        {
            Map<String, String> mapping = new HashMap<>();
            BufferedReader reader = Files.newBufferedReader(Paths.get(filePath));
            reader.lines().map(line -> line.split(" ")).forEach(line -> mapping.put(line[0], line[1]));
            reader.close();
            return mapping;
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
}
