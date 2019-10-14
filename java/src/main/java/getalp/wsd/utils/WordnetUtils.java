package getalp.wsd.utils;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

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

    public static Map<String, String> getSenseCompressionThroughHypernymsClusters(WordnetHelper wn)
    {
        return getReducedSynsetKeysWithHypernyms3(wn);
    }

    public static Map<String, String> getSenseCompressionThroughHypernymsClusters()
    {
        return getReducedSynsetKeysWithHypernyms3(WordnetHelper.wn30());
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

    public static Map<String, String> initMapping()
    {
        WordnetHelper wn = WordnetHelper.wn30();
        Map<String, String> mapping = new HashMap<>();
        for (String wordKey : wn.getVocabulary())
        {
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                mapping.putIfAbsent(synsetKey, synsetKey);
            }
        }
        return mapping;
    }

    private static boolean checkIsOkayToChangeInMapping(String key, String newValue, Map<String, String> mapping, Map<String, List<String>> synsetKeyToWordKeys, Map<String, List<String>> wordKeyToSynsetKeys)
    {
        for (String wordKey : synsetKeyToWordKeys.get(key))
        {
            Set<String> set = new HashSet<>();
            for (String synsetKey : wordKeyToSynsetKeys.get(wordKey))
            {
                String mappedSynsetKey;
                if (synsetKey.equals(key))
                {
                    mappedSynsetKey = newValue;
                }
                else
                {
                    mappedSynsetKey = mapping.get(synsetKey);
                }
                if (set.contains(mappedSynsetKey))
                {
                    return false;
                }
                set.add(mappedSynsetKey);
            }
        }
        return true;
    }

    private static boolean checkIsOkayToMergeClusters(String clusterKey1, String clusterKey2, Map<String, List<String>> inverseMapping, Map<String, String> mapping, Map<String, List<String>> synsetKeyToWordKeys, Map<String, List<String>> wordKeyToSynsetKeys)
    {
        for (String synsetKey : inverseMapping.get(clusterKey2))
        {
            if (!checkIsOkayToChangeInMapping(synsetKey, clusterKey1, mapping, synsetKeyToWordKeys, wordKeyToSynsetKeys))
            {
                return false;
            }
        }
        return true;
    }

    public static Map<String, String> getSenseCompressionThroughAllRelationsClusters()
    {
        WordnetHelper wn = WordnetHelper.wn30();

        // create synsetKeyToWordKeys
        Map<String, List<String>> synsetKeyToWordKeys = new HashMap<>();
        for (String wordKey : wn.getVocabulary())
        {
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                synsetKeyToWordKeys.putIfAbsent(synsetKey, new ArrayList<>());
                synsetKeyToWordKeys.get(synsetKey).add(wordKey);
            }
        }

        // create wordKeyToSynsetKeys
        Map<String, List<String>> wordKeyToSynsetKeys = new HashMap<>();
        for (String wordKey : wn.getVocabulary())
        {
            wordKeyToSynsetKeys.put(wordKey, new ArrayList<>());
            for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
            {
                String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                wordKeyToSynsetKeys.get(wordKey).add(synsetKey);
            }
        }

        // create mapping (synset to cluster)
        Map<String, String> mapping = initMapping();

        // create inverseMapping (cluster to synsets)
        Map<String, List<String>> inverseMapping = new HashMap<>();
        for (String synsetKey : mapping.keySet())
        {
            inverseMapping.put(synsetKey, new ArrayList<>(Collections.singletonList(synsetKey)));
        }

        // create relatedClusters (cluster to related clusters)
        Map<String, List<String>> relatedClusters = new HashMap<>();
        for (String synsetKey : mapping.keySet())
        {
            relatedClusters.put(synsetKey, new ArrayList<>(wn.getRelatedSynsetsKeyFromSynsetKey(synsetKey)));
            if (wn.getSenseKeyListFromSynsetKey(synsetKey).size() == 1)
            {
                relatedClusters.get(synsetKey).addAll(wn.getRelatedSynsetsKeyFromSenseKey(wn.getSenseKeyListFromSynsetKey(synsetKey).get(0)));
            }
        }

        // ensure symetry in related clusters
        for (String synsetKey : mapping.keySet())
        {
            for (String relatedSynsetKey : relatedClusters.get(synsetKey))
            {
                relatedClusters.get(relatedSynsetKey).add(synsetKey);
            }
        }

        // ensure uniqueness in related clusters
        for (String synsetKey : mapping.keySet())
        {
            relatedClusters.put(synsetKey, relatedClusters.get(synsetKey).stream().distinct().filter(key -> !synsetKey.equals(key)).collect(Collectors.toList()));
        }

        // create clusterSizes (size of each cluster)
        Map<String, Integer> clusterSizes = new HashMap<>();
        for (String synsetKey : mapping.keySet())
        {
            clusterSizes.put(synsetKey, 1);
        }

        int previousTotal = inverseMapping.size();
        int total = -1;
        int step = 0;
        outer:
        while (total != previousTotal)
        {
            System.out.print("sense cluster creation step " + step);
            step++;

            int i = 0;
            previousTotal = inverseMapping.size();

            //for (String clusterKey : new ArrayList<>(inverseMapping.keySet()))
            //for (String clusterKey : inverseMapping.keySet().stream().sorted(Comparator.comparingInt(clusterSizes::get)).collect(Collectors.toList()))
            List<String> iter = clusterSizes.entrySet().stream().sorted(Map.Entry.comparingByValue()).map(Map.Entry::getKey).collect(Collectors.toList());
            //Collections.shuffle(iter);
            for (String clusterKey : iter)
            {
                //System.out.print("  " + i + "/" + total + "\r");
                //System.out.flush();
                //i++;

                //if (!inverseMapping.containsKey(clusterKey)) continue;

                //for (String relatedClusterKey : new ArrayList<>(relatedClusters.get(clusterKey)))
                List<String> iter2 = relatedClusters.get(clusterKey).stream().sorted(Comparator.comparingInt(clusterSizes::get)).collect(Collectors.toList());
                //Collections.shuffle(iter2);
                for (String relatedClusterKey : iter2)
                {
                    //if (!inverseMapping.containsKey(relatedClusterKey)) continue;

                    if (checkIsOkayToMergeClusters(clusterKey, relatedClusterKey, inverseMapping, mapping, synsetKeyToWordKeys, wordKeyToSynsetKeys))
                    {
                        for (String relatedSynsetKey : inverseMapping.get(relatedClusterKey))
                        {
                            mapping.put(relatedSynsetKey, clusterKey);
                        }

                        inverseMapping.get(clusterKey).addAll(inverseMapping.get(relatedClusterKey));
                        inverseMapping.remove(relatedClusterKey);

                        for (String relatedRelatedClusters : relatedClusters.get(relatedClusterKey))
                        {
                            if (relatedRelatedClusters.equals(clusterKey)) continue;

                            if (!relatedClusters.get(clusterKey).contains(relatedRelatedClusters))
                            {
                                relatedClusters.get(clusterKey).add(relatedRelatedClusters);
                                relatedClusters.get(relatedRelatedClusters).add(clusterKey);
                            }

                            relatedClusters.get(relatedRelatedClusters).remove(relatedClusterKey);
                        }
                        relatedClusters.get(clusterKey).remove(relatedClusterKey);
                        relatedClusters.remove(relatedClusterKey);

                        clusterSizes.put(clusterKey, clusterSizes.get(clusterKey) + clusterSizes.get(relatedClusterKey));
                        clusterSizes.remove(relatedClusterKey);

                        total = inverseMapping.size();
                        System.out.println(" - size is " + total);

                        continue outer;
                    }
                }
            }

            //total = inverseMapping.size();
            //System.out.println();
            //System.out.println("  Size is " + total);
        }
        return mapping;
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
