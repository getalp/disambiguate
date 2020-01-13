package getalp.wsd.method.neural;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.IntStream;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.utils.StringUtils;
import getalp.wsd.common.utils.Wrapper;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.embeddings.TextualModelLoader;
import getalp.wsd.embeddings.WordVectors;
import getalp.wsd.ufsac.simple.core.*;
import getalp.wsd.ufsac.streaming.reader.StreamingCorpusReaderSentence;
import getalp.wsd.utils.Json;
import getalp.wsd.utils.WordnetUtils;
import getalp.wsd.common.utils.*;

public class NeuralDataPreparator
{
    private static final String paddingToken = "<pad>"; // input index "0"

    private static final String unknownToken = "<unk>"; // input index "1"

    private static final String beginningOfSentenceToken = "<bos>"; // input index "2"

    private static final String endOfSentenceToken = "<eos>"; // input index "3"

    private static final String skipToken = "<skip>"; // output index "0"


    private static final String inputVocabularyFileName = "/input_vocabulary";

    private static final String outputVocabularyFileName = "/output_vocabulary";

    private static final String outputTranslationVocabularyFileName1 = "/output_translation";

    private static final String outputTranslationVocabularyFileName2 = "_vocabulary";

    private static final String trainFileName = "/train";

    private static final String devFileName = "/dev";

    private static final String configFileName = "/config.json";


    private WordnetHelper wn = WordnetHelper.wn30();

    private String senseTag = "wn" + wn.getVersion() + "_key";

    private int inputFeatures = 0;

    public List<String> txtCorpusFeatures = new ArrayList<>();

    private List<String> inputAnnotationName = new ArrayList<>();

    private List<String> inputEmbeddingsPath = new ArrayList<>();

    private List<String> inputVocabularyPath = new ArrayList<>();

    private List<Map<String, Integer>> inputVocabulary = new ArrayList<>();

    private int outputFeatures = 0;

    private List<String> outputAnnotationName = new ArrayList<>();

    private List<String> outputFixedVocabularyPath = new ArrayList<>();

    private List<Map<String, Integer>> outputVocabulary = new ArrayList<>();

    private int outputTranslations = 0;

    private List<String> outputTranslationName = new ArrayList<>();

    private int outputTranslationFeatures = 0;

    private List<String> outputTranslationAnnotationName = new ArrayList<>();

    private List<String> outputTranslationFixedVocabularyPath = new ArrayList<>();

    private List<List<Map<String, Integer>>> outputTranslationVocabulary = new ArrayList<>();

    private String outputDirectoryPath = "data/neural/wsd/";

    private List<String> originalTrainPaths = new ArrayList<>();

    private List<String> originalDevPaths = new ArrayList<>();

    private int outputFeatureSenseIndex = -1;

    private String corpusFormat;

    private int inputVocabularyLimit;

    private List<Boolean> inputClearText;

    private int outputFeatureVocabularyLimit;

    private int outputTranslationVocabularyLimit;

    private boolean outputTranslationClearText;

    private boolean shareTranslationVocabulary;

    private Set<String> extraWordKeys = null;

    // --- begin public options

    public int maxLineLength = 80;

    public boolean lowercaseWords = true;

    public boolean filterLemma = true;

    public boolean addWordKeyFromSenseKey = false;

    public boolean uniformDash = false;

    public boolean multisenses = false;

    public boolean removeAllCoarseGrained = true;

    public boolean removeMonosemics = false;

    public boolean addMonosemics = false;

    public boolean removeDuplicateSentences = true;

    public Map<String, String> reducedOutputVocabulary = null;

    public int additionalDevFromTrainSize = 0;

    // --- end public options

    public NeuralDataPreparator()
    {

    }

    public void setOutputDirectoryPath(String path)
    {
        outputDirectoryPath = path;
    }

    public void addTrainingCorpus(String corpusPath)
    {
        originalTrainPaths.add(corpusPath);
    }

    public void addDevelopmentCorpus(String corpusPath)
    {
        originalDevPaths.add(corpusPath);
    }

    public void addInputFeature(String annotationName, String embeddingsPath, String vocabularyPath)
    {
        inputFeatures += 1;
        inputAnnotationName.add(annotationName);
        inputEmbeddingsPath.add(embeddingsPath);
        inputVocabularyPath.add(vocabularyPath);
    }

    public void addOutputFeature(String annotationName, String vocabularyPath)
    {
        outputFeatures += 1;
        outputAnnotationName.add(annotationName);
        outputFixedVocabularyPath.add(vocabularyPath);
        if (annotationName.equals(senseTag))
        {
            outputFeatureSenseIndex = outputFeatures - 1;
        }
    }

    public void addOutputTranslation(String translationName, List<String> translationAnnotationName, String vocabularyPath)
    {
        outputTranslations += 1;
        outputTranslationName.add(translationName);
        outputTranslationFixedVocabularyPath.add(vocabularyPath);
        outputTranslationFeatures = translationAnnotationName.size();
        outputTranslationAnnotationName = new ArrayList<>(translationAnnotationName);
    }

    public void setCorpusFormat(String corpusFormat)
    {
        this.corpusFormat = corpusFormat;
    }

    public void setInputVocabularyLimit(int inputVocabularyLimit)
    {
        this.inputVocabularyLimit = inputVocabularyLimit;
    }

    public void setInputClearText(List<Boolean> inputClearText)
    {
        this.inputClearText = inputClearText;
    }

    public void setOutputFeatureVocabularyLimit(int outputFeatureVocabularyLimit)
    {
        this.outputFeatureVocabularyLimit = outputFeatureVocabularyLimit;
    }

    public void setOutputTranslationVocabularyLimit(int outputTranslationVocabularyLimit)
    {
        this.outputTranslationVocabularyLimit = outputTranslationVocabularyLimit;
    }

    public void setOutputTranslationClearText(boolean outputTranslationClearText)
    {
        this.outputTranslationClearText = outputTranslationClearText;
    }

    public void setShareTranslationVocabulary(boolean shareTranslationVocabulary)
    {
        this.shareTranslationVocabulary = shareTranslationVocabulary;
    }

    public void setExtraWordKeys(Set<String> extraWordKeys)
    {
        if (extraWordKeys != null && extraWordKeys.isEmpty()) extraWordKeys = null;
        this.extraWordKeys = extraWordKeys;
    }

    public void prepareTrainingFile() throws Exception
    {
        Files.createDirectories(Paths.get(outputDirectoryPath));

        List<Sentence> trainSentences = extractSentencesFromCorpora(originalTrainPaths);
        List<Sentence> devSentences = extractSentencesFromCorpora(originalDevPaths);
        List<List<Sentence>> translationTrainSentences = new ArrayList<>();
        List<List<Sentence>> translationDevSentences = new ArrayList<>();
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            String translationName = outputTranslationName.get(i);
            List<String> translationTrainPaths = originalTrainPaths.stream().map(str -> getTranslationCorpusName(str, translationName)).collect(Collectors.toList());
            List<String> translationDevPaths = originalDevPaths.stream().map(str -> getTranslationCorpusName(str, translationName)).collect(Collectors.toList());
            List<Sentence> translationNameTrainSentences = extractSentencesFromCorpora(translationTrainPaths);
            List<Sentence> translationNameDevSentences = extractSentencesFromCorpora(translationDevPaths);
            assert(translationNameTrainSentences.size() == trainSentences.size());
            assert(translationNameDevSentences.size() == devSentences.size());
            translationTrainSentences.add(translationNameTrainSentences);
            translationDevSentences.add(translationNameDevSentences);
        }
        extractDevSentencesParallel(trainSentences, translationTrainSentences, devSentences, translationDevSentences, additionalDevFromTrainSize);

        buildExtraWordKeysVocabulary(trainSentences, true);
        buildExtraWordKeysVocabulary(devSentences, false);

        initInputVocabulary();
        initOutputVocabulary();
        initOutputTranslationVocabulary();

        buildVocabulary(trainSentences, inputAnnotationName, inputEmbeddingsPath, inputVocabulary, true, inputVocabularyLimit);
        buildVocabulary(trainSentences, outputAnnotationName, outputFixedVocabularyPath, outputVocabulary, false, outputFeatureVocabularyLimit);
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            buildVocabulary(translationTrainSentences.get(i), outputTranslationAnnotationName, outputTranslationFixedVocabularyPath, outputTranslationVocabulary.get(i), true, outputTranslationVocabularyLimit);
        }
        if (shareTranslationVocabulary && outputTranslations > 0 && outputTranslationFeatures > 0)
        {
            // TODO: construct vocabulary together in buildVocabulary
            // TODO: allow sharing vocabularies of multiples features and languages
            mergeVocabularies(inputVocabulary.get(0), outputTranslationVocabulary.get(0).get(0));
        }
        // TODO: filtering train for translations
        if (outputTranslations == 0)
        {
            trainSentences = filterSentencesWithoutFeature(trainSentences, outputAnnotationName, outputVocabulary);
            if (removeDuplicateSentences)
            {
                trainSentences = removeDuplicates(trainSentences);
            }
        }
        removeEmptyParallelSentences(trainSentences, translationTrainSentences);
        removeEmptyParallelSentences(devSentences, translationDevSentences);
        for (int i = 0 ; i < inputFeatures ; i++)
        {
            writeVocabulary(inputVocabulary.get(i), outputDirectoryPath + inputVocabularyFileName + i, inputClearText.get(i));
        }
        for (int i = 0 ; i < outputFeatures ; i++)
        {
            writeVocabulary(outputVocabulary.get(i), outputDirectoryPath + outputVocabularyFileName + i, false);
        }
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            for (int j = 0 ; j < outputTranslationFeatures ; j++)
            {
                writeVocabulary(outputTranslationVocabulary.get(i).get(j), outputDirectoryPath + outputTranslationVocabularyFileName1 + i + outputTranslationVocabularyFileName2 + j, outputTranslationClearText);
            }
        }
        writeCorpus(trainSentences, translationTrainSentences, outputDirectoryPath + trainFileName);
        writeCorpus(devSentences, translationDevSentences, outputDirectoryPath + devFileName);
        writeConfigFile(outputDirectoryPath + configFileName);
    }

    private void initInputVocabulary() throws IOException
    {
        assert(inputFeatures > 0);
        for (int i = 0 ; i < inputFeatures ; i++)
        {
            if (inputEmbeddingsPath.get(i) != null)
            {
                inputVocabulary.add(loadEmbeddings(inputEmbeddingsPath.get(i)));
            }
            else if (inputVocabularyPath.get(i) != null)
            {
                inputVocabulary.add(loadVocabulary(inputVocabularyPath.get(i)));
            }
            else
            {
                inputVocabulary.add(createNewInputVocabulary());
            }
        }
    }

    private void initOutputVocabulary() throws IOException
    {
        for (int i = 0 ; i < outputFeatures ; i++)
        {
            if (outputFixedVocabularyPath.get(i) != null)
            {
                outputVocabulary.add(readVocabulary(outputFixedVocabularyPath.get(i)));
            }
            else
            {
                outputVocabulary.add(createNewOutputVocabulary());
            }
        }
    }

    private void initOutputTranslationVocabulary() throws IOException
    {
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            List<Map<String, Integer>> translationVocabulary = new ArrayList<>();
            assert(!outputTranslationAnnotationName.isEmpty());
            assert(outputTranslationAnnotationName.size() == outputTranslationFeatures);
            for (int j = 0 ; j < outputTranslationFeatures ; j++)
            {
                if (outputTranslationFixedVocabularyPath.get(i) != null)
                {
                    translationVocabulary.add(loadVocabulary(outputTranslationFixedVocabularyPath.get(i)));
                }
                else
                {
                    translationVocabulary.add(createNewOutputTranslationVocabulary());
                }
            }
            outputTranslationVocabulary.add(translationVocabulary);
        }
    }

    private Map<String, Integer> loadEmbeddings(String embeddingsPath)
    {
        WordVectors embeddings = new TextualModelLoader(false).loadVocabularyOnly(embeddingsPath);
        Map<String, Integer> vocabulary = createNewInputVocabulary();
        int i = vocabulary.size();
        for (String vocab : embeddings.getVocabulary())
        {
            vocabulary.put(vocab, i);
            i++;
        }
        return vocabulary;
    }

    private Map<String, Integer> loadVocabulary(String vocabularyPath) throws IOException
    {
        Map<String, Integer> vocabulary = createNewInputVocabulary();
        Wrapper<Integer> i = new Wrapper<>(vocabulary.size());
        BufferedReader in = Files.newBufferedReader(Paths.get(vocabularyPath));
        in.lines().forEach(line ->
        {
            vocabulary.put(line, i.obj);
            i.obj++;
        });
        in.close();
        return vocabulary;
    }

    private Map<String, Integer> readVocabulary(String vocabularyPath) throws IOException
    {
        Map<String, Integer> vocabulary = new HashMap<>();
        BufferedReader in = Files.newBufferedReader(Paths.get(vocabularyPath));
        in.lines().forEach((line) ->
        {
            String[] tokens = line.split(RegExp.anyWhiteSpaceGrouped.toString());
            vocabulary.put(tokens[1], Integer.valueOf(tokens[0]));
        });
        in.close();
        return vocabulary;
    }

    private Map<String, Integer> createNewInputVocabulary()
    {
        Map<String, Integer> vocabulary = new HashMap<>();
        vocabulary.put(paddingToken, 0);
        vocabulary.put(unknownToken, 1);
        vocabulary.put(beginningOfSentenceToken, 2);
        vocabulary.put(endOfSentenceToken, 3);
        return vocabulary;
    }

    private Map<String, Integer> createNewOutputVocabulary()
    {
        Map<String, Integer> vocabulary = new HashMap<>();
        vocabulary.put(skipToken, 0);
        return vocabulary;
    }

    private Map<String, Integer> createNewOutputTranslationVocabulary()
    {
        Map<String, Integer> vocabulary = new HashMap<>();
        vocabulary.put(paddingToken, 0);
        vocabulary.put(unknownToken, 1);
        vocabulary.put(beginningOfSentenceToken, 2);
        vocabulary.put(endOfSentenceToken, 3);
        return vocabulary;
    }

    private void writeVocabulary(Map<String, Integer> vocabulary, String vocabularyPath, boolean clearText) throws IOException
    {
        BufferedWriter out = Files.newBufferedWriter(Paths.get(vocabularyPath));
        if (!clearText)
        {
            for (Map.Entry<String, Integer> vocab : vocabulary.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toList()))
            {
                out.write("" + vocab.getKey() + "\n");
            }
        }
        out.close();
    }

    private void writeConfigFile(String configFilePath) throws IOException
    {
        Map<Object, Object> config = new LinkedHashMap<>();

        config.put("input_features", inputFeatures);
        config.put("input_annotation_name", inputAnnotationName);
        config.put("input_embeddings_path", inputEmbeddingsPath.stream().map(p -> p != null ? Paths.get(p).toAbsolutePath().toString() : null).collect(Collectors.toList()));
        config.put("input_clear_text", IntStream.range(0, inputFeatures).boxed().map(i -> inputClearText.get(i)).collect(Collectors.toList()));
        config.put("output_features", outputFeatures);
        config.put("output_annotation_name", outputAnnotationName);
        config.put("output_translations", outputTranslations);
        config.put("output_translation_name", outputTranslationName);
        config.put("output_translation_features", outputTranslationFeatures);
        config.put("output_translation_annotation_name", outputTranslationAnnotationName);
        config.put("output_translation_clear_text", outputTranslationClearText);

        BufferedWriter out = Files.newBufferedWriter(Paths.get(configFilePath));
        Json.write(out, config);
        out.close();
    }

    private List<Sentence> extractSentencesFromCorpora(List<String> originalCorpusPaths) throws Exception
    {
        List<Sentence> sentences;
        if (corpusFormat.equals("xml"))
        {
            sentences = extractSentencesFromUFSACCorpora(originalCorpusPaths);
        }
        else
        {
            sentences = extractSentencesFromTXTCorpora(originalCorpusPaths);
        }
        cleanSentences(sentences);
        return sentences;
    }

    private List<Sentence> extractSentencesFromTXTCorpora(List<String> originalCorpusPaths) throws Exception
    {
        List<Sentence> allSentences = new ArrayList<>();
        if (txtCorpusFeatures.isEmpty())
        {
            txtCorpusFeatures = new ArrayList<>();
            for (int i = 0; i < inputFeatures; i++)
            {
                txtCorpusFeatures.add(inputAnnotationName.get(i));
            }
            for (int i = 0; i < outputFeatures; i++)
            {
                txtCorpusFeatures.add(outputAnnotationName.get(i));
            }
        }
        for (String originalCorpusPath : originalCorpusPaths)
        {
            System.out.println("Extracting sentences from corpus " + originalCorpusPath);
            BufferedReader reader = Files.newBufferedReader(Paths.get(originalCorpusPath));
            for (String line = reader.readLine(); line != null ; line = reader.readLine())
            {
                Sentence sentence = new Sentence();
                String[] words = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
                for (String word : words)
                {
                    Word ufsacWord = new Word();
                    String[] wordFeatures = word.split(Pattern.quote("|"));
                    if (wordFeatures.length < 1)
                    {
                        System.out.println("Warning: empty word in sentence: " + line);
                        wordFeatures = new String[]{"/"};
                    }
                    ufsacWord.setValue(wordFeatures[0]);
                    for (int i = 1; i < txtCorpusFeatures.size(); i++)
                    {
                        if (wordFeatures.length > i)
                        {
                            ufsacWord.setAnnotation(txtCorpusFeatures.get(i), wordFeatures[i]);
                        }
                    }
                    sentence.addWord(ufsacWord);
                }
                allSentences.add(sentence);
            }
        }
        return allSentences;
    }

    private List<Sentence> extractSentencesFromUFSACCorpora(List<String> originalCorpusPaths)
    {
        List<Sentence> allSentences = new ArrayList<>();
        StreamingCorpusReaderSentence reader = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(getalp.wsd.ufsac.core.Sentence sentence)
            {
                allSentences.add(new Sentence(sentence));
            }
        };

        for (String originalCorpusPath : originalCorpusPaths)
        {
            System.out.println("Extracting sentences from corpus " + originalCorpusPath);
            reader.load(originalCorpusPath);
        }
        return allSentences;
    }

    private void cleanSentences(List<Sentence> sentences)
    {
        System.out.println("Cleaning sentences");
        for (Sentence s : sentences)
        {
            s.limitSentenceLength(maxLineLength);

            List<Word> words = s.getWords();

            for (Word w : words)
            {
                /// add monosemics if asked
                if (addMonosemics && !w.hasAnnotation(senseTag) && w.hasAnnotation("lemma") && w.hasAnnotation("pos"))
                {
                    String wordKey = w.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(w.getAnnotationValue("pos"));
                    if (wn.isWordKeyExists(wordKey))
                    {
                        List<String> senseKeys = wn.getSenseKeyListFromWordKey(wordKey);
                        if (senseKeys.size() == 1)
                        {
                            w.setAnnotation(senseTag, senseKeys.get(0));
                        }
                    }
                }

                /// clean output sense tags, convert them to synset keys
                if (w.hasAnnotation(senseTag))
                {
                    List<String> senseKeys = w.getAnnotationValues(senseTag, ";");

                    if (!w.hasAnnotation("lemma"))
                    {
                        w.setAnnotation("lemma", WordnetUtils.extractLemmaFromSenseKey(senseKeys.get(0)));
                    }
                    if (!w.hasAnnotation("pos"))
                    {
                        w.setAnnotation("pos", WordnetUtils.extractPOSFromSenseKey(senseKeys.get(0)));
                    }

                    String wordKey = w.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(w.getAnnotationValue("pos"));

                    if (addWordKeyFromSenseKey)
                    {
                        w.setAnnotation("word_key", wordKey);
                    }

                    if (removeMonosemics && wn.getSenseKeyListFromWordKey(wordKey).size() == 1)
                    {
                        w.removeAnnotation(senseTag);
                        senseKeys = Collections.emptyList();
                    }

                    Set<String> synsetKeys = WordnetUtils.getUniqueSynsetKeysFromSenseKeys(wn, senseKeys);
                    List<String> finalSynsetKeys = new ArrayList<>();

                    if (removeAllCoarseGrained && synsetKeys.size() > 1)
                    {
                        synsetKeys.clear();
                    }
                    for (String synsetKey : synsetKeys)
                    {
                        if (reducedOutputVocabulary != null)
                        {
                            synsetKey = reducedOutputVocabulary.getOrDefault(synsetKey, synsetKey);
                        }
                        finalSynsetKeys.add(synsetKey);
                    }
                    if (finalSynsetKeys.isEmpty())
                    {
                        w.removeAnnotation(senseTag);
                    }
                    else
                    {
                        if (!multisenses)
                        {
                            finalSynsetKeys = finalSynsetKeys.subList(0, 1);
                        }
                        w.setAnnotation(senseTag, finalSynsetKeys, ";");
                    }
                }

                // lowercase word
                if (lowercaseWords)
                {
                    w.setValue(w.getValue().toLowerCase());
                }

                // uniformize dash
                if (uniformDash)
                {
                    w.setValue(w.getValue().replaceAll("_", "-"));
                }
            }
        }
    }

    private void buildExtraWordKeysVocabulary(List<Sentence> sentences, boolean isTrain)
    {
        if (extraWordKeys == null) return;
        System.out.println("Building extra wordkeys vocabulary");
        Map<String, Set<String>> sensesPerWordKey = new HashMap<>();
        for (Sentence s : sentences)
        {
            List<Word> words = s.getWords();
            for (Word w : words)
            {
                if (w.hasAnnotation(senseTag))
                {
                    String senseKey = w.getAnnotationValue(senseTag);
                    for (String extraWordKey : extraWordKeys)
                    {
                        // si extraWordKey pourrait avoir le sens senseKey, alors on met cette annotation
                        boolean isPossibleSense = false;
                        for (String extrasenseKey : wn.getSenseKeyListFromWordKey(extraWordKey))
                        {
                            String extraSynsetKey = wn.getSynsetKeyFromSenseKey(extrasenseKey);
                            if (reducedOutputVocabulary != null)
                            {
                                extraSynsetKey = reducedOutputVocabulary.getOrDefault(extraSynsetKey, extraSynsetKey);
                            }
                            if (extraSynsetKey.equals(senseKey))
                            {
                                isPossibleSense = true;
                                break;
                            }
                        }
                        if (isPossibleSense)
                        {
                            w.setAnnotation(extraWordKey, senseKey);
                            if (isTrain)
                            {
                                sensesPerWordKey.putIfAbsent(extraWordKey, new HashSet<>());
                                sensesPerWordKey.get(extraWordKey).add(senseKey);
                            }
                        }
                    }
                }
            }
        }
        if (isTrain)
        {
            for (String extraWordKey : sensesPerWordKey.keySet())
            {
                if (sensesPerWordKey.get(extraWordKey).size() > 1)
                {
                    addOutputFeature(extraWordKey, null);
                }
            }
        }
    }

    private void buildVocabulary(List<Sentence> allSentences,
                                 List<String> annotationName, List<String> fixedVocabularyPath,
                                 List<Map<String, Integer>> vocabulary,
                                 boolean isInputVocabulary, int vocabularyLimit)
    {
        System.out.println("Building vocabulary");

        List<Map<String, Integer>> vocabularyFrequencies = new ArrayList<>();
        for (int i = 0 ; i < annotationName.size(); i++)
        {
            vocabularyFrequencies.add(new HashMap<>());
        }

        for (Sentence s : allSentences)
        {
            List<Word> words = s.getWords();

            for (Word w : words)
            {
                for (int i = 0 ; i < annotationName.size() ; i++)
                {
                    if (fixedVocabularyPath.get(i) != null) continue;
                    List<String> featureValues;
                    if (isInputVocabulary)
                    {
                        featureValues = Collections.singletonList(w.getAnnotationValue(annotationName.get(i)));
                        if (featureValues.get(0).isEmpty()) continue;
                    }
                    else
                    {
                        featureValues = w.getAnnotationValues(annotationName.get(i), ";");
                        if (featureValues.isEmpty()) continue;
                    }
                    for (String featureValue : featureValues)
                    {
                        Map<String, Integer> featureFrequencies = vocabularyFrequencies.get(i);
                        int currentFrequency = featureFrequencies.getOrDefault(featureValue, 0);
                        currentFrequency += 1;
                        featureFrequencies.put(featureValue, currentFrequency);
                    }
                }
            }
        }

        for (int i = 0 ; i < annotationName.size() ; i++)
        {
            if (fixedVocabularyPath.get(i) != null) continue;
            Map<String, Integer> featureFrequencies = vocabularyFrequencies.get(i);
            Map<String, Integer> featureVocabulary;
            if (isInputVocabulary)
            {
                featureVocabulary = createNewInputVocabulary();
            }
            else
            {
                featureVocabulary = createNewOutputVocabulary();
            }
            int initVocabularySize = featureVocabulary.size();
            List<String> sortedKeys = featureFrequencies.entrySet().stream().sorted(Map.Entry.comparingByValue()).map(Map.Entry::getKey).collect(Collectors.toList());
            Collections.reverse(sortedKeys);
            if (vocabularyLimit <= 0)
            {
                vocabularyLimit = sortedKeys.size();
            }
            else
            {
                vocabularyLimit = Math.min(vocabularyLimit, sortedKeys.size());
            }
            for (int j = 0; j < vocabularyLimit; j++)
            {
                featureVocabulary.put(sortedKeys.get(j), j + initVocabularySize);
            }
            vocabulary.set(i, featureVocabulary);
        }
    }

    private void mergeVocabularies(Map<String, Integer> vocabulary1, Map<String, Integer> vocabulary2)
    {
        int j = vocabulary1.size();
        for (String wordInVocabulary2 : vocabulary2.keySet())
        {
            if (!vocabulary1.containsKey(wordInVocabulary2))
            {
                vocabulary1.put(wordInVocabulary2, j);
                j++;
            }
        }
        vocabulary2.putAll(vocabulary1);
    }

    private List<Sentence> filterSentencesWithoutFeature(List<Sentence> allSentences, List<String> annotationName,
                                                         List<Map<String, Integer>> vocabulary)
    {
        System.out.println("Filtering sentences without feature");

        List<Sentence> filteredSentences = new ArrayList<>();

        for (Sentence s : allSentences)
        {
            List<Word> words = s.getWords();
            boolean sentenceHasOutputFeatures = false;
            for (Word w : words)
            {
                for (int i = 0 ; i < annotationName.size() ; i++)
                {
                    List<String> featureValues = w.getAnnotationValues(annotationName.get(i), ";");
                    if (featureValues.isEmpty()) continue;
                    Map<String, Integer> featureVocabulary = vocabulary.get(i);
                    for (String featureValue : featureValues)
                    {
                        if (featureVocabulary.containsKey(featureValue))
                        {
                            sentenceHasOutputFeatures = true;
                        }
                    }
                }
            }

            if (sentenceHasOutputFeatures)
            {
                filteredSentences.add(s);
            }
        }

        return filteredSentences;
    }

    private List<Sentence> removeDuplicates(List<Sentence> sentences)
    {
        System.out.println("Removing duplicate sentences");

        Map<String, Sentence> realSentences = new HashMap<>();
        for (Sentence currentSentence : sentences)
        {
            String sentenceAsString = currentSentence.toString();
            if (!realSentences.containsKey(sentenceAsString))
            {
                realSentences.put(sentenceAsString, currentSentence);
            }
            else
            {
                Sentence realSentence = realSentences.get(sentenceAsString);
                assert(realSentence.getWords().size() == currentSentence.getWords().size());
                for (int i = 0; i < currentSentence.getWords().size(); i++)
                {
                    Word currentSentenceWord = currentSentence.getWords().get(i);
                    Word realSentenceWord = realSentence.getWords().get(i);
                    if (currentSentenceWord.hasAnnotation(senseTag))
                    {
                        List<String> currentSenseKeys = currentSentenceWord.getAnnotationValues(senseTag, ";");
                        List<String> realSenseKeys = new ArrayList<>(realSentenceWord.getAnnotationValues(senseTag, ";"));
                        if (realSenseKeys.isEmpty())
                        {
                            realSentenceWord.setAnnotation("lemma", currentSentenceWord.getAnnotationValue("lemma"));
                            realSentenceWord.setAnnotation("pos", currentSentenceWord.getAnnotationValue("pos"));
                        }
                        else if (!realSentenceWord.getAnnotationValue("lemma").equals(currentSentenceWord.getAnnotationValue("lemma")) ||
                                 !realSentenceWord.getAnnotationValue("pos").equals(currentSentenceWord.getAnnotationValue("pos")))
                        {
                            continue;
                        }
                        for (String sense : currentSenseKeys)
                        {
                            if (!realSenseKeys.contains(sense))
                            {
                                realSenseKeys.add(sense);
                            }
                        }
                        realSentenceWord.setAnnotation(senseTag, realSenseKeys, ";");
                    }
                }
            }
        }
        sentences.clear();
        System.gc();
        List<Sentence> realTrueSentences = new ArrayList<>(realSentences.values());
        realSentences.clear();
        System.gc();
        for (Sentence s : realTrueSentences)
        {
            for (Word w : s.getWords())
            {
                if (w.hasAnnotation(senseTag))
                {
                    List<String> senseKeys = w.getAnnotationValues(senseTag, ";");
                    if (senseKeys.size() > 1)
                    {
                        if (removeAllCoarseGrained)
                        {
                            w.removeAnnotation(senseTag);
                        }
                        else if (!multisenses)
                        {
                            w.setAnnotation(senseTag, senseKeys.subList(0, 1), ";");
                        }
                    }
                }
            }
        }
        return realTrueSentences;
    }

    private void extractDevSentencesParallel(List<Sentence> trainSentences, List<List<Sentence>> translatedTrainSentences,
                                             List<Sentence> devSentences, List<List<Sentence>> translatedDevSentences,
                                             int count)
    {
        if (count <= 0) return;
        if (trainSentences.size() <= count) return;
        System.out.println("Extracting dev sentences from train");

        // generating random indices of sentences to transfer
        List<Integer> randomIndices = new ArrayList<>();
        for (int i = 0 ; i < trainSentences.size() ; i++)
        {
            randomIndices.add(i);
        }
        Collections.shuffle(randomIndices);
        randomIndices = randomIndices.subList(0, count);

        // fetching Sentence object from indices
        List<Sentence> trainSentencesToExtract = new ArrayList<>();
        List<List<Sentence>> translatedTrainSentencesToExtract = new ArrayList<>();
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            translatedTrainSentencesToExtract.add(new ArrayList<>());
        }
        for (int index : randomIndices)
        {
            trainSentencesToExtract.add(trainSentences.get(index));
            for (int i = 0 ; i < outputTranslations ; i++)
            {
                translatedTrainSentencesToExtract.get(i).add(translatedTrainSentences.get(i).get(index));
            }
        }

        // actual remove from train / add to dev the sentences
        trainSentences.removeAll(trainSentencesToExtract);
        devSentences.addAll(trainSentencesToExtract);
        for (int i = 0 ; i < outputTranslations ; i++)
        {
            translatedTrainSentences.get(i).removeAll(translatedTrainSentencesToExtract.get(i));
            translatedDevSentences.get(i).addAll(translatedTrainSentencesToExtract.get(i));
        }
    }

    private void removeEmptyParallelSentences(List<Sentence> sentences, List<List<Sentence>> translatedSentences)
    {
        System.out.println("Removing empty parallel sentences");
        List<Integer> sentenceIndicesToRemove = new ArrayList<>();
        for (int i = 0 ; i < sentences.size() ; i++)
        {
            for (List<Sentence> oneLanguageTranslatedSentences : translatedSentences)
            {
                Sentence translatedSentence = oneLanguageTranslatedSentences.get(i);
                List<Word> translatedSentenceWords = translatedSentence.getWords();
                boolean empty = true;
                for (Word w : translatedSentenceWords)
                {
                    if (!empty) break;
                    for (String annotation : outputTranslationAnnotationName)
                    {
                        if (w.hasAnnotation(annotation))
                        {
                            empty = false;
                            break;
                        }
                    }
                }
                if (empty)
                {
                    sentenceIndicesToRemove.add(i);
                    break;
                }
            }
        }
        sentences.removeAll(sentenceIndicesToRemove.stream().map(sentences::get).collect(Collectors.toList()));
        for (List<Sentence> tsentences : translatedSentences)
        {
            tsentences.removeAll(sentenceIndicesToRemove.stream().map(tsentences::get).collect(Collectors.toList()));
        }
    }

    private String getTranslationCorpusName(String originalCorpusName, String translationName)
    {
        if (corpusFormat.equals("xml"))
        {
            return originalCorpusName.substring(0, originalCorpusName.lastIndexOf(".xml")) + "." + translationName + ".xml";
        }
        else
        {
            return originalCorpusName.substring(0, originalCorpusName.lastIndexOf(".")) + "." + translationName;
        }
    }

    private void writeCorpus(List<Sentence> sentences, List<List<Sentence>> translatedSentences, String corpusPath) throws Exception
    {
        System.out.println("Writing corpus " + corpusPath);
        
        Wrapper<BufferedWriter> writer = new Wrapper<>();
        writer.obj = Files.newBufferedWriter(Paths.get(corpusPath));
        for (int si = 0 ; si < sentences.size(); si++)
        {
            Sentence s = sentences.get(si);
            List<Word> words = s.getWords();

            /// step 1 : write input features
            for (Word w : words)
            {
                List<String> featureValues = new ArrayList<>();
                for (int i = 0 ; i < inputFeatures ; i++)
                {
                    String featureValue = w.getAnnotationValue(inputAnnotationName.get(i));
                    Map<String, Integer> featureVocabulary = inputVocabulary.get(i);
                    if (featureValue.isEmpty() || !(featureVocabulary.containsKey(featureValue) || (inputVocabularyLimit <= 0 && inputClearText.get(i))))
                    {
                        featureValue = unknownToken;
                    }
                    if (inputClearText.get(i))
                    {
                        featureValue = featureValue.replace("/", "<slash>");
                    }
                    else
                    {
                        featureValue = Integer.toString(featureVocabulary.get(featureValue));
                    }
                    featureValues.add(featureValue);
                }
                writer.obj.write(StringUtils.join(featureValues, "/") + " ");
            }
            writer.obj.newLine();

            if (outputFeatures > 0)
            {
                /// step 2 : write output features
                for (Word w : words)
                {
                    List<String> featureValues = new ArrayList<>();
                    for (int i = 0; i < outputFeatures; i++)
                    {
                        Map<String, Integer> featureVocabulary = outputVocabulary.get(i);
                        List<String> thisFeatureValues = w.getAnnotationValues(outputAnnotationName.get(i), ";");
                        thisFeatureValues = thisFeatureValues.stream().filter(featureVocabulary::containsKey).collect(Collectors.toList());
                        if (thisFeatureValues.isEmpty())
                        {
                            thisFeatureValues = Collections.singletonList(skipToken);
                        }
                        thisFeatureValues = thisFeatureValues.stream().map(value -> Integer.toString(featureVocabulary.get(value))).collect(Collectors.toList());
                        featureValues.add(StringUtils.join(thisFeatureValues, ";"));
                    }
                    writer.obj.write(StringUtils.join(featureValues, "/") + " ");
                }
                writer.obj.newLine();

                /// step 3 : write output (sense_tag) restrictions
                for (Word w : words)
                {
                    for (int i = 0 ; i < outputFeatures ; i++)
                    {
                        if (i > 0)
                        {
                            writer.obj.write("/");
                        }
                        String featureTag = outputAnnotationName.get(i);
                        Map<String, Integer> featureVocabulary = outputVocabulary.get(i);
                        if (w.hasAnnotation(featureTag) && w.getAnnotationValues(featureTag, ";").stream().anyMatch(featureVocabulary::containsKey))
                        {
                            if (outputFeatureSenseIndex == i)
                            {
                                List<String> restrictedSenses = new ArrayList<>();
                                String wordKey = w.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(w.getAnnotationValue("pos"));
                                for (String senseKey : wn.getSenseKeyListFromWordKey(wordKey))
                                {
                                    String synsetKey = wn.getSynsetKeyFromSenseKey(senseKey);
                                    if (reducedOutputVocabulary != null)
                                    {
                                        synsetKey = reducedOutputVocabulary.getOrDefault(synsetKey, synsetKey);
                                    }
                                    if (featureVocabulary.containsKey(synsetKey))
                                    {
                                        restrictedSenses.add("" + featureVocabulary.get(synsetKey));
                                    }
                                }
                                writer.obj.write(StringUtils.join(restrictedSenses, ";"));
                            }
                            else
                            {
                                writer.obj.write("-1");
                            }
                        }
                        else
                        {
                            writer.obj.write("0");
                        }
                    }
                    writer.obj.write(" ");
                }
                writer.obj.newLine();
            }

            /// step 4 : write translation output
            for (int ti = 0 ; ti < outputTranslations ; ti++)
            {
                Sentence translatedSentence = translatedSentences.get(ti).get(si);
                words = translatedSentence.getWords();
                for (Word w : words)
                {
                    List<String> featureValues = new ArrayList<>();
                    for (int i = 0; i < outputTranslationFeatures; i++)
                    {
                        String featureValue = w.getAnnotationValue(outputTranslationAnnotationName.get(i));
                        Map<String, Integer> featureVocabulary = outputTranslationVocabulary.get(ti).get(i);
                        if (featureValue.isEmpty() || !featureVocabulary.containsKey(featureValue))
                        {
                            featureValue = unknownToken;
                        }
                        if (outputTranslationClearText)
                        {
                            featureValue = featureValue.replace("/", "<slash>");
                        }
                        else
                        {
                            featureValue = Integer.toString(featureVocabulary.get(featureValue));
                        }
                        featureValues.add(featureValue);
                    }
                    writer.obj.write(StringUtils.join(featureValues, "/") + " ");
                }
                writer.obj.newLine();
            }
        }
        writer.obj.close();
    }
}

