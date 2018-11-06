package getalp.wsd.method.neural;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import java.nio.file.Files;
import java.nio.file.Paths;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.utils.StringUtils;
import getalp.wsd.common.utils.Wrapper;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.embeddings.TextualModelLoader;
import getalp.wsd.embeddings.WordVectors;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.streaming.reader.StreamingCorpusReaderSentence;
import getalp.wsd.utils.Json;
import getalp.wsd.utils.WordnetUtils;
import getalp.wsd.common.utils.*;

public class NeuralDataPreparator
{
    private static final String paddingToken = "<pad>"; // input index "0"

    private static final String unknownToken = "<unk>"; // input index "1"

    private static final String skipToken = "<skip>"; // output index "0"


    private static final String inputVocabularyFileName = "/input_vocabulary";

    private static final String outputVocabularyFileName = "/output_vocabulary";

    private static final String trainFileName = "/train";

    private static final String devFileName = "/dev";

    private static final String configFileName = "/config.json";


    private WordnetHelper wn = WordnetHelper.wn30();

    private String senseTag = "wn" + wn.getVersion() + "_key";

    private int inputFeatures = 0;

    private List<String> inputAnnotationName = new ArrayList<>();

    private List<String> inputEmbeddingsPath = new ArrayList<>();

    private List<Map<String, Integer>> inputVocabulary = new ArrayList<>();

    private List<Integer> inputVocabularyCurrentIndex = new ArrayList<>();

    private int outputFeatures = 0;

    private List<String> outputAnnotationName = new ArrayList<>();

    private List<Map<String, Integer>> outputVocabulary = new ArrayList<>();

    private List<Integer> currentOutputVocabularyIndex = new ArrayList<>();

    private String outputDirectoryPath = "data/neural/wsd/";

    private List<String> originalTrainPaths = new ArrayList<>();

    private List<String> originalDevPaths = new ArrayList<>();

    private int outputFeatureSenseIndex = -1;

    private List<String> fixedOutputVocabularyPath = new ArrayList<>();

    // --- begin public options

    public int maxLineLength = 80;

    public boolean lowercaseWords = true;

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

    public void addInputFeature(String annotationName)
    {
        this.addInputFeature(annotationName, null);
    }

    public void addInputFeature(String annotationName, String embeddingsPath)
    {
        inputFeatures += 1;
        inputAnnotationName.add(annotationName);
        inputEmbeddingsPath.add(embeddingsPath);
    }

    public void addOutputFeature(String annotationName)
    {
        addOutputFeature(annotationName, null);
    }

    public void addOutputFeature(String annotationName, String vocabularyPath)
    {
        outputFeatures += 1;
        outputAnnotationName.add(annotationName);
        fixedOutputVocabularyPath.add(vocabularyPath);
        if (annotationName.equals(senseTag))
        {
            outputFeatureSenseIndex = outputFeatures - 1;
        }
    }

    public void prepareTrainingFile() throws Exception
    {
        Files.createDirectories(Paths.get(outputDirectoryPath));
        initInputVocabulary();
        initOutputVocabulary();
        List<Sentence> trainSentences = preprocessCorpora(originalTrainPaths, false);
        if (removeDuplicateSentences)
        {
            trainSentences = removeDuplicates(trainSentences);
        }
        List<Sentence> additionalDevSentences = extractDevSentences(trainSentences, additionalDevFromTrainSize);
        for (int i = 0 ; i < inputFeatures ; i++)
        {
            writeVocabulary(inputVocabulary.get(i), outputDirectoryPath + inputVocabularyFileName + i);
        }
        for (int i = 0 ; i < outputFeatures ; i++)
        {
            writeVocabulary(outputVocabulary.get(i), outputDirectoryPath + outputVocabularyFileName + i);
        }
        writeCorpus(trainSentences, outputDirectoryPath + trainFileName);
        List<Sentence> devSentences = preprocessCorpora(originalDevPaths, true);
        devSentences.addAll(additionalDevSentences);
        writeCorpus(devSentences, outputDirectoryPath + devFileName);
        writeConfigFile(outputDirectoryPath + configFileName);
    }

    private void initInputVocabulary()
    {
        for (int i = 0 ; i < inputFeatures ; i++)
        {
            if (inputEmbeddingsPath.get(i) != null)
            {
                inputVocabulary.add(loadEmbeddings(inputEmbeddingsPath.get(i)));
            }
            else
            {
                inputVocabulary.add(createNewInputVocabulary());
            }
            inputVocabularyCurrentIndex.add(inputVocabulary.get(inputVocabulary.size() - 1).size());
        }
    }

    private void initOutputVocabulary() throws IOException
    {
        for (int i = 0 ; i < outputFeatures ; i++)
        {
            if (fixedOutputVocabularyPath.get(i) != null)
            {
                outputVocabulary.add(readVocabulary(fixedOutputVocabularyPath.get(i)));
            }
            else
            {
                outputVocabulary.add(createNewOutputVocabulary());
            }
            currentOutputVocabularyIndex.add(outputVocabulary.get(outputVocabulary.size() - 1).size());
        }
    }

    private Map<String, Integer> loadEmbeddings(String embeddingsPath)
    {
        WordVectors embeddings = new TextualModelLoader(false).loadVocabularyOnly(embeddingsPath);
        Map<String, Integer> vocabulary = createNewInputVocabulary();
        int i = 2;
        for (String vocab : embeddings.getVocabulary())
        {
            vocabulary.put(vocab, i);
            i++;
        }
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
        return vocabulary;
    }

    private Map<String, Integer> createNewOutputVocabulary()
    {
        Map<String, Integer> vocabulary = new HashMap<>();
        vocabulary.put(skipToken, 0);
        return vocabulary;
    }

    private void writeVocabulary(Map<String, Integer> vocabulary, String vocabularyPath) throws IOException
    {
        BufferedWriter out = Files.newBufferedWriter(Paths.get(vocabularyPath));
        for (Map.Entry<String, Integer> vocab : vocabulary.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toList()))
        {
            out.write("" + vocab.getValue() + " " + vocab.getKey() + "\n");
        }
        out.close();
    }

    private void writeConfigFile(String configFilePath) throws IOException
    {
        Map<Object, Object> config = new LinkedHashMap<>();

        config.put("input_features", inputFeatures);
        config.put("input_annotation_name", inputAnnotationName);
        config.put("input_embeddings_path", inputEmbeddingsPath.stream().map(p -> p != null ? Paths.get(p).toAbsolutePath().toString() : null).collect(Collectors.toList()));
        config.put("input_embeddings_size", inputEmbeddingsPath.stream().map(p -> p != null ? null : 300).collect(Collectors.toList()));
        config.put("output_features", outputFeatures);
        config.put("output_annotation_name", outputAnnotationName);
        config.put("lstm_units_size", 1000);
        config.put("lstm_layers", 1);
        config.put("linear_before_lstm", false);
        config.put("dropout_rate_before_lstm", null);
        config.put("dropout_rate", 0.5);
        config.put("word_dropout_rate", null);
        config.put("attention_layer", false);
        config.put("legacy_model", false);

        BufferedWriter out = Files.newBufferedWriter(Paths.get(configFilePath));
        Json.write(out, config);
        out.close();
    }
    
    private List<Sentence> preprocessCorpora(List<String> originalCorpusPaths, boolean vocabularyIsFixed)
    {
        List<Sentence> allSentences = new ArrayList<>();
        StreamingCorpusReaderSentence reader = new StreamingCorpusReaderSentence()
        {
            @Override
            public void readSentence(Sentence s)
            {
                List<Word> words = s.getWords();

                /// truncate lines too long
                if (words.size() > maxLineLength)
                {
                    words = words.subList(0, maxLineLength);
                }

                /// filtering out sentences with no output features
                boolean sentenceHasOutputFeatures = false;
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
                        String wordKey = w.getAnnotationValue("lemma") + "%" + POSConverter.toWNPOS(w.getAnnotationValue("pos"));

                        if (removeMonosemics && wn.getSenseKeyListFromWordKey(wordKey).size() == 1)
                        {
                            w.removeAnnotation(senseTag);
                        }

                        List<String> senseKeys = w.getAnnotationValues(senseTag, ";");
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

                    /// check if any word contains any output feature 
                    /// and construct output vocabulary
                    for (int i = 0 ; i < outputFeatures ; i++)
                    {
                        Map<String, Integer> featureVocabulary = outputVocabulary.get(i);
                        List<String> featureValues = w.getAnnotationValues(outputAnnotationName.get(i), ";");
                        if (!featureValues.isEmpty())
                        {
                            for (String featureValue : featureValues)
                            {
                                if (featureVocabulary.containsKey(featureValue))
                                {
                                    sentenceHasOutputFeatures = true;
                                }
                                else if (!vocabularyIsFixed && fixedOutputVocabularyPath.get(i) == null)
                                {
                                    int currentIndex = currentOutputVocabularyIndex.get(i);
                                    featureVocabulary.put(featureValue, currentIndex);
                                    currentOutputVocabularyIndex.set(i, currentIndex + 1);
                                    sentenceHasOutputFeatures = true;
                                }
                            }
                        }
                    }
                }

                /// skip this sentence
                if (!sentenceHasOutputFeatures) return;

                for (Word w : words)
                {
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

                    // construct input vocabulary
                    for (int i = 0 ; i < inputFeatures ; i++)
                    {
                        String featureValue = w.getAnnotationValue(inputAnnotationName.get(i));
                        Map<String, Integer> featureVocabulary = inputVocabulary.get(i);
                        if (!featureValue.isEmpty() && !featureVocabulary.containsKey(featureValue) && !vocabularyIsFixed && inputEmbeddingsPath.get(i) == null)
                        {
                            int currentIndex = inputVocabularyCurrentIndex.get(i);
                            featureVocabulary.put(featureValue, currentIndex);
                            inputVocabularyCurrentIndex.set(i, currentIndex + 1);
                        }
                    }
                }

                /// add the sentence
                allSentences.add(new Sentence(new ArrayList<>(words)));
            }
        };

        for (String originalCorpusPath : originalCorpusPaths)
        {
            reader.load(originalCorpusPath);
        }

        return allSentences;
    }

    private List<Sentence> removeDuplicates(List<Sentence> sentences)
    {
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

    private List<Sentence> extractDevSentences(List<Sentence> trainSentences, int count)
    {
        if (count <= 0) return Collections.emptyList();
        Collections.shuffle(trainSentences);
        List<Sentence> subPartOfTrainSentences = trainSentences.subList(0, count);
        List<Sentence> devSentences = new ArrayList<>(subPartOfTrainSentences);
        subPartOfTrainSentences.clear();
        return devSentences;
    }

    private void writeCorpus(List<Sentence> sentences, String corpusPath) throws Exception
    {
        Wrapper<BufferedWriter> writer = new Wrapper<>();
        writer.obj = Files.newBufferedWriter(Paths.get(corpusPath));
        for (Sentence s : sentences)
        {
            List<Word> words = s.getWords();

            /// step 1 : write input features
            for (Word w : words)
            {
                List<String> featureValues = new ArrayList<>();
                for (int i = 0 ; i < inputFeatures ; i++)
                {
                    String featureValue = w.getAnnotationValue(inputAnnotationName.get(i));
                    Map<String, Integer> featureVocabulary = inputVocabulary.get(i);
                    if (featureValue.isEmpty() || !featureVocabulary.containsKey(featureValue))
                    {
                        featureValue = unknownToken;
                    }                       
                    featureValue = Integer.toString(featureVocabulary.get(featureValue));
                    featureValues.add(featureValue);
                }
                writer.obj.write(StringUtils.join(featureValues, "/") + " ");
            }
            writer.obj.newLine();

            /// step 3 : write output features
            for (Word w : words)
            {
                List<String> featureValues = new ArrayList<>();
                for (int i = 0 ; i < outputFeatures ; i++)
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

            /// step 4 : write output (sense_tag) restrictions
            for (Word w : words)
            {
                for (int i = 0 ; i < outputFeatures ; i++)
                {
                    if (i > 0)
                    {
                        writer.obj.write("/");
                    }
                    if (outputFeatureSenseIndex == i && w.hasAnnotation(senseTag) && w.getAnnotationValues(senseTag, ";").stream().anyMatch(outputVocabulary.get(i)::containsKey))
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
                            if (outputVocabulary.get(i).containsKey(synsetKey))
                            {
                                restrictedSenses.add("" + outputVocabulary.get(outputFeatureSenseIndex).get(synsetKey));
                            }
                        }
                        writer.obj.write(StringUtils.join(restrictedSenses, ";"));
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
        writer.obj.close();
    }
}

