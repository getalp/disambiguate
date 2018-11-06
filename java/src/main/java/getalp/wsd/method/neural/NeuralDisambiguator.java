package getalp.wsd.method.neural;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.utils.RegExp;
import getalp.wsd.common.utils.StringUtils;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.DisambiguatorContextSentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.utils.Json;

public class NeuralDisambiguator extends DisambiguatorContextSentence implements AutoCloseable
{
    private static final String unknownToken = "<unk>";
    
    private WordnetHelper wn = WordnetHelper.wn30();

    private int inputFeatures;
    
    private List<String> inputAnnotationNames;
    
    private List<Map<String, Integer>> inputVocabulary;
    
    private int outputFeatures;

    // private List<String> outputAnnotationNames;
    
    private List<Map<String, Integer>> outputVocabulary;
    
    private List<Map<Integer, String>> reversedOutputVocabulary;

    private Process pythonProcess = null;
    
    private BufferedReader pythonProcessReader = null;
    
    private BufferedWriter pythonProcessWriter = null;

    // --- begin public options

    public boolean lowercaseWords = true;
        
    public Map<String, String> reducedOutputVocabulary = null;
            
    // --- end public options

    public NeuralDisambiguator(String pythonPath, String neuralPath, String weightsPath)
    {
        this(pythonPath, neuralPath, Collections.singletonList(weightsPath));
    }

    public NeuralDisambiguator(String pythonPath, String neuralPath, List<String> weightsPaths)
    {
        try
        {
        	initPythonProcess(pythonPath, neuralPath, weightsPaths);
        	readConfigFile(neuralPath);
            initInputVocabulary(neuralPath);
            initOutputVocabulary(neuralPath);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

    private void initPythonProcess(String pythonPath, String neuralPath, List<String> weightsPaths) throws IOException
    {
        List<String> args = new ArrayList<>(Arrays.asList(pythonPath + "/launch.sh", "getalp.wsd.predict", "--data_path", neuralPath, "--weights"));
        args.addAll(weightsPaths);
        ProcessBuilder pb = new ProcessBuilder(args);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        pythonProcess = pb.start();
        pythonProcessReader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
        pythonProcessWriter = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
    }
    
    @SuppressWarnings("unchecked")
	private void readConfigFile(String neuralPath) throws IOException
    {
    	Map<Object, Object> config = Json.readMap(neuralPath + "/config.json");
    	inputFeatures = (int) config.get("input_features");
    	inputAnnotationNames = (List<String>) config.get("input_annotation_name");
    	outputFeatures = (int) config.get("output_features");
    	// outputAnnotationNames = (List<String>) config.get("output_annotation_name");
    }
    
    private void initInputVocabulary(String neuralPath) throws Exception
    {
    	inputVocabulary = new ArrayList<>();
        for (int i = 0 ; i < inputFeatures ; i++)
        {
            inputVocabulary.add(initVocabulary(neuralPath + "/input_vocabulary" + i));
        }
    }

    private void initOutputVocabulary(String neuralPath) throws Exception
    {
    	outputVocabulary = new ArrayList<>();
    	reversedOutputVocabulary = new ArrayList<>();
        for (int i = 0 ; i < outputFeatures ; i++)
        {
        	Map<String, Integer> vocabulary = initVocabulary(neuralPath + "/output_vocabulary" + i);
        	Map<Integer, String> reversedVocabulary = new HashMap<>();
        	for (String key : vocabulary.keySet())
        	{
        		reversedVocabulary.put(vocabulary.get(key), key);
        	}
        	outputVocabulary.add(vocabulary);
        	reversedOutputVocabulary.add(reversedVocabulary);
        }
    }

    private HashMap<String, Integer> initVocabulary(String filePath) throws Exception
    {
        HashMap<String, Integer> ret = new HashMap<>();
        BufferedReader reader = Files.newBufferedReader(Paths.get(filePath));
        reader.lines().forEach(line ->
        {
            String[] tokens = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
            ret.put(tokens[1], Integer.valueOf(tokens[0]));
        });
        reader.close();
        return ret;
    }

    private void writePredictInput(List<Word> words) throws Exception
    {
        writePredictInputSampleX(words);
        writePredictInputSampleZ(words);
        pythonProcessWriter.flush();
    }

    private void writePredictInputSampleX(List<Word> words) throws Exception
    {
        for (Word w : words)
        {
            if (lowercaseWords)
            {
                w.setValue(w.getValue().toLowerCase());
            }
            List<String> featureValues = new ArrayList<>();
            for (int i = 0 ; i < inputFeatures ; i++)
            {
                String featureValue = w.getAnnotationValue(inputAnnotationNames.get(i));
                Map<String, Integer> featureVocabulary = inputVocabulary.get(i);
                if (featureValue.isEmpty() || !featureVocabulary.containsKey(featureValue))
                {
                    featureValue = unknownToken;
                }                       
                featureValue = Integer.toString(featureVocabulary.get(featureValue));
                featureValues.add(featureValue);
            }
            pythonProcessWriter.write(StringUtils.join(featureValues, "/") + " ");
        }
        pythonProcessWriter.newLine();
    }

    private void writePredictInputSampleZ(List<Word> words) throws Exception
    {
        for (Word word : words)
        {
            List<String> possibleSenseKeys = new ArrayList<>();
            
            if (word.hasAnnotation("lemma") && word.hasAnnotation("pos"))
            {
                String pos = POSConverter.toWNPOS(word.getAnnotationValue("pos"));
                List<String> lemmas = word.getAnnotationValues("lemma", ";");
                for (String lemma : lemmas)
                {
                    String wordKey = lemma + "%" + pos;
                    if (!wn.isWordKeyExists(wordKey)) continue;
                    possibleSenseKeys.addAll(wn.getSenseKeyListFromWordKey(wordKey));
                }
            }
            
            if (!possibleSenseKeys.isEmpty())
            {
                List<String> possibleSenseKeyIndices = new ArrayList<>();
                for (String possibleSenseKey : possibleSenseKeys)
                {
                    String possibleSynsetKey = wn.getSynsetKeyFromSenseKey(possibleSenseKey);
                    if (reducedOutputVocabulary != null)
                    {
                        possibleSynsetKey = reducedOutputVocabulary.getOrDefault(possibleSynsetKey, possibleSynsetKey);
                    }
                    if (outputVocabulary.get(0).containsKey(possibleSynsetKey))
                    {
                        possibleSenseKeyIndices.add("" + outputVocabulary.get(0).get(possibleSynsetKey));
                    }
                }
                if (possibleSenseKeyIndices.isEmpty())
                {
                    pythonProcessWriter.write("0 ");
                }
                else
                {
                    pythonProcessWriter.write(StringUtils.join(possibleSenseKeyIndices, ";") + " ");
                }
            }
            else
            {
                pythonProcessWriter.write("0 ");
            }
        }
        pythonProcessWriter.newLine();
    }
    
    private void readPredictOutput(List<Word> words, String senseTag, String confidenceTag) throws Exception
    {
    	String line = pythonProcessReader.readLine();
    	int[] output = parsePredictOutput(line);
    	propagatePredictOutput(words, output, senseTag, confidenceTag);
    }

    private int[] parsePredictOutput(String line)
    {
        String[] lineSplit = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
        int[] output = new int[lineSplit.length];
        for (int i = 0 ; i < lineSplit.length ; i++)
        {
            output[i] = Integer.valueOf(lineSplit[i]);
        }
        return output;
    }
    
    private void propagatePredictOutput(List<Word> words, int[] output, String senseTag, String confidenceTag)
    {
        for (int i = 0 ; i < output.length ; i++)
        {
            Word word = words.get(i);
            if (word.hasAnnotation(senseTag)) continue;
            if (!word.hasAnnotation("lemma")) continue;
            if (!word.hasAnnotation("pos")) continue;
            int wordOutput = output[i];
            String pos = POSConverter.toWNPOS(word.getAnnotationValue("pos"));
            List<String> lemmas = word.getAnnotationValues("lemma", ";");
            for (String lemma : lemmas)
            {
                String wordKey = lemma + "%" + pos;
                if (!wn.isWordKeyExists(wordKey)) continue;
                List<String> lemmaSenseKeys = wn.getSenseKeyListFromWordKey(wordKey);
                for (String possibleSenseKey : lemmaSenseKeys)
                {
                    String possibleSynsetKey = wn.getSynsetKeyFromSenseKey(possibleSenseKey);
                    if (reducedOutputVocabulary != null)
                    {
                        possibleSynsetKey = reducedOutputVocabulary.getOrDefault(possibleSynsetKey, possibleSynsetKey);
                    }
                    if (reversedOutputVocabulary.get(0).get(wordOutput).equals(possibleSynsetKey))
                    {
                        word.setAnnotation(senseTag, possibleSenseKey);
                        //word.setAnnotation(confidenceTag, confidenceInfo);
                    }
                }
            }
        }
    }

    private void disambiguateNoCatch(List<Word> words, String senseTag, String confidenceTag) throws Exception
    {
        writePredictInput(words);
        readPredictOutput(words, senseTag, confidenceTag);
    }
    
    @Override
    public void disambiguate(List<Word> words, String senseTag, String confidenceTag)
    {
        try
        {
            disambiguateNoCatch(words, senseTag, confidenceTag);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws Exception
    {
        pythonProcessReader.close();
        pythonProcessWriter.close();
        pythonProcess.destroyForcibly();
    }
}
