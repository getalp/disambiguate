package getalp.wsd.evaluation;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.result.MultipleDisambiguationResult;
import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.utils.ObjectUsingSystemOutALot;
import getalp.wsd.method.result.DisambiguationResult;


public class WSDEvaluator extends ObjectUsingSystemOutALot
{
    private boolean printFailed;
    
    private boolean printResults;
    
    public WSDEvaluator()
    {
        printFailed = false;
        printResults = false;
    }
    
    public void setPrintFailed(boolean printFailed)
    {
        this.printFailed = printFailed;
    }
    
    public MultipleDisambiguationResult evaluate(Disambiguator disambiguator, String corpusPath, String senseAnnotationTag, int n)
    {
        return evaluate(disambiguator, Corpus.loadFromXML(corpusPath), senseAnnotationTag, n);
    }
    
    public MultipleDisambiguationResult evaluate(Disambiguator disambiguator, Corpus corpus, String senseAnnotationTag, int n)
    {
        MultipleDisambiguationResult results = new MultipleDisambiguationResult();
        println("WSD " + disambiguator);
        for (int i = 0 ; i < n ; i++)
        {
            print("" + (i+1) + "/" + n + " ");
            DisambiguationResult totalScore = evaluate(disambiguator, corpus, senseAnnotationTag);
            results.addDisambiguationResult(totalScore);
        }
        println();
        println("WSD " + disambiguator);
        println("Mean Scores : " + results.scoreMean());
        println("Standard Deviation Scores : " + results.scoreStandardDeviation());
        println("Mean Times : " + results.timeMean());
        println();
        return results;
    }

    public DisambiguationResult evaluate(Disambiguator disambiguator, String corpusPath, String senseAnnotationTag)
    {
        return evaluate(disambiguator, Corpus.loadFromXML(corpusPath), senseAnnotationTag);
    }
    
    public DisambiguationResult evaluate(Disambiguator disambiguator, Corpus corpus, String senseAnnotationTag)
    {
        DisambiguationResult totalScore = new DisambiguationResult();
        long startTime = System.currentTimeMillis();
        for (Document document : corpus.getDocuments())
        {
            print("(" + document.getAnnotationValue("id") + ") ");
            disambiguator.disambiguate(document, "wsd_test");
            DisambiguationResult documentScore = computeDisambiguationResult(document.getWords(), senseAnnotationTag, "wsd_test");
            double documentScoreRatioPercent = documentScore.scoreF1();
            print("[" + String.format("%.2f", documentScoreRatioPercent) + "] ");
            totalScore.concatenateResult(documentScore);
        }
        long endTime = System.currentTimeMillis();
        double time = (endTime - startTime) / 1000.0;
        totalScore.time = time;
        print("; good/bad/missed/total : " + totalScore.good + "/" + totalScore.bad + "/" + totalScore.missed + "/" + totalScore.total);
        println(" ; C/P/R/F1 : " + String.format("%.4f", totalScore.coverage()) + "/" + String.format("%.4f", totalScore.scorePrecision()) + "/" + String.format("%.4f", totalScore.scoreRecall()) + "/" + String.format("%.4f", totalScore.scoreF1()) + " ; time : " + totalScore.time + " seconds");
        printFailed();
        saveResultToFile(corpus.getDocuments(), "wsd_test");
        return totalScore;
    }
    
    public DisambiguationResult computeDisambiguationResult(List<Word> wordList, String referenceSenseTag, String candidateSenseTag)
    {
        return computeDisambiguationResult(wordList, referenceSenseTag, candidateSenseTag, null, 0);
    }
    
    public DisambiguationResult computeDisambiguationResult(List<Word> wordList, String referenceSenseTag, String candidateSenseTag, String confidenceValueTag, double confidenceThreshold)
    {
        return computeDisambiguationResult(wordList, referenceSenseTag, candidateSenseTag, confidenceValueTag, confidenceThreshold, WordnetHelper.wn30());
    }
    
    public DisambiguationResult computeDisambiguationResult(List<Word> wordList, String referenceSenseTag, String candidateSenseTag, String confidenceValueTag, double confidenceThreshold, WordnetHelper wn)
    {
        int total = 0;
        int good = 0;
        int bad = 0;
        for (int i = 0 ; i < wordList.size() ; i++)
        {
            Word word = wordList.get(i);
            
            List<String> referenceSenseKeys = word.getAnnotationValues(referenceSenseTag, ";");
            if (referenceSenseKeys.isEmpty()) continue;
            
            List<String> referenceSynsetKeys = new ArrayList<>();
            for (String refSenseKey : referenceSenseKeys)
            {
                refSenseKey = refSenseKey.toLowerCase();
                if (!wn.isSenseKeyExists(refSenseKey)) continue;
                String refSynsetKey = wn.getSynsetKeyFromSenseKey(refSenseKey);
                if (!referenceSynsetKeys.contains(refSynsetKey))
                {
                    referenceSynsetKeys.add(refSynsetKey);
                }
            }
            if (referenceSynsetKeys.isEmpty()) continue;
            
            total += 1;
            
            String candidateSenseKey = word.getAnnotationValue(candidateSenseTag);
            if (candidateSenseKey.isEmpty()) continue;
            candidateSenseKey = candidateSenseKey.toLowerCase();
            if (!wn.isSenseKeyExists(candidateSenseKey)) continue;
            if (word.hasAnnotation(confidenceValueTag))
            {
                double confidenceValue = Double.valueOf(word.getAnnotationValue(confidenceValueTag));
                if (confidenceValue != Double.POSITIVE_INFINITY && confidenceValue < confidenceThreshold) continue;
            }
            String candidateSynsetKey = wn.getSynsetKeyFromSenseKey(candidateSenseKey);
            bad += 1;
            for (String refSynsetKey : referenceSynsetKeys)
            {
                if (refSynsetKey.equals(candidateSynsetKey))
                {
                    good += 1;
                    bad -= 1;
                    break;
                }
            }
        }
        return new DisambiguationResult(total, good, bad);
    }
    
    private void saveResultToFile(List<Document> documents, String candidateSenseTag)
    {
        if (printResults)
        {
            try
            {
                int index = 0;
                Path pathToIndex = Paths.get("data/output/index");
                if (!Files.isRegularFile(pathToIndex))
                {
                    Files.createFile(pathToIndex);
                    Files.write(pathToIndex, "0".getBytes(StandardCharsets.UTF_8));
                }
                index = Integer.valueOf(new String(Files.readAllBytes(Paths.get("data/output/index")), StandardCharsets.UTF_8));
                
                PrintStream ps = new PrintStream("data/output/" + index + ".ans");
        
                for (int i = 0 ; i < documents.size() ; i++)
                {
                    String documentId = documents.get(i).getAnnotationValue("id");
                    List<Word> wordList = documents.get(i).getWords();
                    for (int j = 0 ; j < wordList.size() ; j++) 
                    {
                        Word word = wordList.get(j);
                        String wordId = word.getAnnotationValue("id");
                        String wordSenseTag = word.getAnnotationValue(candidateSenseTag);
                        if (!wordId.equals("") && !wordSenseTag.equals(""))
                        {
                           ps.printf("%s %s %s \n", documentId, wordId, wordSenseTag);
                        }
                    }
                }
                
                ps.close();

                Files.write(pathToIndex, Integer.toString(index + 1).getBytes(StandardCharsets.UTF_8));
            }
            catch (Exception e)
            {
                throw new RuntimeException(e);
            }
        }
    }
    
    private void printFailed()
    {
        if (printFailed)
        {
            // System.out.println();
            // perfectScorer.printFailed(document, configuration);
        }
    }
}
