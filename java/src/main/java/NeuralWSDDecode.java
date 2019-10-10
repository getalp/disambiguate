import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.FirstSenseDisambiguator;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.utils.CorpusPOSTaggerAndLemmatizer;
import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.zeromq.ZMQ;
import org.zeromq.ZContext;

public class NeuralWSDDecode
{
    public static void main(String[] args) throws Exception
    {
        new NeuralWSDDecode().decode(args);
    }

    private boolean mfsBackoff;

    private Disambiguator firstSenseDisambiguator;

    private NeuralDisambiguator neuralDisambiguator;

    private int truncateMaxLength;

    private CorpusPOSTaggerAndLemmatizer tagger;

    private int batchSize;

    private boolean verbose;

    private void decode(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgument("lowercase", "false");
        parser.addArgument("sense_compression_hypernyms", "true");
        parser.addArgument("sense_compression_instance_hypernyms", "false");
        parser.addArgument("sense_compression_antonyms", "false");
        parser.addArgument("sense_compression_file", "");
        parser.addArgument("clear_text", "false");
        parser.addArgument("batch_size", "1");
        parser.addArgument("truncate_max_length", "150");
        parser.addArgument("mfs_backoff", "true");
        parser.addArgument("zmq", "false");
        parser.addArgument("verbose", "false");
        if (!parser.parse(args)) return;

        String pythonPath = parser.getArgValue("python_path");
        String dataPath = parser.getArgValue("data_path");
        List<String> weights = parser.getArgValueList("weights");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseCompressionHypernyms = parser.getArgValueBoolean("sense_compression_hypernyms");
        boolean senseCompressionInstanceHypernyms = parser.getArgValueBoolean("sense_compression_instance_hypernyms");
        boolean senseCompressionAntonyms = parser.getArgValueBoolean("sense_compression_antonyms");
        String senseCompressionFile = parser.getArgValue("sense_compression_file");
        boolean clearText = parser.getArgValueBoolean("clear_text");
        batchSize = parser.getArgValueInteger("batch_size");
        truncateMaxLength = parser.getArgValueInteger("truncate_max_length");
        mfsBackoff = parser.getArgValueBoolean("mfs_backoff");
        boolean zmq = parser.getArgValueBoolean("zmq");
        verbose = parser.getArgValueBoolean("verbose");

        Map<String, String> senseCompressionClusters = null;
        if (senseCompressionHypernyms || senseCompressionAntonyms)
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClusters(WordnetHelper.wn30(), senseCompressionHypernyms, senseCompressionInstanceHypernyms, senseCompressionAntonyms);
        }
        if (!senseCompressionFile.isEmpty())
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClustersFromFile(senseCompressionFile);
        }

        tagger = new CorpusPOSTaggerAndLemmatizer();
        firstSenseDisambiguator = new FirstSenseDisambiguator(WordnetHelper.wn30());
        neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize);
        neuralDisambiguator.lowercaseWords = lowercase;
        neuralDisambiguator.reducedOutputVocabulary = senseCompressionClusters;
        neuralDisambiguator.verbose = verbose;
        if (zmq)
        {
            ZMQLoop();
        }
        else
        {
            IOLoop();
        }
        neuralDisambiguator.close();
    }

    private void IOLoop() throws Exception
    {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out));
        List<Sentence> sentences = new ArrayList<>();
        for (String line = reader.readLine(); line != null ; line = reader.readLine())
        {
            Sentence sentence = new Sentence(line);
            if (sentence.getWords().size() > truncateMaxLength)
            {
               sentence.getWords().stream().skip(truncateMaxLength).collect(Collectors.toList()).forEach(sentence::removeWord);
            }
            tagger.tag(sentence.getWords());
            sentences.add(sentence);
            if (sentences.size() >= batchSize)
            {
                writer.write(decodeSentenceBatch(sentences));
                writer.flush();
                sentences.clear();
            }
        }
        writer.write(decodeSentenceBatch(sentences));
        writer.flush();
        writer.close();
        reader.close();
    }

    private void ZMQLoop() throws Exception
    {
        try (ZContext context = new ZContext()) {
            // Socket to talk to clients
            ZMQ.Socket socket = context.createSocket(ZMQ.REP);
            socket.bind("tcp://*:5555");

            List<Sentence> sentences = new ArrayList<>();
            while (!Thread.currentThread().isInterrupted()) {
                // Block until a message is received

                String req = new String(socket.recv(0), ZMQ.CHARSET);


                // Print the message
                if (verbose)
                {
                    System.out.println("Received: [" + req + "]");
                }

                Sentence sentence = new Sentence(req);
                if (sentence.getWords().size() > truncateMaxLength)
                {
                   sentence.getWords().stream().skip(truncateMaxLength).collect(Collectors.toList()).forEach(sentence::removeWord);
                }
                tagger.tag(sentence.getWords());
                sentences.add(sentence);
                if (sentences.size() >= batchSize)
                {
                    String resp = decodeSentenceBatch(sentences);
                    socket.send(resp.getBytes(ZMQ.CHARSET), 0);
                    sentences.clear();
                }
            }
        }
    }


    private String decodeSentenceBatch(List<Sentence> sentences)
    {
        String ret = "";
        neuralDisambiguator.disambiguateDynamicSentenceBatch(sentences, "wsd", "");
        for (Sentence sentence : sentences)
        {
            if (mfsBackoff)
            {
                firstSenseDisambiguator.disambiguate(sentence, "wsd");
            }
            for (Word word : sentence.getWords())
            {
                ret += word.getValue().replace("|", "/");
                if (word.hasAnnotation("lemma") && word.hasAnnotation("pos") && word.hasAnnotation("wsd"))
                {
                    ret += "|" + word.getAnnotationValue("wsd");
                }
                ret += " ";
            }
            ret += "\n";
        }
        return ret;
    }
}

