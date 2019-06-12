package getalp.wsd.method;

import getalp.wsd.ufsac.core.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class DisambiguatorContextSentenceBatch extends Disambiguator
{
    private int batchSize;

    public DisambiguatorContextSentenceBatch(int batchSize)
    {
        this.batchSize = batchSize;
    }

    @Override
    public void disambiguate(Corpus corpus, String newSenseTags, String confidenceTag)
    {
        disambiguateDynamicSentenceBatch(corpus.getSentences(), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(Document document, String newSenseTags, String confidenceTag)
    {
        disambiguateDynamicSentenceBatch(document.getSentences(), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(Paragraph paragraph, String newSenseTags, String confidenceTag)
    {
        disambiguateDynamicSentenceBatch(paragraph.getSentences(), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(Sentence sentence, String newSenseTags, String confidenceTag)
    {
        disambiguateDynamicSentenceBatch(Collections.singletonList(sentence), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(List<Word> words, String newSenseTags, String confidenceTag)
    {
        disambiguateDynamicSentenceBatch(Collections.singletonList(new Sentence(words)), newSenseTags, confidenceTag);
    }

    public void disambiguateDynamicSentenceBatch(List<Sentence> originalSentences, String newSenseTags, String confidenceTag)
    {
        List<Sentence> sentences = new ArrayList<>(originalSentences);
        while (sentences.size() > batchSize)
        {
            List<Sentence> subSentences = sentences.subList(0, batchSize);
            disambiguateFixedSentenceBatch(subSentences, newSenseTags, confidenceTag);
            subSentences.clear();
        }
        if (!sentences.isEmpty())
        {
            int paddingSize = batchSize - sentences.size();
            for (int i = 0 ; i < paddingSize ; i++)
            {
                sentences.add(new Sentence("<pad>"));
            }
            disambiguateFixedSentenceBatch(sentences, newSenseTags, confidenceTag);
        }
    }

    protected abstract void disambiguateFixedSentenceBatch(List<Sentence> sentences, String newSenseTags, String confidenceTag);
}
