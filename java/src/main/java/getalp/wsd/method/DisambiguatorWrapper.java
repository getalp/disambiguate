package getalp.wsd.method;

import java.util.Arrays;
import java.util.List;
import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Paragraph;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;

public class DisambiguatorWrapper extends Disambiguator
{
    private List<Disambiguator> disambiguators;

    public DisambiguatorWrapper(Disambiguator... disambiguators)
    {
        this.disambiguators = Arrays.asList(disambiguators);
    }

    @Override
    public void disambiguate(Corpus corpus, String newSenseTags, String confidenceTag)
    {
        for (Disambiguator disambiguator : disambiguators)
        {
            disambiguator.disambiguate(corpus, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(Document document, String newSenseTags, String confidenceTag)
    {
        for (Disambiguator disambiguator : disambiguators)
        {
            disambiguator.disambiguate(document, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(Paragraph paragraph, String newSenseTags, String confidenceTag)
    {
        for (Disambiguator disambiguator : disambiguators)
        {
            disambiguator.disambiguate(paragraph, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(Sentence sentence, String newSenseTags, String confidenceTag)
    {
        for (Disambiguator disambiguator : disambiguators)
        {
            disambiguator.disambiguate(sentence, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(List<Word> words, String newSenseTags, String confidenceTag)
    {
        for (Disambiguator disambiguator : disambiguators)
        {
            disambiguator.disambiguate(words, newSenseTags, confidenceTag);
        }
    }
}
