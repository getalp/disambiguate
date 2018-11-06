package getalp.wsd.method;

import java.util.List;
import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Paragraph;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;

public abstract class Disambiguator
{
    /**
     * Disambiguates and add sense tags to words in a corpus
     */
    public final void disambiguate(Corpus corpus, String newSenseTags)
    {
        disambiguate(corpus, newSenseTags, null);
    }

    /**
     * Disambiguates and add sense tags to words in a corpus
     * Adds a confidence value
     */
    public abstract void disambiguate(Corpus corpus, String newSenseTags, String confidenceTag);

    /**
     * Disambiguates and add sense tags to words in a document
     */
    public final void disambiguate(Document document, String newSenseTags)
    { 
        disambiguate(document, newSenseTags, null);
    }

    /**
     * Disambiguates and add sense tags to words in a document
     * Adds a confidence value
     */
    public abstract void disambiguate(Document document, String newSenseTags, String confidenceTag);

    /**
     * Disambiguates and add sense tags to words in a paragraph
     */
    public final void disambiguate(Paragraph paragraph, String newSenseTags)
    { 
        disambiguate(paragraph, newSenseTags, null);
    }

    /**
     * Disambiguates and add sense tags to words in a paragraph
     * Adds a confidence value
     */
    public abstract void disambiguate(Paragraph paragraph, String newSenseTags, String confidenceTag);
 
    /**
     * Disambiguates and add sense tags to words in a sentence
     */
    public final void disambiguate(Sentence sentence, String newSenseTags)
    {
        disambiguate(sentence, newSenseTags, null);
    }

    /**
     * Disambiguates and add sense tags to words in a sentence
     * Adds a confidence value
     */
    public abstract void disambiguate(Sentence sentence, String newSenseTags, String confidenceTag);

    /**
     * Disambiguates and add sense tags to a list of words
     */
    public final void disambiguate(List<Word> words, String newSenseTags)
    {
        disambiguate(words, null);
    }

    /**
     * Disambiguates and add sense tags to a list of words
     * Adds a confidence value
     */
    public abstract void disambiguate(List<Word> words, String newSenseTags, String confidenceTag);
    
}
