package getalp.wsd.method;

import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Paragraph;
import getalp.wsd.ufsac.core.Sentence;

public abstract class DisambiguatorContextSentence extends Disambiguator
{
    @Override
    public void disambiguate(Corpus corpus, String newSenseTags, String confidenceTag)
    {
        for (Document d : corpus.getDocuments())
        {
            disambiguate(d, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(Document document, String newSenseTags, String confidenceTag)
    {
        for (Paragraph p : document.getParagraphs())
        {
            disambiguate(p, newSenseTags, confidenceTag);
        }
    }

    @Override
    public void disambiguate(Paragraph paragraph, String newSenseTags, String confidenceTag)
    {
        for (Sentence s : paragraph.getSentences())
        {
            disambiguate(s, newSenseTags, confidenceTag);
        }
    }
 
    @Override
    public void disambiguate(Sentence sentence, String newSenseTags, String confidenceTag)
    {
        disambiguate(sentence.getWords(), newSenseTags, confidenceTag);
    }
}
