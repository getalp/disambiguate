package getalp.wsd.method;

import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Paragraph;
import getalp.wsd.ufsac.core.Sentence;

public abstract class DisambiguatorContextCorpus extends Disambiguator
{
    @Override
    public void disambiguate(Corpus corpus, String newSenseTags, String confidenceTag)
    {
        disambiguate(corpus.getWords(), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(Document document, String newSenseTags, String confidenceTag)
    {
        disambiguate(document.getWords(), newSenseTags, confidenceTag);
    }

    @Override
    public void disambiguate(Paragraph paragraph, String newSenseTags, String confidenceTag)
    {
        disambiguate(paragraph.getWords(), newSenseTags, confidenceTag);
    }
 
    @Override
    public void disambiguate(Sentence sentence, String newSenseTags, String confidenceTag)
    {
        disambiguate(sentence.getWords(), newSenseTags, confidenceTag);
    }
}
