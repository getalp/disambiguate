package getalp.wsd.method;

import java.util.List;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.ufsac.core.Word;

public class MonosemicDisambiguator extends DisambiguatorContextCorpus
{
    private WordnetHelper wn;

    public MonosemicDisambiguator(WordnetHelper wn)
    {
        this.wn = wn;
    }

	@Override
	public void disambiguate(List<Word> words, String newSenseTags, String confidenceTag)
	{
		for (Word w : words)
		{
		    if (w.hasAnnotation(newSenseTags)) continue;
            String pos = POSConverter.toWNPOS(w.getAnnotationValue("pos"));
	        for (String lemma : w.getAnnotationValues("lemma", ";"))
	        {
				String wordKey = lemma + "%" + pos;
				if (wn.isWordKeyExists(wordKey))
				{
				    List<String> senseKeys = wn.getSenseKeyListFromWordKey(wordKey);
				    if (senseKeys.size() == 1)
				    {   
	                    w.setAnnotation(newSenseTags, senseKeys.get(0));  
	                    break;
				    }
				}
	        }
		}
	}
}
