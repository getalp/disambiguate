package getalp.wsd.method;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import getalp.wsd.common.utils.POSConverter;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.ufsac.core.Word;

public class FirstSenseDisambiguator extends DisambiguatorContextCorpus
{
    private WordnetHelper wn;

    private Map<String, String> mfs;

    public FirstSenseDisambiguator(WordnetHelper wn)
    {
        this.wn = wn;
        this.mfs = null;
    }

    public FirstSenseDisambiguator(String mfsPath)
    {
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(mfsPath));
            String line = "";
            mfs = new HashMap<>();
            while ((line = br.readLine()) != null)
            {
                mfs.put(line.split(" ")[0], line.split(" ")[1]);
            }
            br.close();
            wn = null;
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

	@Override
	public void disambiguate(List<Word> words, String newSenseTags, String confidenceTag)
	{
		for (Word w : words)
		{
			if (!w.hasAnnotation(newSenseTags))
			{
                String pos = POSConverter.toWNPOS(w.getAnnotationValue("pos"));
                List<String> lemmas = w.getAnnotationValues("lemma", ";");
                //Collections.reverse(lemmas);
			    for (String lemma : lemmas)
			    {
			        String wordKey = lemma + "%" + pos;
    				if (wn.isWordKeyExists(wordKey))
    				{		
			            String senseKey = "";
    	                if (wn != null)
    	                {
    	                    senseKey = wn.getFirstSenseKeyFromWordKey(wordKey);
    	                }
    	                else
    	                {
    	                    senseKey = mfs.get(wordKey);
    	                }				
    					w.setAnnotation(newSenseTags, senseKey);	
    					break;
    				}
			    }
			}
		}
	}
}
