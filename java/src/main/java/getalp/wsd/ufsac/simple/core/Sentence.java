package getalp.wsd.ufsac.simple.core;

import getalp.wsd.common.utils.RegExp;

import java.util.ArrayList;
import java.util.List;

public class Sentence extends LexicalEntity
{
	List<Word> words = new ArrayList<>();

    public Sentence()
    {
        super();
    }

    public Sentence(getalp.wsd.ufsac.core.Sentence sentenceToCopy)
    {
        super(sentenceToCopy);
        for (getalp.wsd.ufsac.core.Word word : sentenceToCopy.getWords())
        {
            this.addWord(new Word(word));
        }
    }

	public Sentence(String value)
	{
	    addWordsFromString(value);
	}

    public Sentence(List<Word> words)
    {
        for (Word word : new ArrayList<>(words))
        {
            addWord(word);
        }
    }

    public void addWord(Word word)
	{
	    words.add(word);
	}
	
	public void removeWord(Word word)
	{
	    words.remove(word);
	}
	
	public void removeAllWords()
	{
	    words.clear();
	}
	
    public List<Word> getWords()
	{
		return words;
	}

	public void limitSentenceLength(int maxLength)
    {
        if (words.size() > maxLength)
        {
            words = words.subList(0, maxLength);
        }
    }

    public Sentence clone()
    {
        Sentence newSentence = new Sentence();
        transfertWordsToCopy(newSentence);
        transfertAnnotationsToCopy(newSentence);
        return newSentence;
    }
    
    public void transfertWordsToCopy(Sentence other)
    {
        for (Word word : getWords())
        {
            other.addWord(word.clone());
        }
    }
    
    public void addWordsFromString(String value)
    {
        String[] wordsArray = value.split(RegExp.anyWhiteSpaceGrouped.toString());
        for (String wordInArray : wordsArray)
        {
            addWord(new Word(wordInArray));
        }
    }
    
	public String toString()
	{
		String ret = "";
		for (Word word : getWords())
		{
			ret += word.toString() + " ";
		}
		return ret.trim();
	}
}
