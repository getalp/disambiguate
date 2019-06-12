package getalp.wsd.ufsac.simple.core;

public class Word extends LexicalEntity
{    
    public Word()
    {
        super();
    }

	public Word(String value)
	{
		setAnnotation("surface_form", value);
	}

	public Word(getalp.wsd.ufsac.core.Word wordToCopy)
    {
        super(wordToCopy);
    }

	public void setValue(String value)
	{
	    setAnnotation("surface_form", value);
	}

	public String getValue()
	{
		return getAnnotationValue("surface_form");
	}

	public Word clone()
	{
		Word copy = new Word();
		transfertAnnotationsToCopy(copy);
		return copy;
	}
	
	public String toString()
	{
		return getAnnotationValue("surface_form");
	}
}
