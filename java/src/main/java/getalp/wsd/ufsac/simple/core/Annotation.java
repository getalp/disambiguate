package getalp.wsd.ufsac.simple.core;

import getalp.wsd.common.utils.StringUtils;

import java.util.Arrays;
import java.util.List;

public class Annotation
{	
	private String annotationName;
	
	private String annotationValue;

    public Annotation(String name, String value)
    {
        if (name == null) this.annotationName = "";
        else this.annotationName = name;
        if (value == null) this.annotationValue = "";
        else this.annotationValue = value;
    }

    public Annotation(String name, List<String> values, String delimiter)
    {
        if (name == null) this.annotationName = "";
        else this.annotationName = name;
        if (values == null) this.annotationValue = "";
        else this.annotationValue = StringUtils.join(values, delimiter);
    }

    public String getAnnotationName()
    {
        return annotationName;
    }

    public String getAnnotationValue()
    {
        return annotationValue;
    }
    
    public void setAnnotationValue(String value)
    {
        if (value == null) this.annotationValue = "";
        else this.annotationValue = value;
    }

    public List<String> getAnnotationValues(String delimiter)
    {
        return Arrays.asList(annotationValue.split(delimiter));
    }

    public void setAnnotationValues(List<String> values, String delimiter)
    {
        if (values == null) this.annotationValue = "";
        else this.annotationValue = StringUtils.join(values, delimiter);
    }
    
    public String toString()
    {
        return annotationName + "=" + annotationValue;
    }
}
