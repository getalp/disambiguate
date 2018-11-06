package getalp.wsd.utils;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.gson.stream.JsonReader;

public class Json 
{
	public static final int indentSize = 4;
	
	@SuppressWarnings("unchecked")
	public static Map<Object, Object> readMap(String filePath) throws IOException
	{
        JsonReader reader = new JsonReader(new FileReader(filePath));
        Map<Object, Object> map = (Map<Object, Object>) readObject(reader);
        reader.close();
        return map;
	}
	
	public static Object readObject(JsonReader reader) throws IOException
	{
        Object value = null;
        if (value == null) { try {
        	value = reader.nextBoolean();
        } catch (IllegalStateException e) {} }
        if (value == null) { try {
        	value = reader.nextInt();
        } catch (IllegalStateException|NumberFormatException e) {} }
        if (value == null) { try {
        	value = reader.nextLong();
        } catch (IllegalStateException|NumberFormatException e) {} }
        if (value == null) { try {
        	value = reader.nextDouble();
        } catch (IllegalStateException|NumberFormatException e) {} }
        if (value == null) { try {
        	value = reader.nextString();
        } catch (IllegalStateException e) {} }
        if (value == null) { try {
        	reader.beginArray();
        	List<Object> list = new ArrayList<>();
        	while (reader.hasNext())
        	{
        		list.add(readObject(reader));
        	}
        	reader.endArray();
        	value = list;
        } catch (IllegalStateException e) {} }
        if (value == null) { try {
        	reader.beginObject();
        	Map<Object, Object> map = new LinkedHashMap<>();
            while (reader.hasNext())
            {
                map.put(reader.nextName(), readObject(reader));
            }
            reader.endObject();
            value = map;
        } catch (IllegalStateException e) {} }
        if (value == null) { try {
        	reader.nextNull();
        } catch (IllegalStateException e) {} }
        return value;
	}
	
	public static void write(BufferedWriter out, Map<Object, Object> map) throws IOException
	{
		write(out, map, 0);
	}
	
	public static void write(BufferedWriter out, Map<Object, Object> map, int indentLevel) throws IOException
	{
		out.write(generateIndent(indentLevel) + "{\n");
		int mapKeySize = map.keySet().size();
		int i = 0;
		for (Object key : map.keySet())
		{
			indentLevel += 1;
			out.write(generateIndent(indentLevel) + "\"" + key.toString() + "\": ");
			out.write(toJsonString(map.get(key)));
			if (++i < mapKeySize) out.write(",");
			out.write("\n");
			indentLevel -= 1;
		}
		out.write(generateIndent(indentLevel) + "}\n");
	}
	
	@SuppressWarnings("unchecked")
	public static String toJsonString(Object obj) 
	{
		if (obj == null)
		{
			return toJsonStringNull();
		}
		else if (obj instanceof String)
		{
			return toJsonStringString((String) obj);
		}
		else if (obj instanceof Float || obj instanceof Double)
		{
			return toJsonStringDouble((Double) obj);
		}
		else if (obj instanceof List)
		{
			return toJsonStringList((List<Object>) obj);
		}
		else
		{
			return toJsonStringDefault(obj);
		}
	}

	public static String toJsonStringNull()
	{
		return "null";
	}
	
	public static String toJsonStringString(String str)
	{
		return "\"" + str + "\"";
	}
	
	public static String toJsonStringDouble(Double obj)
	{
		DecimalFormat df = new DecimalFormat("0", DecimalFormatSymbols.getInstance(Locale.ENGLISH));
		df.setMaximumFractionDigits(340);
		return df.format(obj);
	}
	
	public static String toJsonStringList(List<Object> list)
	{
		return list.stream().map(e -> toJsonString(e)).collect(Collectors.joining(", ", "[", "]"));
	}

	public static String toJsonStringDefault(Object obj)
	{
		return obj.toString();
	}
	
	public static String generateIndent(int indentLevel)
	{
		String ret = "";
		for (int i = 0 ; i < indentLevel * indentSize ; i++)
		{
			ret += " ";
		}
		return ret;
	}
}
