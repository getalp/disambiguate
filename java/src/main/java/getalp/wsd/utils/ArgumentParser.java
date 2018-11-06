package getalp.wsd.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

import org.apache.commons.cli.*;


public class ArgumentParser
{
	private CommandLineParser parser;
	
	private Options options;

	private Map<String, Object> defaultValues;
	
	private Map<String, Object> parsedArgs;

	public ArgumentParser()
	{
    	parser = new DefaultParser();
    	options = new Options();
    	defaultValues = new HashMap<>();
    	parsedArgs = new HashMap<>();
        addOptionalArgument("help");
	}

	public void addOptionalArgument(String name)
	{
		options.addOption(Option.builder().longOpt(name).build());
		defaultValues.put(name, false);
	}

	public void addArgument(String name)
	{
		options.addOption(Option.builder().longOpt(name).hasArg().required().build());
		defaultValues.put(name, null);
	}

	public void addArgument(String name, String defaultValue)
	{
    	options.addOption(Option.builder().longOpt(name).hasArg().build());
    	defaultValues.put(name, defaultValue);
	}

	public void addArgumentList(String name)
	{
		options.addOption(Option.builder().longOpt(name).hasArgs().required().build());
		defaultValues.put(name, null);
	}

	public void addArgumentList(String name, List<String> defaultValue)
	{
    	options.addOption(Option.builder().longOpt(name).hasArgs().build());
    	defaultValues.put(name, defaultValue);
	}

    public boolean parse(String[] args)
    {
        return parse(args, false);
    }

    public boolean parse(String[] args, boolean printArgs)
	{
		try
        {
            CommandLine cd = parser.parse(options, args);
            for (Option opt : cd.getOptions())
            {
                if (opt.hasArgs())
                {
                    parsedArgs.put(opt.getLongOpt(), opt.getValuesList());
                }
                else if (opt.hasArg())
                {
                    parsedArgs.put(opt.getLongOpt(), opt.getValue());
                }
                else
                {
                    parsedArgs.put(opt.getLongOpt(), true);
                }
            }
            if (hasArg("help"))
            {
                printArgs();
                return false;
            }
            else
            {
                if (printArgs)
                {
                    printArgs();
                }
                return true;
            }
        }
        catch (ParseException e)
        {
            printArgs();
            System.err.println("Error : " + e.getMessage());
            //new HelpFormatter().printHelp("<command>", options);
            return false;
        }
	}

    public boolean hasArg(String name)
    {
        return getArgValueGeneric(name);
    }

	public String getArgValue(String name)
	{
		return getArgValueGeneric(name);
	}

	public boolean getArgValueBoolean(String name)
	{
		return getArgValue(name).equals("true");
	}

	public int getArgValueInteger(String name)
	{
		return Integer.valueOf(getArgValue(name));
	}

	public List<String> getArgValueList(String name)
    {
    	return getArgValueGeneric(name);
    }

    @SuppressWarnings("unchecked")
    private <T> T getArgValueGeneric(String name)
    {
    	if (parsedArgs.containsKey(name))
    	{
    		return (T) parsedArgs.get(name);
    	}
    	else
    	{
    		return (T) defaultValues.get(name);
    	}
    }
    
    public void printArgs()
    {
        System.out.println("Arguments:");
    	for (Option opt : options.getOptions())
    	{
    		if (parsedArgs.containsKey(opt.getLongOpt()))
    		{
    			System.out.println("    --" + opt.getLongOpt() + " = " + parsedArgs.get(opt.getLongOpt()));
    		}
            else if (defaultValues.get(opt.getLongOpt()) != null)
            {
                System.out.println("    --" + opt.getLongOpt() + " (default value) = " + defaultValues.get(opt.getLongOpt()));
            }
    		else
            {
                System.out.println("    --" + opt.getLongOpt() + " (missing)");
            }
    	}
    }
}
