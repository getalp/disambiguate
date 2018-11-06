package getalp.wsd.embeddings;

import java.util.Random;

public class VectorOperation
{
    public static double[] mul(double[] a, double b)
    {
        double[] ret = new double[a.length];
        for (int i = 0 ; i < a.length ; i++)
        {
            ret[i] = a[i] * b;
        }
        return ret;
    }
    
    public static double[] sub(double[] a, double[] b)
    {
        double[] ret = new double[a.length];
        for (int i = 0 ; i < a.length ; i++)
        {
            ret[i] = a[i] - b[i];
        }
        return ret;
    }

    public static double[] add(double[] a, double[] b)
    {
        double[] ret = new double[a.length];
        for (int i = 0 ; i < a.length ; i++)
        {
            ret[i] = a[i] + b[i];
        }
        return ret;
    }
    
    public static double[] sum(double[]... vectors)
    {
        double[] ret = new double[vectors[0].length];
        for (int i = 0 ; i < ret.length ; i++) 
        {
            ret[i] = 0;
            for (int j = 0 ; j < vectors.length ; j++) 
            {
                ret[i] += vectors[j][i];
            }
        }
        return ret;
    }
    
    public static double norm(double[] v) 
    {
        double ret = 0;
        for (int i = 0 ; i < v.length ; i++) 
        {
            ret += v[i] * v[i];
        }
        return Math.sqrt(ret);
    }
    
    public static double[] normalize(double[] v) 
    {
        double[] ret = new double[v.length];
        double norm = norm(v);
        if (norm == 0) throw new RuntimeException();
        for (int i = 0 ; i < v.length ; i++) 
        {
            ret[i] = v[i] / norm;
        }
        return ret;
    }

    public static double dot_product(double[] a, double[] b) 
    {
        double ret = 0;
        for (int i = 0 ; i < a.length ; i++) 
        {
            ret += a[i] * b[i];
        }
        return ret;
    }

    public static double[] term_to_term_product_squared(double[] a, double[] b)
    {
        double[] ret = new double[a.length];
        for (int i = 0 ; i < ret.length ; i++)
        {
            int sign = a[i] * b[i] < 0 ? -1 : 1;
            ret[i] = sign * Math.sqrt(Math.abs(a[i] * b[i]));
        }
        return ret;
    }

    public static double[] term_to_term_product(double[] a, double[] b)
    {
        double[] ret = new double[a.length];
        for (int i = 0 ; i < ret.length ; i++)
        {
            ret[i] = a[i] * b[i];
        }
        return ret;
    }

    public static double[] to_vector(String string)
    {
        if (string == null || string.isEmpty() || string.equals("[]")) return new double[0];
        String[] strValues = string.trim().replace("[", "").replace("]", "").replace(" ", "").split(",");
        double[] ret = new double[strValues.length];
        for (int i = 0 ; i < ret.length ; i++) ret[i] = Double.parseDouble(strValues[i]);
        return ret;
    }
    
    public static boolean same(double[] a, double[] b)
    {
        if (a.length != b.length) return false;
        for (int i = 0 ; i < a.length ; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
    
    public static double[] generateRandomUnitVector(int dimension)
    {
        double[] a = new double[dimension];
        while (norm(a) == 0)
        {
            for (int i = 0 ; i < dimension ; i++)
            {
                a[i] = new Random().nextGaussian();
            }
        }
        return normalize(a);
    }
    
    /**
     * Example : "2 0 42 1 3 -5"
     */
    public static String toString1(double[] vector)
    {
        StringBuilder res = new StringBuilder();
        for (int i = 0 ; i < vector.length ; i++)
        {
            res.append(vector[i]);
            res.append(" ");
        }
        return res.toString();
    }
}
