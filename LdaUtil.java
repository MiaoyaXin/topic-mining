import java.util.*;

/**
 * @author hankcs
 */
public class LdaUtil
{
    /**
     * 根据主题号，获取主题词列表及其值列表，词个数limit指定。
     * @param tNum 主题号
     * @param vocabularyWord 所有主题的词汇表
     * @param vocabularyValue 所有主题的权重列表
     * @param limit 取出词的个数
     * @return
     */
    public static List<List<String>> translateV1(
            int tNum,
            List<List<String>> vocabularyWord,
            List<List<String>> vocabularyValue,
            int limit)
    {
        List<List<String>> topicWordsAndValue = new ArrayList<>();
        List<String> words = vocabularyWord.get(tNum);
        List<String> values = vocabularyValue.get(tNum);

        topicWordsAndValue.add(words.subList(0, limit));
        topicWordsAndValue.add(values.subList(0, limit));

        return topicWordsAndValue;
    }

    /**
     * To translate a LDA matrix to readable result
     * @param phi the LDA model
     * @param vocabulary
     * @param limit limit of max words in a topic
     * @return a map array
     */
    public static Map<String, Double>[] translate(double[][] phi, Vocabulary vocabulary, int limit)
    {
        limit = Math.min(limit, phi[0].length);
        @SuppressWarnings("unchecked")
        Map<String, Double>[] result = new Map[phi.length];
        for (int k = 0; k < phi.length; k++)
        {
            Map<Double, String> rankMap = new TreeMap<Double, String>(Collections.reverseOrder());
            for (int i = 0; i < phi[k].length; i++)
            {
                rankMap.put(phi[k][i], vocabulary.getWord(i));
            }
            Iterator<Map.Entry<Double, String>> iterator = rankMap.entrySet().iterator();
            result[k] = new LinkedHashMap<String, Double>();
            for (int i = 0; i < limit; ++i)
            {
                Map.Entry<Double, String> entry = iterator.next();
                result[k].put(entry.getValue(), entry.getKey());
            }
        }
        return result;
    }

    public static Map<String, Double> translate(double[] tp, double[][] phi, Vocabulary vocabulary, int limit)
    {
        Map<String, Double>[] topicMapArray = translate(phi, vocabulary, limit);
        double p = -1.0;
        int t = -1;
        for (int k = 0; k < tp.length; k++)
        {
            if (tp[k] > p)
            {
                p = tp[k];
                t = k;
            }
        }
        return topicMapArray[t];
    }

    /**
     * To print the result in a well formatted form
     * @param result
     */
    public static void explain(Map<String, Double>[] result)
    {
        int i = 0;
        for (Map<String, Double> topicMap : result)
        {
            System.out.printf("topic %d :\n", i++);
            explain(topicMap);
            System.out.println();
        }
    }

    public static void explain(Map<String, Double> topicMap)
    {
        for (Map.Entry<String, Double> entry : topicMap.entrySet())
        {
            System.out.println(entry);
        }
    }
}
