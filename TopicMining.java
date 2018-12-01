import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.lda.*;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.Config;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class topicMining{

    public static void main(String[] args) throws Exception{
//        // 训练语料处理入口
//        corpusPrepare.startProcess("!@#$");

//        // 训练主题模型
//        // 1. Load corpus from disk
//        Corpus corpus = Corpus.load("data/mini");
//        // 2. Create a LDA sampler
//        LdaGibbsSampler ldaGibbsSampler = new LdaGibbsSampler(corpus.getDocument(), corpus.getVocabularySize());
//        // 3. Train it
//        ldaGibbsSampler.gibbs(11);  // 信息技术、科学、体育、健康、军事、招聘、教育、文化、旅游、财经、金融等11个主题
//        // 4. The phi matrix is a LDA model, you can use LdaUtil to explain it.
//        double[][] phi = ldaGibbsSampler.getPhi();
//        // Map<String, Double>[] topicMap = LdaUtil.translate(phi, corpus.getVocabulary(), 10);
//        // LdaUtil.explain(topicMap);

        // 加载主题模型及其矩阵
        LDAModel ldaModel = new LDAModel(Config.hanLDAModelPath());
        double[][] phiV1 = ldaModel.getPhiMatrix();
        // 读入主题模型词汇表及其权重列表
        String vocabularyfile = "data/model/SogouCS_LDA.model.txt";
        List<List<String>> vocabularyWord = new ArrayList<>();
        List<List<String>> vocabularyValue = new ArrayList<>();
        buildTopicVocabulary(vocabularyfile, vocabularyWord, vocabularyValue);
//        System.out.println(vocabularyWord.size());
//        System.out.println(vocabularyValue.size());
//        for (int i=0; i<100; ++i){
//            System.out.println("topic" + String.valueOf(i));
//            for (int j=0; j<20; ++j)
//                System.out.println(vocabularyWord.get(i).get(j) + " " + vocabularyValue.get(i).get(j));
//            System.out.println();
//        }

        // 读取文章存入列表
        List<String> articles = new ArrayList<>();
        articles = readArticles();
        // 构建停用词集合
        List<String> stopwords = new ArrayList<>();
        stopwords = makeStopLst();
        // 话题挖掘与主题聚类
        List<List<List<String>>> doubleTopics = new ArrayList<>();
        doubleTopics = vote(
                articles,
                ldaModel, phiV1,
                stopwords,
                vocabularyWord,
                vocabularyValue);
        // 打印话题挖掘结果
        printTopicsIdsSubjects(doubleTopics);
        // 统计分析
        stasticAnalysis(doubleTopics, articles.size(), 1);// 第一大主题聚类
        stasticAnalysis(doubleTopics, articles.size(), 2);// 第二大主题聚类

        System.out.println("==================== The End ====================");

    }

    /**
     * 构建主题模型对应的二维词汇表及所有主题词的二维权重列表
     * @param filename
     * @param vocabularyWord
     * @param vocabularyValue
     * @throws IOException
     */
    public static void buildTopicVocabulary(
            String filename,
            List<List<String>> vocabularyWord,
            List<List<String>> vocabularyValue) throws IOException
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "utf-8"));
        String line;
        // int count = 0;
        while ((line = br.readLine()) != null){
            if (line.trim().length() < 2) continue;
            // System.out.println(removeBorderBlank(line));
            if (line.contains("topic") && line.contains(":")) {
                // count += 1;
                List<String> words = new ArrayList<>();
                List<String> values = new ArrayList<>();
                for (int i = 0; i < 20; ++i){
                    String inline = br.readLine();
                    // System.out.println(inline);
                    words.add(inline.split("=")[0]);
                    values.add(inline.split("=")[1]);
                }
                vocabularyWord.add(words);
                vocabularyValue.add(values);
            }
        }
        br.close();
        // System.out.println(count);
    }

    /**
     * 根据主题对话题挖掘结果进行统计分析
     * @param doubleTopics
     */
    public static void stasticAnalysis(List<List<List<String>>> doubleTopics, int count, int thT){
        // 大主题下的话题数量统计
        int thTopic = thT + 1;
        Map<List, Integer> subjectCount = new HashMap<>();
        Map<List, List<String>> subjectTopic = new HashMap<>();
        Map<List, List<String>> subjectArticleId = new HashMap<>();
        for (List<List<String>> ll: doubleTopics){
            if (subjectCount.containsKey(ll.get(thTopic))) {
                subjectCount.put(ll.get(thTopic), subjectCount.get(ll.get(thTopic)) + 1);
                // 话题聚类
                List<String> tmpLst1 = subjectTopic.get(ll.get(thTopic));
                tmpLst1.add(ll.get(4).get(0));
                subjectTopic.put(ll.get(thTopic), tmpLst1);
                // 文章列表ID聚类
                List<String> tmpLst2 = subjectArticleId.get(ll.get(thTopic));
                tmpLst2.add(ll.get(0).get(0));
                subjectArticleId.put(ll.get(thTopic), tmpLst2);
            }
            else {
                subjectCount.put(ll.get(thTopic), 1);
                List<String> tmpLst3 = new ArrayList<>();
                tmpLst3.add(ll.get(4).get(0));
                subjectTopic.put(ll.get(thTopic), tmpLst3);
                List<String> tmpLst4 = new ArrayList<>();
                tmpLst4.add(ll.get(0).get(0));
                subjectArticleId.put(ll.get(thTopic), tmpLst4);
            }
        }
        System.out.println("-------------------- 聚类结果" + String.valueOf(thT) + " --------------------\n");
        System.out.print("处理文章：" + count + "篇，");
        System.out.println("挖掘话题：" + doubleTopics.size() + "个，聚类效果如下：");
        System.out.println();
        for (List ll2: subjectCount.keySet()) {
            System.out.println(ll2 + "\t\t" + String.valueOf(subjectCount.get(ll2)));
            for (String ss: subjectArticleId.get(ll2))
                System.out.print(ss + " ");
            System.out.println();
            for (String ss: subjectTopic.get(ll2))
                System.out.println(ss);
            System.out.println();
        }
    }

    /**
     * 打印话题，属于该话题的文章id列表，主题词列表
     * @param doubleTopics
     */
    public static void printTopicsIdsSubjects(List<List<List<String>>> doubleTopics)
    {
        System.out.println("==================== 详情列表 ====================");
        for (List<List<String>> tiw: doubleTopics){
            System.out.println("索引：\t" + tiw.get(0).get(0)); // 文章编号
            // 文章导语
            System.out.print("导语：\t");
            for (String w: tiw.get(1))
                System.out.print(w);
            System.out.println("...");
            System.out.println("大主题：\t" + tiw.get(2)); // 大主题，根据模型预测得到
            System.out.println("\t\t" + tiw.get(3)); // 大主题中词的权重-->第二大主题
            System.out.println("话题：\t" + tiw.get(4).get(0)); // 话题
            System.out.println("小主题：\t" + tiw.get(5)); // 小主题，待分析文章的
            System.out.println("\t\t" + tiw.get(6)); // 小主题词的权重
            System.out.println("\t\t" + tiw.get(7));  // 话题属于该第一大主题的权重
            System.out.println();
        }
    }

    /**
     * 读取测试文章列表
     * @return
     * @throws IOException
     */
    public static List<String> readArticles() throws IOException {
        List<String> articles = new ArrayList<>();
        File folder = new File("data/articlebak");
        // File folder = new File("data/jinrong");
        for (File file: folder.listFiles()){

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
            String line;
            String lines = "";
            while ((line = br.readLine()) != null){
                if (line.trim().length() < 2) continue;
                lines +=  line + "\n";
            }
            br.close();
            articles.add(lines);
            // System.out.println(file);
        }
        // System.out.println(articles.size());
        return articles;
    }

    /**
     * 使用投票机制优化话题挖掘效果
     * @param articles 待分析文章
     * @param ldaModel 主题模型
     * @param phiV1 主题模型矩阵
     * @param stopwords 停用词
     * @return
     * @throws Exception
     */
    public static List<List<List<String>>> vote(List<String> articles,
                                                LDAModel ldaModel,
                                                double[][] phiV1,
                                                List<String> stopwords,
                                                List<List<String>> vocabularyWord,
                                                List<List<String>> vocabularyValue) throws Exception{

        List<List<String>> docs = new ArrayList<>();
        docs = txtWash(articles, stopwords);  // 分词，清洗，去停用词

        List<List<List<String>>> doubleTopics1 = new ArrayList<>();
        doubleTopics1 = entranceV1(
                articles, docs,
                ldaModel, phiV1,
                vocabularyWord,
                vocabularyValue);  // 第一次话题挖掘
        List<List<List<String>>> doubleTopics2 = new ArrayList<>();
        doubleTopics2 = entranceV1(
                articles, docs,
                ldaModel, phiV1,
                vocabularyWord,
                vocabularyValue);  // 第二次话题挖掘
//        List<List<List<String>>> doubleTopics3 = new ArrayList<>();
//        doubleTopics3 = entrance(articles, docs, corpus, phi);  // 第三次话题挖掘

        // 投票
        List<List<List<String>>> doubleTopicsFinal = intersection(doubleTopics1, doubleTopics2);
        // List<List<List<String>>> doubleTopicsMerge2 = intersection(doubleTopics1, doubleTopics3);
        // List<List<List<String>>> doubleTopicsFinal = intersection(doubleTopics1, doubleTopics2, doubleTopics3);

        return doubleTopicsFinal;
    }

    /**
     * 求三次投票一致的直接返回，
     * @param doubleTopics1
     * @param doubleTopics2
     * @param doubleTopics3
     * @return
     */
    public static List<List<List<String>>> intersection(
            List<List<List<String>>> doubleTopics1,
            List<List<List<String>>> doubleTopics2,
            List<List<List<String>>> doubleTopics3){

        List<List<List<String>>> reLst = new ArrayList<>();
        for (int idx = 0; idx < doubleTopics1.size(); ++idx) {
            if (doubleTopics1.get(idx).get(2).equals(doubleTopics2.get(idx).get(2))
                    && doubleTopics1.get(idx).get(4).equals(doubleTopics2.get(idx).get(4))
            && doubleTopics1.get(idx).get(2).equals(doubleTopics3.get(idx).get(2))
                    && doubleTopics1.get(idx).get(4).equals(doubleTopics3.get(idx).get(4)))  // 三次达成一致的话题和大主题

                reLst.add(doubleTopics1.get(idx));
        }

        return reLst;
    }

    /**
     * 1.两次投票机制；2.同一主题同一篇文章的不同话题'|'链接合并
     * @param doubleTopics1
     * @param doubleTopics2
     * @return
     */
    public static List<List<List<String>>> intersection(
            List<List<List<String>>> doubleTopics1,
            List<List<List<String>>> doubleTopics2 ){

        List<List<List<String>>> reLst = new ArrayList<>();
        for (int idx = 0; idx < doubleTopics1.size(); ++idx) {
            // 两次大主题预测和话题挖掘结果均一致
            if (doubleTopics1.get(idx).get(2).equals(doubleTopics2.get(idx).get(2))
                    && doubleTopics1.get(idx).get(4).equals(doubleTopics2.get(idx).get(4))) {
                reLst.add(doubleTopics1.get(idx));
            }
            // 两次大主题预测结果一致，话题挖掘结果不一致
            else if (doubleTopics1.get(idx).get(2).equals(doubleTopics2.get(idx).get(2))){
                List<List<String>> tmp2DLst = new ArrayList<>();
                tmp2DLst.add(doubleTopics1.get(idx).get(0));  // 文章列表索引
                tmp2DLst.add(doubleTopics1.get(idx).get(1));  // 文章导语
                tmp2DLst.add(doubleTopics1.get(idx).get(2));  // 大主题词列表
                tmp2DLst.add(doubleTopics1.get(idx).get(3));  // 大主题词权重列表-->第二大主题词列表
                if (doubleTopics1.get(idx).get(4).get(0).contains(doubleTopics2.get(idx).get(4).get(0))){
                    tmp2DLst.add(doubleTopics1.get(idx).get(4));  // 同一大主题下两个不同的话题列表，前面列表包含后面的
                    tmp2DLst.add(doubleTopics1.get(idx).get(5));  // 同一大主题下两个不同的小主题词列表，前面列表包含后面的
                    tmp2DLst.add(doubleTopics1.get(idx).get(6));  // 同一大主题下两个不同的小主题词权重列表，前面列表包含后面的
                }else {
                    List<String> tmpLst = new ArrayList<>();
                    tmpLst.add(doubleTopics1.get(idx).get(4).get(0) + " | " + doubleTopics2.get(idx).get(4).get(0));
                    tmp2DLst.add(tmpLst);  // 同一大主题下两个不同的话题列表合并

                    List<String> tmpLst2 = doubleTopics1.get(idx).get(5);
                    tmpLst2.addAll(doubleTopics2.get(idx).get(5));
                    tmp2DLst.add(tmpLst2);  // 同一大主题下两个不同的小主题词列表合并

                    List<String> tmpLst3 = doubleTopics1.get(idx).get(6);
                    tmpLst3.addAll(doubleTopics2.get(idx).get(6));
                    tmp2DLst.add(tmpLst3);  // 同一大主题下两个不同的小主题词权重列表合并
                }
                tmp2DLst.add(doubleTopics1.get(idx).get(7));  // 最大主题概率值入列表
                reLst.add(tmp2DLst);
            }
            // 两次大主题预测和话题挖掘结果均不一致
            else{
                if (Double.parseDouble(doubleTopics1.get(idx).get(7).get(0))
                        < Double.parseDouble(doubleTopics2.get(idx).get(7).get(0))){
                    reLst.add(doubleTopics2.get(idx));
                }
                else
                    reLst.add(doubleTopics1.get(idx));
            }
        }
        return reLst;
    }

    /**
     * 构建停用词集合
     * @return
     * @throws IOException
     */
    public static List<String> makeStopLst() throws IOException{
        List<String> tmpLst = new ArrayList<>();
        File folder = new File("data/stopwords");
        for (File file: folder.listFiles()){
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
            String line;
            while ((line = br.readLine()) != null){
                if (line.trim().length() < 1) continue;
                tmpLst.add(line);
            }
            br.close();
        }
        return tmpLst;
    }

    /**
     * 感知器分词，加载人民日报训练的分词模型，去停用词，去单字词
     * @param articles
     * @return
     * @throws IOException
     */
    public static List<List<String>> txtWash(List<String> articles, List<String> stopLst) throws IOException
    {
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer(
                "data/model/perceptron/pku199801/cws.bin",
                HanLP.Config.PerceptronPOSModelPath,
                HanLP.Config.PerceptronNERModelPath);  // 感知器分词
        List<List<String>> docs = new ArrayList<>();
        String emailRegEx ="^[a-zA-Z0-9_.-]+@([a-zA-Z0-9-]+\\.)+[a-zA-Z0-9]{2,4}$";
        String dateRegex= "[\\d]+[年月日]$";
        String digitRegex = "^[\\-\\\\\\d]+$";
        String urlRegex =
                "(http://|ftp://|https://|www){0,1}[^\u4e00-\u9fa5\\s]*?\\.(com|net|cn|me|tw|fr)[^\u4e00-\u9fa5\\s]*";
        Pattern pUrlEx = Pattern.compile(urlRegex);
        Pattern pEmail = Pattern.compile(emailRegEx);
        Pattern pDate = Pattern.compile(dateRegex);
        Pattern pDigit = Pattern.compile(digitRegex);
        for (String doc: articles){
            List<String> tmpDoc = new ArrayList<>();
            for (String line: doc.split("\n")){
                for (Word w: analyzer.analyze(line.trim()).toSimpleWordList()){
                    String Wd = removeBorderBlank(w.toString().substring(0, w.toString().indexOf("/")).trim());
                    if (Wd.length() < 2 || stopLst.contains(Wd) || pDate.matcher(Wd).find()
                            ||  pEmail.matcher(Wd).find() || pDigit.matcher(Wd).find() || pUrlEx.matcher(Wd).find())
                        continue;
                    tmpDoc.add(Wd);
                }
            }
            docs.add(tmpDoc);
        }
        return docs;
    }

    /**
     * 话题挖掘入口1
     * @param articles
     * @param texts
     * @param ldaModel
     * @param phiV1
     * @return
     * @throws Exception
     */
    public static List<List<List<String>>> entranceV1(List<String> articles,
                                                      List<List<String>> texts,
                                                      LDAModel ldaModel,
                                                      double[][] phiV1,
                                                      List<List<String>> vocabularyWord,
                                                      List<List<String>> vocabularyValue) throws Exception
    {
        // 构造候选话题集
        // 1.Load corpus from disk
        Corpus txts = loadDocs(texts);
        if (txts == null) return null;
        // 2.Create a LDA sampler
        LdaGibbsSampler ldaGibbsSampler1 = new LdaGibbsSampler(txts.getDocument(), txts.getVocabularySize());
        // 3.Train it
        int kTopics;
        if (texts.size() > 50)
            kTopics = 20;
        else if (texts.size() > 20)
            kTopics = 16;
        else if (texts.size() > 10)
            kTopics = 10;
        else
            kTopics = texts.size();
        ldaGibbsSampler1.gibbs(kTopics);
        // 4.The phi matrix is a LDA model, you can use LdaUtil to explain it.
        double[][] phi1 = ldaGibbsSampler1.getPhi();
        Map<String, Double>[] topicMap1 = LdaUtil.translate(phi1, txts.getVocabulary(), 10);
        List<List<List<String>>> topicsList = explain(topicMap1);
        // displayTopicsAndValues(topicsList);
        List<List<List<String>>> topicsIdsSubjects = generateTopics(articles, topicsList);

        // 预测文章主题
        List<List<List<String>>> doubleTopics = new ArrayList<>();
        for (int idx = 0; idx < texts.size(); ++idx){
            int[] document = Corpus.loadText(texts.get(idx), ldaModel.getVocabulary());
            double[] tp = LdaGibbsSampler.inference(phiV1, document);
            // 计算大主题
            int tNum = HanLDA.getMaxProbTopic(tp); // 获取最大概率主题号
            double weight = tp[tNum];  // 获取最大概率值
            tp[tNum] = Double.MIN_VALUE;
            int tNum2 = HanLDA.getMaxProbTopic(tp); // 获取第二大概率主题号
            List<List<String>> bigTopicLst = LdaUtil.translateV1(tNum, vocabularyWord, vocabularyValue,10);
            List<List<String>> secondBigTopicLst = LdaUtil.translateV1(tNum2, vocabularyWord, vocabularyValue, 10);
            List<List<String>> tmpTopics = new ArrayList<>();

            tmpTopics.add(topicsIdsSubjects.get(idx).get(1)); // 文章编号
            tmpTopics.add(texts.get(idx).subList(0, 16)); // 文章导语

            tmpTopics.add(bigTopicLst.get(0)); // 第一大主题，根据模型预测得到
            // tmpTopics.add(bigTopicLst.get(1)); // 大主题中词的权重
            tmpTopics.add(secondBigTopicLst.get(0)); // 第二大主题，根据模型预测得到

            tmpTopics.add(topicsIdsSubjects.get(idx).get(0)); // 话题列表

            tmpTopics.add(topicsIdsSubjects.get(idx).get(2)); // 小主题，待分析文章的
            tmpTopics.add(topicsIdsSubjects.get(idx).get(3)); // 小主题词的权重

            List<String> tmpWeight = new ArrayList<>();  // 第一大主题的概率值
            tmpWeight.add(String.valueOf(weight));  // 第一大主题概率值入列表
            tmpTopics.add(tmpWeight);  // 用于投票阶段大主题不一致时进行主题选择的判断依据

            doubleTopics.add(tmpTopics); // 双主题结果入列表
        }

        return doubleTopics;
    }

    /**
     * 清洗后的文档转换为Corpus类型对象
     * @param texts
     * @return
     * @throws IOException
     */
    public static Corpus loadDocs(List<List<String>> texts) throws IOException
    {
        Corpus corpus = new Corpus();

        for (List<String> wordList: texts)
            corpus.addDocument(wordList);

        if (corpus.getVocabularySize() == 0) return null;

        return corpus;
    }

    /**
     * 根据主题聚类效果，对文章列表进行话题挖掘
     * @param articles  文章列表
     * @param topicsList 已聚类主题列表
     * @return [[话题列表][文章ID列表][主题词列表]]
     */
    public static List<List<List<String>>> generateTopics(List<String> articles, List<List<List<String>>> topicsList)
    {
        List<List<List<String>>> topicsIdsSubjects = new ArrayList<>();
        String punctuationStr =
                ",|\\\\/|;|\\\\'|`|\\[|\\\\]|<|>|\\?|:|：|\"|\\{|\\\\}|\\\\~|!|@|#|\\$|%|\\^|&|\\(|\\)|-|=|\\\\_|\\+|\\\\，|，|。|；|‘|’|【|】|！| |…|（|）";
        String zeroStr = "^[\n\r\t]+$";
        Pattern pZero = Pattern.compile(zeroStr);
        String skipStr1 = "来源|编辑|作者|供稿|我|你|他|她|它|；";
        Pattern pSkip1 = Pattern.compile(skipStr1);
        String skipStr2 = "根据|因为";
        Pattern pSkip2 = Pattern.compile(skipStr2);

        for (int idx = 0; idx < articles.size(); ++idx){

            boolean flagTopic = false;
            List<String> topicLst = new ArrayList<>();
            List<String> idLst = new ArrayList<>();
            List<List<String>> topicIdSubject = new ArrayList<>();
            for (String line: articles.get(idx).split("\n")){
                if (pZero.matcher(line).find() || line.length() < 7 || pSkip1.matcher(line).find())
                    continue;
                if (line.length() < 32) {
                    for (List<List<String>> tLst : topicsList) {
                        for (String t_ : tLst.get(0)) {
                            if (line.contains(t_)) {
                                topicLst.add(removeBorderPunc(removeBorderBlank(line)));  // 此处为标题的概率很大，作为话题
                                idLst.add(String.valueOf(idx));  // 文章ID列表
                                serialize2topicIdSubject(
                                        topicLst,
                                        idLst,
                                        tLst.get(0),
                                        tLst.get(1),
                                        topicIdSubject);
                                flagTopic = true;
                                break;
                            }
                        }
                        if (flagTopic)
                            break;
                    }
                }
                else{
                    for (String subLine: line.split(punctuationStr)) {
                        if (subLine.length() < 12) continue;
                        for (List<List<String>> tLst : topicsList) {
                            for (String t_ : tLst.get(0)) {
                                if (subLine.contains(t_) && !pSkip2.matcher(subLine).find()) {
                                    topicLst.add(removeBorderPunc(removeBorderBlank(subLine)));  // 作为话题的短句
                                    idLst.add(String.valueOf(idx));  // 文章ID列表
                                    serialize2topicIdSubject(
                                            topicLst,
                                            idLst,
                                            tLst.get(0),
                                            tLst.get(1),
                                            topicIdSubject);
                                    flagTopic = true;
                                    break;
                                }
                            }
                            if (flagTopic)
                                break;
                        }
                        if (flagTopic)
                            break;
                    }
                }
                if (flagTopic) {
                    break;
                }
            }
            topicsIdsSubjects.add(topicIdSubject);
        }
        return topicsIdsSubjects;
    }

    /**
     *
     * @param topicLst  话题列表
     * @param idLst  文章ID列表
     * @param wdLst  主题词列表
     * @param wtLst  权重列表
     * @param topicIdSubject  目标列表，结构：[[话题][文章ID][主题词][权重]]
     */
    public static void serialize2topicIdSubject(List<String> topicLst,
                                                List<String> idLst,
                                                List<String> wdLst,
                                                List<String> wtLst,
                                                List<List<String>> topicIdSubject)
    {
        topicIdSubject.add(topicLst);  // 话题列表入列表
        topicIdSubject.add(idLst);  // 文章ID列表入列表
        topicIdSubject.add(wdLst);  // 主题列表入列表
        topicIdSubject.add(wtLst);  // 主题词权重列表入列表
    }

    /**
     * 提取所有主题的主题词及相关值，三维列表存储
     * @param result
     * @return
     */
    public static List<List<List<String>>> explain(Map<String, Double>[] result)
    {
        List<List<List<String>>> topicsList = new ArrayList<>();
        for (Map<String, Double> topicMap : result)
        {
            topicsList.add(explain(topicMap));
        }
        return topicsList;
    }

    /**
     * 提取某个主题下的主题词及其相关值，二维列表存储
     * @param topicMap
     * @return
     */
    public static List<List<String>> explain(Map<String, Double> topicMap)
    {
        List<List<String>> topicWordsAndValue = new ArrayList<>();
        List<String> words = new ArrayList<>();
        List<String> values = new ArrayList<>();

        for (Map.Entry<String, Double> entry : topicMap.entrySet())
        {
            words.add(entry.getKey());
            values.add(String.valueOf(entry.getValue()));
        }

        topicWordsAndValue.add(words);
        topicWordsAndValue.add(values);

        return topicWordsAndValue;
    }

    /**
     * 打印主题列表
     * @param arg_topics
     */
    public static void displayTopicsAndValues(List<List<List<String>>> arg_topics)
    {
        int i = 1;
        for (List<List<String>> topicWordsAndValues: arg_topics){
            System.out.printf("topic %d:\n", i++);
            System.out.println(topicWordsAndValues.get(0));
            System.out.println(topicWordsAndValues.get(1));
            System.out.println();
        }
    }

    /**
     * 打印主题列表
     * @param arg_topics
     */
    public static void displayTopicsAndValues2Dim(List<List<String>> arg_topics)
    {
        for (List<String> topicWordsAndValues: arg_topics)
            System.out.println(topicWordsAndValues);

        System.out.println();
    }

    /**
     * 去除字符串中所包含的空格（包括:空格(全角，半角)、制表符、换页符等）
     * @params
     * @return
     */
    public static String removeAllBlank(String s){
        String result = "";
        if(null != s && !"".equals(s)){
            result = s.replaceAll("[　 \\s*]*", "");
        }
        return result;
    }

    /**
     * 去除字符串中头部和尾部所包含的空格（包括:空格(全角，半角)、制表符、换页符等）
     * @params
     * @return
     */
    public static String removeBorderBlank(String s){
        String result = "";
        if(null != s && !"".equals(s)){
            result = s.replaceAll("^[　 \\s*]*", "").replaceAll("[　 \\s*]*$", "");
        }
        return result;
    }

    /**
     * 移除两端的标点等特殊符号
     * @param s
     * @return
     */
    public static String removeBorderPunc(String s){
        return s.replaceAll("[,，;'`?:：\"{}~!@#$%^&=_+.。；‘’【】！ …（）、]+$", "")
                .replaceAll("^[,，;'`?:：\"{}~!@#$%^&=_+.。；‘’【】！ …（）、]+", "");
    }

    /**
     * 定义一个专门用于为主题模型准备训练语料的类，对文本进行清洗、分词，输出至文本文件
     */
    public static class corpusPrepare{

        public static void startProcess(String sn) throws IOException{
            if (!sn.equalsIgnoreCase("!@#$")) {
                System.out.println("你没有权限训练语料！程序已返回。");
                return;
            }
            String sourcePathStr = "data/science";
            String destFileName = "data" + File.separator + "mini" + File.separator + "SCI_";
            int fileId = 10;
            // 读取文章存入列表
            List<String> articles = new ArrayList<>();
            articles = readArticles(sourcePathStr);
            processing(articles, destFileName, fileId);
            System.out.println("语料处理完毕！");
        }

        /**
         * 读取测试文章列表
         * @return
         * @throws IOException
         */
        public static List<String> readArticles(String sourceFileStr) throws IOException {
            List<String> articles = new ArrayList<>();
            File folder = new File(sourceFileStr);
            for (File file: folder.listFiles()){
                BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
                String line;
                String lines = "";
                while ((line = br.readLine()) != null){
                    if (line.trim().length() < 2) continue;
                    lines +=  line.trim() + "\n";
                }
                br.close();
                articles.add(lines);
                // System.out.println(file);
            }
            // System.out.println(articles.size());
            return articles;
        }

        /**
         *
         * @param articles 待分析文章
         * @throws Exception
         */
        public static void processing(List<String> articles, String destFileName, int fileId) throws IOException{
            // 构建停用词集合
            String pathStr ="data/stopwords";
            List<String> stopwords = readFile2List(pathStr);
            List<List<String>> docs = new ArrayList<>();
            docs = txtWash(articles, stopwords);  // 分词，清洗，去停用词
            saveJr(docs, destFileName, fileId);  // 语料存储
        }

        /**
         * 构建停用词集合
         * @return
         * @throws IOException
         */
        public static List<String> readFile2List(String fileNameAndPath) throws IOException{
            File folder = new File(fileNameAndPath);
            List<String> reLst = new ArrayList<>();
            for (File f_: folder.listFiles()){
                BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f_), "utf-8"));
                String line;
                while ((line = br.readLine())!=null){
                    if (line.trim().length() < 1) continue;
                    reLst.add(line.trim());
                }
                br.close();
            }
            return reLst;
        }

        /**
         * 感知器分词，加载人民日报训练的分词模型，去停用词，去单字词
         * @param articles
         * @return
         * @throws IOException
         */
        public static List<List<String>> txtWash(List<String> articles, List<String> stopLst) throws IOException
        {
            PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer(
                    "data/model/perceptron/pku199801/cws.bin",
                    HanLP.Config.PerceptronPOSModelPath,
                    HanLP.Config.PerceptronNERModelPath);  // 感知器分词
            List<List<String>> docs = new ArrayList<>();
            String emailRegEx ="^[a-zA-Z0-9_.-]+@([a-zA-Z0-9-]+\\.)+[a-zA-Z0-9]{2,4}$";
            String dateRegex= "[\\d]+[年|月|日]$";
            String digitRegex = "^[\\-\\\\\\d]+$";
            String urlRegex =
                    "(http://|ftp://|https://|www){0,1}[^\u4e00-\u9fa5\\s]*?\\.(com|net|cn|me|tw|fr)[^\u4e00-\u9fa5\\s]*";
            Pattern pUrlEx = Pattern.compile(urlRegex);
            Pattern pEmail = Pattern.compile(emailRegEx);
            Pattern pDate = Pattern.compile(dateRegex);
            Pattern pDigit = Pattern.compile(digitRegex);
            for (String doc: articles){
                List<String> tmpDoc = new ArrayList<>();
                for (String line: doc.split("\n")){
                    for (Word w: analyzer.analyze(line.trim()).toSimpleWordList()){
                        String Wd = removeBorderBlank(w.toString().substring(0, w.toString().indexOf("/")).trim());
                        if (Wd.length() < 2 || stopLst.contains(Wd) || pDate.matcher(Wd).find()
                                ||  pEmail.matcher(Wd).find() || pDigit.matcher(Wd).find() || pUrlEx.matcher(Wd).find())
                            continue;
                        tmpDoc.add(Wd);
                        System.out.print(Wd + " ");
                    }
                    tmpDoc.add("\n");  // 训练语料换行用
                    System.out.println();
                }
                System.out.println("========================================");
                docs.add(tmpDoc);
            }
            System.out.println(articles.size());
            System.out.println(docs.size());
            return docs;
        }

        /**
         * 分词，清洗的语料，存储到主题模型训练语料文件夹
         * @param docs
         */
        public static void saveJr(List<List<String>> docs, String destFileName, int fileId) throws IOException{

            for (List<String> doc: docs){
                // 1：利用File类找到要操作的对象
                File file = new File(destFileName + String.valueOf(fileId) + ".txt");
                // 2：准备输出流
                Writer out = new FileWriter(file);
                for (String ss: doc)
                    if (ss.equalsIgnoreCase("\n"))
                        out.write(ss);
                    else if (ss.trim().length() > 1)
                        out.write(ss.trim() + " ");
                    else
                        continue;
                out.close();
                fileId += 10;
            }
        }
    }
}
