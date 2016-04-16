import javafx.util.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * 最大熵，训练算法采用GIS训练算法
 */
public class MaxEnt
{
    /**
     * 样本数据集
     */
    List<Instance> instanceList = new ArrayList<Instance>();
    /**
     * 特征列表，来自所有事件的统计结果
     */
    List<Feature> featureList = new ArrayList<Feature>();
    /**
     * 每个特征的出现次数
     */
    List<Integer> featureCountList = new ArrayList<Integer>();
    /**
     * 事件（类别）集
     */
     List<String> labels = new ArrayList<String>();
    /**
     * 每个特征函数的权重
     */
    double[] weight;
    /**
     * 一个事件最多一共有多少种特征
     */
    int C;

    public static void main(String[] args) throws IOException
    {
        String train_path = "data/word2vec_train3.txt";
        String test_path="data/word2vec_test3.txt";
        MaxEnt maxEnt = new MaxEnt();
        maxEnt.loadData(train_path);
        System.out.println();
        //这个是训练模型的迭代次数，刚开始设置了200，程序好久不动，这个数字越大训练时间越长== by szs
        maxEnt.train(200);
        //读取样本  by szc
        File file2=new File(test_path);
		if(!file2.exists()||file2.isDirectory()){
			throw new FileNotFoundException();
		}
		BufferedReader br=new BufferedReader(new FileReader(file2));
		String temp=null;
		temp=br.readLine();
		double num=0;
		double totalNum=0;
		while(temp!=null){
			totalNum++;
			List<String> fieldList = new ArrayList<String>();
			String[] segs = temp.split("\\s");
			String label;
			label=segs[1];
			for(int m=2;m<segs.length;m++){
				 if(segs[m].equals(""))
					 continue;
				System.out.println(segs[m]); 
				fieldList.add(segs[m]);
			}
			Pair<String, Double>[] result = maxEnt.predict(fieldList);  // 预测4 和 0的概率各是多少
			String temp_result;
			if(result[0].getValue()>result[1].getValue()){
				System.out.println("["+result[0].getKey()+"]");
				temp_result = result[0].getKey();
			}
			else{
				System.out.println("["+result[1].getKey()+"]");
				temp_result = result[1].getKey();
			}
			if(label.equals(temp_result))
				num++;
			
	        temp = br.readLine();
		}
		System.out.println("num:"+num+" totalNum:"+totalNum);
		System.out.println("the Precision is :"+ (num/totalNum));
        
    }

    /**
     * 加载数据，并且创建如下域
     * featureList：特征函数的list
     * featureCountList:与特征函数一一对应的，特征函数出现的次数
     * instanceList:样本数据list
     * labels:类别list
     *
     * @param path
     * @throws IOException
     */
    public void loadData(String path) throws IOException
    {
    	File file=new File(path);
		if(!file.exists()||file.isDirectory())
			throw new FileNotFoundException();
		BufferedReader br=new BufferedReader(new FileReader(file));
		String temp=null;
		temp=br.readLine();
	
		while(temp!=null && temp.length()!=0){
			String[] segs = temp.split("\\s");
			 String label = segs[1];
			 //System.out.println("label = " + label);
			 List<String> fieldList = new ArrayList<String>();
			 for(int i=2;i<segs.length;i++){
				 if(segs[i].equals(""))
					 continue;
				 //System.out.println(segs[i]);
				 //这一句是为了过滤所有的标点符号
				 fieldList.add(segs[i]);
	                Feature feature = new Feature(label, segs[i]);
	                int index = featureList.indexOf(feature);
	                if (index == -1)
	                {
	                    featureList.add(feature);
	                    featureCountList.add(1);
	                }
	                else
	                {
	                    featureCountList.set(index, featureCountList.get(index) + 1);
	                }
			 }
			 if (fieldList.size() > C) C = fieldList.size();
	            Instance instance = new Instance(label, fieldList);
	            instanceList.add(instance);
	            if (labels.indexOf(label) == -1 && (!label.equals("")) ) labels.add(label);
	            temp = br.readLine();
		}
		 System.out.println("----------------------------------");
		br.close();
    }

    /**
     * 训练模型
     * @param maxIt 最大迭代次数
     */
    public void train(int maxIt)
    {
        int size = featureList.size();
        weight = new double[size];               
        double[] empiricalE = new double[size];   
        double[] modelE = new double[size];      

        for (int i = 0; i < size; ++i)
        {
            empiricalE[i] = (double) featureCountList.get(i) / instanceList.size();
        }

        double[] lastWeight = new double[weight.length];  // weight of last iteration
        for (int i = 0; i < maxIt; ++i)
        {
        	computeModeE(modelE);
            for (int w = 0; w < weight.length; w++)
            {
                lastWeight[w] = weight[w];
                weight[w] += 1.0 / C * Math.log(empiricalE[w] / modelE[w]);
            }
            if (checkConvergence(lastWeight, weight)) break;
        }
    }

    /**
     * 预测类别
     * @param fieldList
     * @return
     */
    public Pair<String, Double>[] predict(List<String> fieldList)
    {
        double[] prob = calProb(fieldList);
        Pair<String, Double>[] pairResult = new Pair[prob.length];
        for (int i = 0; i < prob.length; ++i)
        {
            pairResult[i] = new Pair<String, Double>(labels.get(i), prob[i]);
        }

        return pairResult;
    }

    /**
     * 检查是否收敛
     * @param w1
     * @param w2
     * @return 是否收敛
     */
    public boolean checkConvergence(double[] w1, double[] w2)
    {
        for (int i = 0; i < w1.length; ++i)
        {
        	System.out.println("收敛"+Math.abs(w1[i] - w2[i]));
            if (Math.abs(w1[i] - w2[i]) >= 0.0001)    // 
                return false;
        }
        return true;
    }

    /**
     * 计算模型期望，即在当前的特征函数的权重下，计算特征函数的模型期望值。
     * @param modelE 储存空间，应当事先分配好内存（之所以不return一个modelE是为了避免重复分配内存）
     */
    public void computeModeE(double[] modelE)
    {
        Arrays.fill(modelE, 0.0f);
        for (int i = 0; i < instanceList.size(); ++i)
        {
            List<String> fieldList = instanceList.get(i).fieldList;
            //calculate the probability
            double[] pro = calProb(fieldList);
            for (int j = 0; j < fieldList.size(); j++)
            {
                for (int k = 0; k < labels.size(); k++)
                {
                    Feature feature = new Feature(labels.get(k), fieldList.get(j));
                    int index = featureList.indexOf(feature);
                    if (index != -1)
                        modelE[index] += pro[k] * (1.0 / instanceList.size());
                }
            }
        }
    }

    /**
     * 计算p(y|x),此时的x指的是instance里的field
     * @param fieldList 实例的特征列表
     * @return 该实例属于每个类别的概率
     */
    public double[] calProb(List<String> fieldList)
    {
        double[] p = new double[labels.size()];
        double sum = 0;  
        for (int i = 0; i < labels.size(); ++i)
        {
            double weightSum = 0;
            for (String field : fieldList)
            {
                Feature feature = new Feature(labels.get(i), field);
                int index = featureList.indexOf(feature);
                if (index != -1)
                    weightSum += weight[index];
            }
            p[i] = Math.exp(weightSum);
            sum += p[i];
        }
        for (int i = 0; i < p.length; ++i)
        {
            p[i] /= sum;
        }
        return p;
    }

    /**
     * 一个观测实例，包含事件和时间发生的环境
     */
    class Instance
    {
        /**
         * 事件（类别），如Outdoor
         */
        String label;
        /**
         * 事件发生的环境集合，如[Sunny, Happy]
         */
        List<String> fieldList = new ArrayList<String>();

        public Instance(String label, List<String> fieldList)
        {
            this.label = label;
            this.fieldList = fieldList;
        }
    }

    /**
     * 特征(二值函数)
     */
    class Feature
    {
        /**
         * 事件，如Outdoor
         */
        String label;
        /**
         * 事件发生的环境，如Sunny
         */
        String value;

        /**
         * 特征函数
         * @param label 类别
         * @param value 环境
         */
        public Feature(String label, String value)
        {
            this.label = label;
            this.value = value;
        }

        @Override
        public boolean equals(Object obj)
        {
            Feature feature = (Feature) obj;
            if (this.label.equals(feature.label) && this.value.equals(feature.value))
                return true;
            return false;
        }

        @Override
        public String toString()
        {
            return "[" + label + ", " + value + "]";
        }

    }
}
