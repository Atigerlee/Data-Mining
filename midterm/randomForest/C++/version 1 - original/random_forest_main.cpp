#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <map>
#include "random_forest_dataTable.h"
#include "random_forest_decisionTree.h"
#include "random_forest_random_forest.h"

int main()
{
    dataTable dt;
    std::string name = "WineQT.txt";
    if(!dt.readfile(name))
    {
        return -1;
    }
    std::cout << "Successfully loaded " << dt.size() << " samples from " << name << std::endl;
    // 隨機打亂數據以確保 8 : 2 分割的公平性
    std::random_device rd;
    std::mt19937 gen(rd());
    // 洗牌
    std::shuffle(dt.samples.begin(),dt.samples.end(),gen);
    // 執行 80% 訓練與 20% 測試分割
    int size = (int)(dt.samples.size() * 0.8);
    // vector<T> 新變數(起點迭代器, 終點迭代器);
    std::vector<sample> train(dt.samples.begin(),dt.samples.begin() + size);
    std::vector<sample> test(dt.samples.begin() + size,dt.samples.end());
    std::cout << "Training set size : " << train.size() << std::endl;
    std::cout << "Testing set size : " << test.size() << std::endl;
    // 初始化並訓練
    int total = 743;
    int maxDepth = 20;
    randomForest rf(total,maxDepth);
    std::cout << std::endl;
    std::cout << "Starting training " << total << " trees......" << std::endl;
    rf.train(train);
    std::cout << "Training completed !" << std::endl;
    std::cout << std::endl;
    // 測試並計算準確率
    int correct = 0;
    // 實際為A預測也是A
    std::map<int,int> tpositive;
    // 預測為A實際不是A
    std::map<int,int> fpositive;
    // 實際為A預測不是A
    std::map<int,int> fnegative;
    std::set<int> allActual;
    std::map<int,int> allQuality;
    for(int i = 0;i < test.size();i++)
    {
        int final = rf.predict(test[i]);
        int actual = test[i].quality;
        allActual.insert(actual);
        allQuality[actual] += 1;
        if(final == actual)
        {
            correct += 1;
            tpositive[actual] += 1;
        }else
        {
            fpositive[final] += 1;
            fnegative[actual] += 1;
        }
    }
    // 輸出最終結果
    double accuracy = (double)correct / test.size() * 100.0;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Experimental Results : " << std::endl;
    std::cout << "Correct Predictions : " << correct << " / " << test.size() << std::endl;
    std::cout << "Model Accuracy : " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "Label" << std::setw(12) << "Precision" << std::setw(12) << "Recall" << std::setw(12) << "F1-Score" << "Support" << std::endl;
    double totalF1 = 0.0;
    double totalLabel = 0;
    double weightF1 = 0.0;
    for(int label : allActual)
    {
        double tp = tpositive[label];
        double fp = fpositive[label];
        double fn = fnegative[label];
        // 所有預測為 A 的樣本中真正是 A 的比例
        double precision = (tp + fp > 0) ? (tp / (tp + fp)) : 0;
        // 所有實際 A 為的樣本中被成功預測為 A 的比例
        double recall = (tp + fn > 0) ? (tp / (tp + fn)) : 0;
        double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;
        totalF1 += f1;
        weightF1 += f1 * allQuality[label];
        totalLabel += 1;
        std::cout << std::left << std::setw(10) << label << std::fixed << std::setprecision(4) << std::setw(12) << precision << std::setw(12) << recall << std::setw(12) << f1 << std::setprecision(0) << allQuality[label] << std::endl;
    }
    std::cout << "------------------------------------------------------------" << std::endl;
    if (totalLabel > 0) 
    {
        std::cout << "Macro-Average F1 Score : " << std::fixed << std::setprecision(4) << (totalF1 / totalLabel) << std::endl;
        std::cout << "Weighted-Average F1 Score : " << std::fixed << std::setprecision(4) << (weightF1 / (double)test.size()) << std::endl;
    }
    return 0;
}