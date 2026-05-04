#ifndef RANDOM_FOREST_RANDOM_FOREST_H
#define RANDOM_FOREST_RANDOM_FOREST_H

#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include "random_forest_decisionTree.h"

class randomForest
{
    public:
        std::vector<decisionTree*> forest;
        int totalTree;
        int maxDepth;
        randomForest(int n,int d) : totalTree(n),maxDepth(d){};
        // 隨機抽樣
        std::vector<sample> bootstrap(const std::vector<sample>& trainmodel)
        {
            std::vector<sample> subset;
            // 產生接近隨機的亂數種子
            std::random_device rd;
            // 偽隨機亂數生產器
            std::mt19937 gen(rd());
            // 映射到區間 [min,max]
            std::uniform_int_distribution<> distribute(0,trainmodel.size() - 1);
            // 抽樣次數與原始訓練集大小相同，這是隨機森林的標準作法
            for(int i = 0;i < trainmodel.size();i++)
            {
                int index = distribute(gen);
                // 因為是放回抽樣所以同一筆資料可能被抽中多次
                subset.push_back(trainmodel[index]);
            }
            return subset;
        }
        void train(const std::vector<sample>& trainmodel)
        {
            for(int i = 0;i < totalTree;i++)
            {
                std::cout << "Training tree #" << i + 1 << "......" << std::endl;
                // 取得子樣本
                std::vector<sample> subtree = bootstrap(trainmodel);
                std::vector<int> indices(subtree.size());
                for(int j = 0;j < subtree.size();j++)
                {
                    indices[j] = j;
                }
                // 建立並訓練樹
                decisionTree* tree = new decisionTree();
                tree -> root = tree -> build(indices,subtree,0,maxDepth);
                forest.push_back(tree);
            }
        }
        int predict(const sample& s)
        {
            std::map<int,int> vote;
            for(int i = 0;i < forest.size();i++)
            {
                int prediction = forest[i] -> predict(s,forest[i] -> root);
                vote[prediction] += 1; 
            }
            int maxVote = -1000;
            int finalQuality = -1000;
            for(auto const&[quality,total] : vote)
            {
                if(total > maxVote)
                {
                    maxVote = total;
                    finalQuality = quality;
                }
            }
            return finalQuality;
        }
        ~randomForest()
        {
            for(int i = 0;i < forest.size();i++)
            {
                delete forest[i] -> root;
                delete forest[i];
            }
        }
};

#endif