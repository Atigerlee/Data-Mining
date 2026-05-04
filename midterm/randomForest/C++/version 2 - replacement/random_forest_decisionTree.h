#ifndef RANDOM_FOREST_DECISIONTREE_H
#define RANDOM_FOREST_DECISIONTREE_H

#include <vector>
#include <cmath>
#include <map>
#include <set>
#include "random_forest_dataTable.h"

struct node
{
    // 用 leaf 存預測之資料
    bool isLeaf = false;
    int leafQuality = -1;
    // 分割之 root 及其值
    int characterIndex = -1;
    double bar = 0.0;
    node* left = nullptr;
    node* right = nullptr;
    ~node();
};

node :: ~node()
{
    delete left;
    delete right;
}

class decisionTree
{
    public:
        node* root = nullptr;
        double calculateGini(const std::vector<sample>& data)
        {
            if(data.size() == 0) return 0.0;
            std::map<int,int> count;
            for(int i = 0;i < data.size();i++)
            {
                count[data[i].quality] += 1;
            }
            int total = data.size();
            double sum = 0.0;
            for(auto const&[quality,counts] : count)
            {
                double p = counts / (double)total;
                sum += pow(p,2);
            }
            return 1.0 - sum;           // Gini 不純度 = 1 - 所有出現機率的平方總和
        }
        void findBestSplitPoint(const std::vector<sample>& data,int& bestLabel,double& bestBar)
        {
            double minGini = 1.0;
            bestLabel = -1;
            bestBar = 0.0;
            // 經歷 11 種特徵
            for(int c = 0;c < 11;c++)
            {
                std::set<double> value;
                for(int i = 0;i < data.size();i++)
                {
                    value.insert(data[i].character[c]);
                }
                // 測試每一種門檻
                for(double now : value)
                {
                    std::vector<sample> left,right;
                    for(int i = 0;i < data.size();i++)
                    {
                        if(data[i].character[c] <= now)
                        {
                            left.push_back(data[i]);
                        }else
                        {
                            right.push_back(data[i]);
                        }
                    }
                    // 不可以有一側是空的
                    if(left.size() == 0 || right.size() == 0) continue;
                    double gini = (left.size() * calculateGini(left) + right.size() * calculateGini(right)) / (double)data.size();
                    if(gini < minGini)
                    {
                        minGini = gini;
                        bestLabel = c;
                        bestBar = now;
                    }
                }
            }
        }
        // 遞迴種樹
        node* build(const std::vector<sample>& data,int depth,int maxDepth)
        {
            if(data.size() == 0)    return nullptr;
            node* newones = new node();
            // 達到最大深度 、 全部都同一列 、 數據量太少
            if(depth >= maxDepth || calculateGini(data) == 0 || data.size() < 9)
            {
                newones -> isLeaf = true;
                // 投票決定葉子之值
                std::map<int,int> count;
                int maxVote = -100;
                for(int i = 0;i < data.size();i++)
                {
                    count[data[i].quality] += 1;
                    if(count[data[i].quality] > maxVote)
                    {
                        maxVote = count[data[i].quality];
                        newones -> leafQuality = data[i].quality;
                    }
                }
                return newones;
            }
            int bestLabel;
            double bestBar;
            findBestSplitPoint(data,bestLabel,bestBar);
            // 找不到分割點
            if(bestLabel == -1)
            {
                newones -> isLeaf = true;
                // 投票決定葉子之值
                std::map<int,int> count;
                int maxVote = -100;
                for(int i = 0;i < data.size();i++)
                {
                    count[data[i].quality] += 1;
                    if(count[data[i].quality] > maxVote)
                    {
                        maxVote = count[data[i].quality];
                        newones -> leafQuality = data[i].quality;
                    }
                }
                return newones;
            }
            newones -> characterIndex = bestLabel;
            newones -> bar = bestBar;
            std::vector<sample> left,right;
            for(int i = 0;i < data.size();i++)
            {
                if(data[i].character[bestLabel] <= bestBar)
                {
                    left.push_back(data[i]);
                }else
                {
                    right.push_back(data[i]);
                }
            }
            newones -> left = build(left,depth + 1,maxDepth);
            newones -> right = build(right,depth + 1,maxDepth);
            return newones;
        }
        // 預測單一樣本
        int predict(const sample& s,node* now)
        {
            if(now -> isLeaf) return now -> leafQuality;
            if(s.character[now -> characterIndex] <= now -> bar)
            {
                return predict(s,now -> left);
            }else
            {
                return predict(s,now -> right);
            }
        }
};

#endif