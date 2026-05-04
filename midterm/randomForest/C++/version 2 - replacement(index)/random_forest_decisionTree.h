#ifndef RANDOM_FOREST_DECISIONTREE_H
#define RANDOM_FOREST_DECISIONTREE_H

#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
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
        double calculateGini(const std::vector<int>& indices,const std::vector<sample>& data)
        {
            if(data.size() == 0) return 0.0;
            int count[11] = {0};
            for(int index : indices)
            {
                count[data[index].quality] += 1;
            }
            double sum = 0.0;
            for(int i = 0;i < 11;i++)
            {
                if(count[i] > 0)
                {
                    double p = count[i] / double(indices.size());
                    sum += p * p;
                }
            }
            return 1.0 - sum;           // Gini 不純度 = 1 - 所有出現機率的平方總和
        }
        void findBestSplitPoint(const std::vector<int>& indices,const std::vector<sample>& data,int& bestLabel,double& bestBar)
        {
            double minGini = 1.0;
            bestLabel = -1;
            bestBar = 0.0;
            // 經歷 11 種特徵
            for(int c = 0;c < 11;c++)
            {
                std::vector<std::pair<double,int>> valVSquality;
                for(int index : indices)
                {
                    valVSquality.push_back({data[index].character[c],data[index].quality});
                }
                // 先比較 first，相同才比較 second
                std::sort(valVSquality.begin(),valVSquality.end());
                // 左空右全滿
                int lcount[11] = {0},rcount[11] = {0},left = 0,right = indices.size();
                for(auto& p : valVSquality)
                {
                    rcount[p.second] += 1;
                }
                // 測試每一種門檻
                for(int i = 0;i < (valVSquality.size() - 1);i++)
                {
                    int q = valVSquality[i].second;
                    lcount[q] += 1;
                    rcount[q] -= 1;
                    right -= 1;
                    left += 1;
                    // 避免相同數值被切開
                    if((valVSquality[i].first == valVSquality[i + 1].first)) continue;
                    double lgini = 1.0,rgini = 1.0;
                    for(int k = 0;k < 11;k++)
                    {
                        if(left > 0)
                        {
                            double p = lcount[k] / (double)left;
                            lgini -= (p * p);
                        }
                        if(right > 0) 
                        {
                            double p = rcount[k] / (double)right;
                            rgini -= (p * p);
                        }
                    }
                    double gini = (left * lgini + right * rgini) / (double)indices.size();
                    if((gini < minGini))
                    {
                        minGini = gini;
                        bestLabel = c;
                        bestBar = valVSquality[i].first;
                    }
                }

            }
        }
        // 遞迴種樹
        node* build(const std::vector<int>& indices,const std::vector<sample>& data,int depth,int maxDepth)
        {
            if(data.size() == 0)    return nullptr;
            node* newones = new node();
            // 達到最大深度 、 全部都同一列 、 數據量太少
            if(depth >= maxDepth || calculateGini(indices,data) == 0 || data.size() < 9)
            {
                newones -> isLeaf = true;
                // 投票決定葉子之值
                int maxVote = -100;
                int count[11] = {0};
                for(int index : indices)
                {
                    int q = data[index].quality;
                    count[q] += 1;
                    if(count[q] > maxVote)
                    {
                        maxVote = count[q];
                        newones -> leafQuality = q;
                    }
                }
                return newones;
            }
            int bestLabel;
            double bestBar;
            findBestSplitPoint(indices,data,bestLabel,bestBar);
            // 找不到分割點
            if(bestLabel == -1)
            {
                newones -> isLeaf = true;
                // 投票決定葉子之值
                int maxVote = -100;
                int count[11] = {0};
                for(int index : indices)
                {
                    int q = data[index].quality;
                    count[q] += 1;
                    if(count[q] > maxVote)
                    {
                        maxVote = count[q];
                        newones -> leafQuality = q;
                    }
                }
                return newones;
            }
            newones -> characterIndex = bestLabel;
            newones -> bar = bestBar;
            std::vector<int> lindices,rindices;
            for(int index : indices)
            {
                if(data[index].character[bestLabel] <= bestBar)
                {
                    lindices.push_back(index);
                }else
                {
                    rindices.push_back(index);
                }
            }
            newones -> left = build(lindices,data,depth + 1,maxDepth);
            newones -> right = build(rindices,data,depth + 1,maxDepth);
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