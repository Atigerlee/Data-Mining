#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

const std::string fileName = "rf_classification_result.csv";
// 搭配 PCA 
const int fnumbers = 16;
const double bar = 0.0;
const int kMin = 2;
const int kMax = 16;
const std::string known1 = "DERMASON";
const std::string known2 = "CALI";
const std::string known3 = "HOROZ";
const std::string known4 = "SIRA";
const std::string known5 = "SEKER";
const std::string condition = "Unlabeled";

struct Data
{
    std::vector<double> features;
    double score;
    std::string tlabel;
    std::string plabel;
    int group;
    Data() : score(0),tlabel(""),plabel(""),group(0){};
};

// 歐氏距離
// 原本用 Distance(const Data& d1,const Data& d2)
double Distance(const std::vector<double>& d1f,const std::vector<double>& d2f,int mode = 0)
{
    double total = 0.0;
    for(int i = 0;i < fnumbers;i++)
    {
        double d = d1f[i] - d2f[i];
        total += (d * d);
    } 
    if(mode == 0) return sqrt(total);
    else return total;
}

int main()
{
    // phase 1 : 讀取檔案
    std::ifstream inFile;
    inFile.open(fileName);
    if(!inFile)
    {
        std::cout << "Cannot successfully open the file !!!" << std::endl;
        return 1;
    }
    std::ofstream outFile;
    outFile.open("plot.txt");
    if(!outFile) std::cout << "Cannot successfully open the output file !!!" << std::endl;
    std::vector<Data> udata;
    int kdata = 0;
    int ckdata = 0;
    std::string dumb;
    std::getline(inFile,dumb);
    std::string line;
    while(std::getline(inFile,line))
    {
        Data cd;
        int count = 0;
        std::string s = "";
        for(char c : line)
        {
            if(c == ',')
            {
                count++;
                if(count <= fnumbers)
                    cd.features.push_back(std::stod(s));
                else if(count == (fnumbers + 1))
                    cd.tlabel = s;
                else if(count == (fnumbers + 2))
                    cd.score = std::stod(s);
                s = "";
            }else
            {
                s += c;
            }
        }
        // 怕 換行符號為 \r\n
        s.erase(std::remove(s.begin(),s.end(),'\r'),s.end());
        cd.plabel = s;
        if(s != condition)
        {
            kdata++;
            if(cd.plabel == cd.tlabel) ckdata++;
        }else
        {
            udata.push_back(cd);
        }
    }
    inFile.close();
    // 資料量太少不支持分群
    if(udata.size() < kMin)
    {
        std::cout << "Datas are not enough to split !!!" << std::endl;
        return 1;
    }
    // phase 2 : Z-score
    for(int i = 0;i < fnumbers;i++)
    {
        double total = 0.0;
        for(int j = 0;j < udata.size();j++) total += udata[j].features[i];
        double avg = total / udata.size();
        double var = 0.0;
        for(int j = 0;j < udata.size();j++) var += ((udata[j].features[i] - avg) * (udata[j].features[i] - avg));
        double std = sqrt(var / udata.size());
        if(std != 0)
        {
            for(int j = 0;j < udata.size();j++)
                udata[j].features[i] = (udata[j].features[i] - avg) / std;
        }
    }
    std::cout << "=================================================================================" << std::endl;
    std::cout << "We test K from " << kMin << " to " << kMax << " with " << fnumbers << " features." << std::endl;
    std::cout << "=================================================================================" << std::endl;
    for(int K = kMin;K <= kMax;K++)
    {
        // phase 3 : K-means++
        std::random_device rd;
        std::mt19937 gen(rd());
        // 原本用 std::vector<Data>
        std::vector<std::vector<double>> center(K);
        std::uniform_int_distribution<> dis(0,udata.size() - 1);
        // 找 K 個群心
        center[0] = udata[dis(gen)].features;
        for(int i = 1;i < K;i++)
        {
            std::vector<double> md(udata.size(),999999999.9);
            for(int j = 0;j < udata.size();j++)
            {
                // 原本長 for(int k = 0;k < center.size();k++)
                for(int k = 0;k < i;k++)
                {
                    double d = Distance(udata[j].features,center[k],1);
                    // 這裡取為小的
                    if(d < md[j]) md[j] = d; 
                }
            }
            std::discrete_distribution<> distr(md.begin(),md.end());
            center[i] = udata[distr(gen)].features;
        }
        int run;
        // 前面已經污染了 -> 回歸原始
        for(int i = 0;i < udata.size();i++) udata[i].group = 0;
        // phase 4 : 找出最終結果
        for(int i = 0;;i++)
        {
            bool change = false;
            for(int j = 0;j < udata.size();j++)
            {
                int g = 0;
                double md = 999999999.9;
                for(int k = 0;k < center.size();k++)
                {
                    double cd = Distance(center[k],udata[j].features);
                    if(cd < md)
                    {
                        md = cd;
                        g = k;
                    }
                }
                if(udata[j].group != g)
                {
                    change = true;
                    udata[j].group = g;
                }
            }
            if(!change)
            {
                run = i + 1;
                break;
            }
            // 新群心就是平均
            std::vector<std::vector<double>> nc(K,std::vector<double>(fnumbers,0.0));
            std::vector<int> total(K,0);
            for(int j = 0;j < udata.size();j++)
            {
                int g = udata[j].group;
                total[g]++;
                for(int k = 0;k < fnumbers;k++)
                {
                    nc[g][k] += udata[j].features[k];
                }
            }
            for(int j = 0;j < K;j++)
            {
                if(total[j] != 0)
                {
                    for(int k = 0;k < fnumbers;k++)
                    {
                        center[j][k] = nc[j][k] / (double)total[j];
                    }
                }else
                {
                    // 沒人是的話就在隨機找一個
                    center[j] = udata[gen() % udata.size()].features;
                }
            }
        }
        // phase 5 : 計算 WCSS(elbow) 和 Macro F1-score 和 Weighted F1-score 和 accuracy
        double wcss = 0.0;
        for(int i = 0;i < udata.size();i++)
        {
            // 這裡是用距離的平方
            wcss += Distance(udata[i].features,center[udata[i].group],1);
        }
        std::string a = "";
        std::string b = "";
        int ranumbers = 0;
        int rbnumbers = 0;
        for(int i = 0;i < udata.size();i++)
        {
            std::string s = udata[i].tlabel;
            if(a == "" && s != known1 && s != known2 && s != known3 && s != known4 && s != known5) a = s;
            else if(b == "" && s != a && s != known1 && s != known2 && s != known3 && s != known4 && s != known5) b = s;
            if(s == a) ranumbers++;
            else if(s == b) rbnumbers++;
        }
        std::vector<std::string> gname(K,"");
        int tnumbers = 0;
        // 多數決
        for(int i = 0;i < K;i++)
        {
            int anumbers = 0;
            int bnumbers = 0;
            for(int j = 0;j < udata.size();j++)
            {
                if(udata[j].group == i)
                {
                    if(udata[j].tlabel == a) anumbers++;
                    else if(udata[j].tlabel == b) bnumbers++;
                }
            }
            // 等號就取 a
            if(anumbers >= bnumbers)
            {
                gname[i] = a;
                tnumbers += anumbers;
            }else
            {
                gname[i] = b;
                tnumbers += bnumbers;
            }
        }
        int TPa = 0,FPa = 0,FNa = 0;
        int TPb = 0,FPb = 0,FNb = 0;
        for(int i = 0;i < udata.size();i++)
        {
            int g = udata[i].group;
            std::string t = udata[i].tlabel;
            std::string p = gname[g];
            // a
            if(t == a && p == a) TPa++;
            else if(p == a && t != a) FPa++;
            else if(t == a && p != a) FNa++;
            // b
            if(t == b && p == b) TPb++;
            else if(p == b && t != b) FPb++;
            else if(t == b && p != b) FNb++;
        }
        // a
        double pa = ((TPa + FPa) != 0) ? (TPa / (double)(TPa + FPa)) : 0;
        double ra = ((TPa + FNa) != 0) ? (TPa / (double)(TPa + FNa)) : 0;
        double f1a = ((pa + ra) != 0) ? (2 * pa * ra / (pa + ra)) : 0;
        // b
        double pb = ((TPb + FPb) != 0) ? (TPb / (double)(TPb + FPb)) : 0;
        double rb = ((TPb + FNb) != 0) ? (TPb / (double)(TPb + FNb)) : 0;
        double f1b = ((pb + rb) != 0) ? (2 * pb * rb / (pb + rb)) : 0;
        double mf1 = (f1b + f1a) / 2.0;
        // 確認定義一致
        double total = (double)(ranumbers + rbnumbers);
        double wf1 = (total != 0) ? ((f1a * ranumbers + f1b * rbnumbers) / total) : 0.0;
        double cacc = (udata.size() != 0) ? ((double)tnumbers / udata.size() * 100.0) : 0.0;
        double tacc = (double)(ckdata + tnumbers) / (kdata + udata.size()) * 100.0;
        std::cout << "[K = " << K << " result]" << std::endl;
        std::cout << "The size of total datas : " << (udata.size() + kdata) << std::endl;
        std::cout << "The size of unknown datas : " << udata.size() << std::endl;
        std::cout << "Numbers of iterations : " << run << " times." << std::endl;
        std::cout << "Cluster WCSS : " << wcss << std::endl;
        outFile << wcss << std::endl;
        std::cout << "Cluster Macro F1-score : " << mf1 << std::endl;
        std::cout << "Cluster Weighted F1-score : " << wf1 << std::endl;
        std::cout << "Cluster accuracy : " << cacc << " %" << std::endl;
        std::cout << "Overall accuracy : " << tacc << " %" << std::endl;
        std::cout << "---------------------------------------------------------------------------------" << std::endl;
    }
    outFile.close();
    return 0;
}