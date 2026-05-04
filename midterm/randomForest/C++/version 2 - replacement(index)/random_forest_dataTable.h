#ifndef RANDOM_FOREST_DATATABLE_H
#define RANDOM_FOREST_DATATABLE_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// 儲存資料的 data type
struct sample
{
    // 依序是 fixed acidity 、 volatile acidity 、 citric acid 、 residual sugar 、 chlorides
    // free sulfur dioxide 、 total sulfur dioxide 、 density 、 pH 、 sulphates 、 alcohol
    std::vector<double> character;
    int quality;
};

class dataTable
{
    public:
        std::vector<sample> samples;
        // 此地方預設使用 txt 檔
        bool readfile(const std::string& name)
        {
            std::ifstream in;
            in.open(name);
            if(!in)
            {
                std::cout << "The file cannot open !" << std::endl;
                return false;
            }
            std::string feature;
            // 忽略 label
            std::getline(in,feature);
            // 正式輸入 samples
            while(std::getline(in,feature))
            {
                // 確保不是空行
                if(feature.empty() || feature == "\r") continue;
                std::stringstream input(feature);
                double state;
                sample temp;
                for(int i = 0;i < 13;i++)
                {
                    // i == 12 忽略 ID
                    if(!(input >> state)) break;
                    if(i < 11)
                        temp.character.push_back(state);
                    else if(i == 11)
                        temp.quality = (int)state;
                }
                // 把 residual sugar 取代成 alcohol / volatile acidity
                temp.character[3] = temp.character[10] / temp.character[1];
                samples.push_back(temp);
            }
            in.close();
            return true;
        }
        size_t size() const
        {
            return samples.size();
        }
};

#endif