#include <iostream>
#include<fstream>
#include<cstdlib>
#include<string>
#include<vector>
#include<cmath>
#include<sstream>
#include<algorithm>
#include<map>
#include<random>
#include<ctime>
#include<iomanip>
#include<chrono>
using namespace std;

struct z_score_para
{
    double mean[11];
    double std[11];
};

struct Result_data //用來儲存要從CalAndOutputResult回傳的資料
{
    double bestMacroF1_score = 0.0;//紀錄最佳的Macro f1
    double bestAccuracy = 0.0;//紀錄最佳acc
    int bestK = 0;//紀錄最佳k
};

struct Wine { //儲存每種酒的資料
    vector<double>feature; //儲存酒的11種特徵
    int quality = 0; //酒的品質
};

struct Distance {//儲存距離和對應得quilt
    double dist;
    int quality;
};

bool compareDist(const Distance& a, const Distance& b) //定義Distance的比較規則 比較其中的dist就好
{
    return a.dist < b.dist;
}

double CalEucliDis(const Wine& a, const Wine& b, const vector<double>& feature_CC); //計算兩組資料之間的歐幾里得距離 且有使用權重

vector<Wine> ReadCSV(string file);//將資料從.csv檔案中讀出來

void normalizationData_minMax(vector<Wine>& trainData, vector<Wine>& testData); //min-max規一化
void normalizationData_Zscore(vector<Wine>& trainData, vector<Wine>& testData);//Z-score歸一化


int predict(const vector<Distance>& dist_data, int k);//傳入測試資料 訓練資料 和 K值 和 相關係數

Result_data CalAndOutputResult(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<vector<Distance>>& all_Test_dist_data, vector<double>&total_f1_score);
Result_data CalAndOutputResult(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<vector<Distance>>& all_Test_dist_data, int bestK);

void OutputTheBestResult(Result_data R, char c_normalized, bool CC);

vector<double> featureWeighting(const vector<Wine>& data); //計算權重

vector<vector<Distance>> CalDistanceDataBetweenTest_Traindata(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<double>& feature_CC);
//計算每一筆testData和全部TraiaData的Distance資料 用於之後計算不同k

int main()
{
    vector<vector<Distance>>all_Test_dist_data;
    Result_data R;
    int fold = 5;
    int fold_size = 0;
    int numOfData;//儲存data的總數
    vector<Wine>cvData;
    vector<Wine>data;//存全部資料
    vector<Wine>testData;//驗證集
    string filename;//儲存資料集的檔案名稱
    char c_normalized;
    char c_weight;
    vector<double> total_f1_score;
    int bestK = 0;
    double bestK_val = 0.0;
    vector<double>feature_CC;//各特徵的相關係數
    bool CC; //代表是否使用特徵權重
    vector<vector<Wine>>quality_num_data(10); //用於統計每一個quality有幾筆資料 要使用分層抽樣使用

    for (int i = 0; i < 400; i++) total_f1_score.push_back(0);
    //quality_num_data.reserve(10);

    cout << "Input file name:";
    cout << "WineQT.csv\n";
    filename = "WineQT.csv";
    data = ReadCSV(filename);//將資料讀進data中


    cout << "Use min-max(m) or Z-score(z):"; //判斷要使用min-max 還是 z-score
    cin >> c_normalized;
    cout << "Use feature weight(y/n):"; //要判斷要不要使用feature weight
    cin >> c_weight;



    numOfData = data.size() * 0.2;//以8:2的比例拆分資料集


    for(int i = 0; i<data.size();i++)   quality_num_data[data[i].quality].push_back(data[i]);  //將data根據其quality進行分類
    
    mt19937 g(15);
    for(int i=0; i<quality_num_data.size(); i++)    shuffle(quality_num_data[i].begin(),quality_num_data[i].end(),g);
    
    for(int i=0; i<quality_num_data.size(); i++) //進行分層抽樣
    {
        int num_data = round(quality_num_data[i].size() * 0.2); //將每個quality的資料 其中20%給testData 80%給cvData
        if(quality_num_data[i].size() !=0 && num_data==0) num_data=1; //其中 保證至少在總數非零的時候至少拿到一格
        for(int j=0; j<quality_num_data[i].size(); j++)
        {
            if(j<num_data) testData.push_back(quality_num_data[i][j]);
            else cvData.push_back(quality_num_data[i][j]);

        }
    }
    
    shuffle(cvData.begin(),cvData.end(),g);
    shuffle(testData.begin(),testData.end(),g);


    for (int i = 0; i < fold; i++) {
        vector<Wine>trainData;//存訓練資料
        vector<Wine>valData;//存驗證資料
        z_score_para Z;
        int data_start,data_end;

        data_start =cvData.size() / 5 * i;//測試集的起點
        data_end = cvData.size() / 5 * (i + 1);//測試集的終點
        
        for (int j = 0; j < cvData.size(); j++)
        {
            if (j >= data_start && j < data_end) valData.push_back(cvData[j]);
            else trainData.push_back(cvData[j]);
        }

       
        if (c_normalized == 'm')
        {
            normalizationData_minMax(trainData, valData);
        }//初始資料集用min max整理完成
        else
        {
            normalizationData_Zscore(trainData, valData);
        }//初始資料集用Z-score 整理完成


        
        if (c_weight == 'y') CC = 1;
        else CC = 0;

       // auto start = chrono::high_resolution_clock::now();//時間記錄開始(不紀錄使用者輸入選項的時間)

        if (CC)feature_CC = featureWeighting(trainData);
        else for (int i = 0; i < 11; i++)feature_CC.push_back(1);//如果沒有要使用特徵權重 就將權重(相關係數平方)設為1

        all_Test_dist_data = CalDistanceDataBetweenTest_Traindata(valData, trainData, feature_CC);//進行預處理

        R = CalAndOutputResult(valData, trainData, all_Test_dist_data,total_f1_score);
        //OutputTheBestResult(R, c_normalized, CC);

        //auto end = chrono::high_resolution_clock::now();//時間記錄結束
        //chrono::duration<double> duration = end - start;//計算時間
        //cout << "Running time: " << duration.count() << " seconds" << endl;
    }

    for (int i = 11; i <=301; i++)
    {
        if (total_f1_score[i] > bestK_val)
        {
            bestK_val = total_f1_score[i];
            bestK = i;
        }
    }

    if (c_normalized == 'm')
    {
        normalizationData_minMax(cvData, testData);
    }//初始資料集用min max整理完成
    else
    {
        normalizationData_Zscore(cvData, testData);
    }//初始資料集用Z-score 整理完成

    if (CC)feature_CC = featureWeighting(cvData);
    else for (int i = 0; i < 11; i++)feature_CC.push_back(1);//如果沒有要使用特徵權重 就將權重(相關係數平方)設為1

    all_Test_dist_data = CalDistanceDataBetweenTest_Traindata(testData, cvData, feature_CC);//進行預處理
    R = CalAndOutputResult(testData, cvData, all_Test_dist_data, bestK);
    
    cout << "Best F1-score " << R.bestMacroF1_score << endl;
    cout << "Best Accuracy " << R.bestAccuracy << endl;
    if(CC==1) cout << "Feature weight:True\n"; else cout << "Feature weight:False\n";
    cout << "Stratified Sampling:True";

}

/*===================================================*//*===================================================*//*===================================================*/

vector<vector<Distance>> CalDistanceDataBetweenTest_Traindata(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<double>& feature_CC)//對資料進行預處理 避免後面重複運算
{//計算每筆testData和整個trainData的關係
    vector<vector<Distance>>all_Test_dist_data;
    Distance D;
    for (int i = 0; i < testData.size(); i++)//計算每筆測試資料所有trainData的關係
    {
        vector<Distance>dist_data;
        for (int j = 0; j < trainData.size(); j++)//計算訓練集中每筆資料跟傳入的test資料的距離 將算好的距離跟該trainData的quality一同存入dsit_data中
        {
            D.dist = CalEucliDis(trainData[j], testData[i], feature_CC);
            D.quality = trainData[j].quality;
            dist_data.push_back(D);
        }
        sort(dist_data.begin(), dist_data.end(), compareDist);//排序所有距離 可以得到由近到遠的距離排序 然後根據k值取最近ㄉ前幾個
        all_Test_dist_data.push_back(dist_data);
    }
    return all_Test_dist_data;
}

/*===================================================*//*===================================================*//*===================================================*/

void OutputTheBestResult(Result_data R, char c_normalized, bool CC)
{
    cout << endl << "best macro f1 score: " << R.bestMacroF1_score << endl;
    cout << "best K(when best f1 score) = " << R.bestK << endl;
    cout << "Total best accuracy: " << R.bestAccuracy << endl;
    if (c_normalized == 'm') cout << "Use model:min-max\n";
    else cout << "Use model:Z-score\n";

    if (CC)cout << "Feature weighting:True\n";
    else cout << "Feature weighting:False\n";
}

/*===================================================*//*===================================================*//*===================================================*/

vector<double> featureWeighting(const vector<Wine>& data)//用個特徵的相關係數平方作為權重 在計算距離時會加入
{
    double feature_mean[11]; //各特徵的平均值
    double data_mean = 0.0; //原始資料的平均值
    double total = 0;
    vector<double>feature_CC;//各特徵的相關係數


    for (int i = 0; i < 11; i++) { feature_CC.push_back(0); feature_mean[i] = 0; } //初始化

    for (int i = 0; i < data.size(); i++)//將data feature的值 加到feature mean裡面
        for (int j = 0; j < 11; j++) feature_mean[j] += data[i].feature[j];

    for (int i = 0; i < 11; i++)feature_mean[i] /= data.size(); //計算各feature的平均值

    for (int i = 0; i < data.size(); i++)data_mean += data[i].quality; //計算quality的平均值
    data_mean /= data.size();

    for (int i = 0; i < 11; i++)
    {
        double mole = 0, deno1 = 0, deno2 = 0; //計算相關係數的分子分母

        for (int j = 0; j < data.size(); j++) //計算各特徵相關係數
        {
            mole += (data[j].quality - data_mean) * (data[j].feature[i] - feature_mean[i]);
            deno1 += pow(data[j].quality - data_mean, 2);
            deno2 += pow(data[j].feature[i] - feature_mean[i], 2);
        }
        if (deno1 == 0 || deno2 == 0) feature_CC[i] = 0; //避免出現分母為0的情況
        else feature_CC[i] = pow(mole / sqrt(deno1 * deno2), 2); //取正的相關係數
    }

    for (int i = 0; i < feature_CC.size(); i++) total += feature_CC[i];
    for (int i = 0; i < feature_CC.size(); i++) feature_CC[i] /= total;


    return feature_CC;//儲存11個特徵的相關係數平方
}

/*===================================================*//*===================================================*//*===================================================*/

Result_data CalAndOutputResult(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<vector<Distance>>& all_Test_dist_data, vector<double>& total_f1_score)
{
    Result_data R;
    double testTotalQuality[10];//存取所有測試quality=0~6數量的 用於計算recall f1-score etc..
    double predictTotalQuality[10];//存取所有預測的quality 相當於猜了這個quality幾次
    double predictCorrectQuality[10];//存取 猜對該quality的數量
    double recall[10], precision[10], f1_score[10];
    int tempPredictResult;
    double macroF1_score = 0.0;
    double accuracy = 0.0;
    double bestMacroF1_score = 0.0;//紀錄最佳的Macro f1
    double bestAccuracy = 0.0;
    int bestK = 0;

    for (int j = 11; j <= 301; j += 1) {//將k從11跑到301避免k過小

        for (int i = 0; i < 10; i++)//初始化陣列
        {
            testTotalQuality[i] = 0.0;
            predictTotalQuality[i] = 0.0;
            predictCorrectQuality[i] = 0.0;
            recall[i] = 0.0;
            precision[i] = 0.0;
            f1_score[i] = 0.0;
        }

        for (int i = 0; i < testData.size(); i++)//統計測試集和預測資料的quality資料數量
        {
            testTotalQuality[testData[i].quality]++;//統計各quality的數量

            tempPredictResult = predict(all_Test_dist_data[i], j);//用k進行預測
            predictTotalQuality[tempPredictResult]++;//將預測結果儲存

            if (tempPredictResult == testData[i].quality)predictCorrectQuality[tempPredictResult]++; //如果預測正確 就將predictCorrectQuality的對應++
        }


        for (int i = 0; i < 10; i++)//計算precision recall f1_score 其中 predictTotalQuality是預測到該quality的次數 predictCorrectQuality是預測且答案正確的次數
        {
            if (predictCorrectQuality[i] == 0 || predictTotalQuality[i] == 0)precision[i] = 0; //計算precision
            else precision[i] = predictCorrectQuality[i] / predictTotalQuality[i];

            if (predictCorrectQuality[i] == 0 || testTotalQuality[i] == 0)recall[i] = 0;//計算recall
            else recall[i] = predictCorrectQuality[i] / testTotalQuality[i];

            if (precision[i] == 0 && recall[i] == 0)f1_score[i] = 0;
            else f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]);
        }
      
        macroF1_score = 0.0;
        accuracy = 0.0;
        for (int i = 3; i <= 8; i++)//計算在當前K值的情況下 macro f1 score的數值 並檢查是否為當前最大的
        {
            macroF1_score += f1_score[i];
            accuracy += predictCorrectQuality[i];
        }
        macroF1_score /= 6;

        if (accuracy >= bestAccuracy)bestAccuracy = accuracy;
        if (macroF1_score >= bestMacroF1_score)
        {
            bestK = j;
            bestMacroF1_score = macroF1_score;
        }
        total_f1_score[j] += macroF1_score;
    }

    R.bestAccuracy = bestAccuracy / testData.size();
    R.bestK = bestK;
    R.bestMacroF1_score = bestMacroF1_score;
    return R;
}

/*===================================================*//*===================================================*//*===================================================*/

Result_data CalAndOutputResult(const vector<Wine>& testData, const vector<Wine>& trainData, const vector<vector<Distance>>& all_Test_dist_data, int bestK)
{
    Result_data R;
    double testTotalQuality[10];//存取所有測試quality=0~6數量的 用於計算recall f1-score etc..
    double predictTotalQuality[10];//存取所有預測的quality 相當於猜了這個quality幾次
    double predictCorrectQuality[10];//存取 猜對該quality的數量
    double recall[10], precision[10], f1_score[10];
    int tempPredictResult;
    double macroF1_score = 0.0;
    double accuracy = 0.0;
    double bestMacroF1_score = 0.0;//紀錄最佳的Macro f1
    double bestAccuracy = 0.0;

    
        for (int i = 0; i < 10; i++)//初始化陣列
        {
            testTotalQuality[i] = 0.0;
            predictTotalQuality[i] = 0.0;
            predictCorrectQuality[i] = 0.0;
            recall[i] = 0.0;
            precision[i] = 0.0;
            f1_score[i] = 0.0;
        }

        for (int i = 0; i < testData.size(); i++)//統計測試集和預測資料的quality資料數量
        {
            testTotalQuality[testData[i].quality]++;//統計各quality的數量

            tempPredictResult = predict(all_Test_dist_data[i], bestK);//用k進行預測
            predictTotalQuality[tempPredictResult]++;//將預測結果儲存

            if (tempPredictResult == testData[i].quality)predictCorrectQuality[tempPredictResult]++; //如果預測正確 就將predictCorrectQuality的對應++
        }


        for (int i = 0; i < 10; i++)//計算precision recall f1_score 其中 predictTotalQuality是預測到該quality的次數 predictCorrectQuality是預測且答案正確的次數
        {
            if (predictCorrectQuality[i] == 0 || predictTotalQuality[i] == 0)precision[i] = 0; //計算precision
            else precision[i] = predictCorrectQuality[i] / predictTotalQuality[i];

            if (predictCorrectQuality[i] == 0 || testTotalQuality[i] == 0)recall[i] = 0;//計算recall
            else recall[i] = predictCorrectQuality[i] / testTotalQuality[i];

            if (precision[i] == 0 && recall[i] == 0)f1_score[i] = 0;
            else f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]);
        }

        macroF1_score = 0.0;
        accuracy = 0.0;
        
        for (int i = 3; i <= 8; i++)
        {
            macroF1_score += f1_score[i];
            accuracy += predictCorrectQuality[i];
        }
        macroF1_score /= 6; // 算出 Macro F1 (6個類別的平均)

        bestMacroF1_score = macroF1_score;
        bestAccuracy = accuracy;

        cout << left << setw(15) << "Class" << setw(15) << "| Precision" << setw(15) << "| Recall" << setw(15) << "| F1_score\n";
        cout << "--------------------------------------\n";
        cout << fixed << setprecision(4);
        for (int i = 3; i <= 8; i++)
        {
            cout << left << setw(15) << i << setw(10) << " | " << precision[i] << setw(10) << " | " << recall[i] << setw(10) << " | " << f1_score[i] << endl;
        }

    R.bestAccuracy = bestAccuracy / testData.size();
    R.bestK = bestK;
    R.bestMacroF1_score = bestMacroF1_score;
    return R;
}
/*===================================================*//*===================================================*//*===================================================*/

void normalizationData_Zscore(vector<Wine>& trainData, vector<Wine>& testData)//先計算trainData的mean 和 std 然後再用這組數字去標準化 trainData和 testData
{

    for (int i = 0; i < 11; i++)
    {
        double ave = 0.0; //單一特徵的平均數
        double SD = 0.0; //單一特徵的標準差
        for (int j = 0; j < trainData.size(); j++) //計算trainData平均數
            ave += trainData[j].feature[i];

        ave /= trainData.size();

        for (int j = 0; j < trainData.size(); j++)//計算trainData標準差
            SD += pow(trainData[j].feature[i] - ave, 2);

        SD /= trainData.size();
        SD = sqrt(SD);

        for (int j = 0; j < trainData.size(); j++)//標準化trainData
        {
            trainData[j].feature[i] -= ave;
            if (SD == 0) trainData[j].feature[i] = 0; //避免SD為0是 程式崩潰的問題
            else trainData[j].feature[i] /= SD;
        }

        for (int j = 0; j < testData.size(); j++)//標準化testData
        {
            testData[j].feature[i] -= ave;
            if (SD == 0) testData[j].feature[i] = 0; //避免SD為0是 程式崩潰的問題
            else testData[j].feature[i] /= SD;
        }

    }
}


/*===================================================*//*===================================================*//*===================================================*/



/*===================================================*//*===================================================*//*===================================================*/
int predict(const vector<Distance>& dist_data, int k)//用來預測quality 有加入權重 權重 是1/(distance+ε) ε:極小值 避免分母為0
{
    const double EPS = 1; //Epsilon作為極小數字從而避免分母為0
    map<int, double>vote;//用來儲存quality的票數 其中 int代表quality的值 double存放票數
    double maxVote = 0;//紀錄最高票數
    int maxQuality = 0; //紀錄最高票的quality



    for (int i = 0; i < k; i++) //選距離最近的k個來投票quality
    {
        vote[dist_data[i].quality] += 1 / (dist_data[i].dist + EPS);
    }

    for (const auto& item : vote)//遍歷整個map
    {
        if (item.second > maxVote)//如果票數比最高票還多的話 就會刷紀錄
        {
            maxVote = item.second;
            maxQuality = item.first;
        }
    }

    return maxQuality;
}

/*===================================================*//*===================================================*//*===================================================*/

void normalizationData_minMax(vector<Wine>& trainData, vector<Wine>& testData)
{
    for (int i = 0; i < 11; i++)
    {
        double min, max; //儲存各特徵的最大最小 且因為所有data >= 0 所以將min設為0合法
        min = trainData[0].feature[i]; //將min設為第一個wine的feature
        max = trainData[0].feature[i]; //將max設為第一個wine的feature
        for (int j = 1; j < trainData.size(); j++) //找出最大最小值
        {
            if (trainData[j].feature[i] > max)
                max = trainData[j].feature[i];
            if (trainData[j].feature[i] < min)
                min = trainData[j].feature[i];
        }
        double range = max - min; //表示max min的差距 用於後面處理max-min=0的情況
        if (range == 0)
        {
            for (int j = 0; j < trainData.size(); j++) //因為max , min值一樣 代表整組資料都相同 就全部規到0
                trainData[j].feature[i] = 0;
        }
        else
        {
            for (int j = 0; j < trainData.size(); j++) //對train進行min-max規一化
                trainData[j].feature[i] = (trainData[j].feature[i] - min) / (max - min);
            for (int j = 0; j < testData.size(); j++) //對test進行min-max規一化
                testData[j].feature[i] = (testData[j].feature[i] - min) / (max - min);
        }
    }
}

/*===================================================*//*===================================================*//*===================================================*/

vector<Wine> ReadCSV(string file)
{
    vector<Wine> data; //Wine陣列
    double dou;//暫存用
    int num;//暫存用整數
    string temp; //暫時儲存用字串
    fstream WineQT;
    try
    {
        WineQT.open(file, ios::in);//開啟檔案
        if (!WineQT)//如果開啟失敗
        {
            throw file;
        }
    }
    catch (string file)
    {
        cout << file << " can't be opened\n";
    }


    getline(WineQT, temp); //把第一行的文字讀掉

    while (!WineQT.eof()) //開始讀資料
    {
        Wine w;
        w.feature.clear();//清除陣列資料
        w.feature.shrink_to_fit();//釋放記憶體

        getline(WineQT, temp, ','); //第一次在for外面讀 如果是空白就直接break 避免問題
        if (temp == "")break;
        dou = stod(temp);
        w.feature.push_back(dou);
        for (int i = 1; i < 11; i++)//讀取前11個feature
        {
            getline(WineQT, temp, ',');
            dou = stod(temp);
            w.feature.push_back(dou);
        }
        getline(WineQT, temp, ',');//讀取quality
        num = stoi(temp);
        w.quality = num;
        getline(WineQT, temp, '\n'); //ID用不到 所以讀掉
        data.push_back(w);
    }
    WineQT.close();
    return data;
}

/*===================================================*//*===================================================*//*===================================================*/

double CalEucliDis(const Wine& a, const Wine& b, const vector<double>& feature_CC) { //計算兩組資料之間的歐幾里的距離 含權重(overloading)
    //假設資料同維度
    double distance = 0.0;
    for (int i = 0; i < a.feature.size(); i++)
    {
        distance += feature_CC[i] * pow(a.feature[i] - b.feature[i], 2); //計算不同特徵的平方和
    }
    distance = sqrt(distance);//將距離開根號 完成歐幾里得距離的計算
    return distance;//回傳兩組之廖間的歐幾里得距離

    /*===================================================*//*===================================================*//*===================================================*/
}