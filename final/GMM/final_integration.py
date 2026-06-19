"""
 Integration & Evaluation, Optimal Label Assignment

 流程：
   1. 讀取分類階段結果 rf_classification_result.csv
   2. 讀取分群階段結果未知樣本的中性群代號 Unknown_Type_A / Unknown_Type_B
   3. 最佳標籤對齊: 分群本身不知道哪一群是 BARBUNYA、哪一群是 BOMBAY,
      所以把 群代號 -> 新類別名字 的所有可能對應都試一次, 取整體準確率最高的那組
   4. 用最佳對應合併出完整預測, 對 dry_bean_test.csv 的真實標籤算準確率 / 各類別 F1 / 混淆矩陣
   5. 輸出 final_prediction.csv 與 confusion matrix

 執行時可帶一個參數, 指定要整合哪份分群結果 oringinal K=2 或 dynamic K

 ※ 準確率 / F1 的計算對象：
   - 預測 = 本階段合併出來的 Final_Predicted_Class
   - 答案 = dry_bean_test.csv 的 Class
   - 全部的測試樣本一起算, 訓練集不參與評估
"""

import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 可讀 original K 或 dynamic K 的結果
gmm_result_file = 'gmm_dynamicK_result.csv'


def main():
    rf_result = pd.read_csv('rf_classification_result.csv')
    gmm_result = pd.read_csv(gmm_result_file)
    y_true = pd.read_csv('dry_bean_test.csv')['Class'].values   # 真實標籤（評估用）

    # 未知樣本 = 分類階段標成 'Unlabeled' 的列；其列順序與分群結果一致
    unknown_mask = (rf_result['pred_label'] == 'Unlabeled').values
    assert unknown_mask.sum() == len(gmm_result), '未知樣本數與分群結果列數不一致'

    # 已知類別 = RF 給出的（非 Unlabeled）標籤；新類別 = 真實標籤裡 RF 從沒預測過的那些
    known_classes = set(rf_result.loc[~unknown_mask, 'pred_label'].unique())
    new_class_names = sorted(set(y_true) - known_classes)
    cluster_labels = sorted(gmm_result['Cluster_Label'].unique())
    print(f'分群得到的群代號 : {cluster_labels}')
    print(f'測試集中的新類別 : {new_class_names}')
    assert len(cluster_labels) <= len(new_class_names), '群數多於新類別數，無法一對一對齊'

    # 最佳標籤對齊：試遍「群代號 -> 新類別名字」的所有排列，取整體準確率最高者
    base_label = rf_result['pred_label']        # 已知類別沿用 RF 標籤
    best_accuracy, best_mapping, best_pred = -1.0, None, None
    print('\n=== 最佳標籤對齊 (Optimal Label Assignment) ===')
    for perm in itertools.permutations(new_class_names, len(cluster_labels)):
        mapping = dict(zip(cluster_labels, perm))
        final_label = base_label.copy()
        final_label.loc[unknown_mask] = gmm_result['Cluster_Label'].map(mapping).values
        accuracy = accuracy_score(y_true, final_label.values)
        print(f'  {mapping} -> 整體準確率 {accuracy:.4f}')
        if accuracy > best_accuracy:
            best_accuracy, best_mapping, best_pred = accuracy, mapping, final_label.values
    print(f'>>> 採用最佳對應: {best_mapping}')

    y_pred = best_pred
    overall_accuracy = best_accuracy
    print(f'\n最終整體準確率 (7 類, 共 {len(y_true)} 筆): {overall_accuracy:.4f}')

    # 兩個子集各自的準確率
    high_conf_mask = ~unknown_mask
    print(f'High_Confidence 子集 ({high_conf_mask.sum():>4} 筆) 準確率: '
          f'{accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask]):.4f}')
    print(f'Low_Confidence  子集 ({unknown_mask.sum():>4} 筆) 準確率: '
          f'{accuracy_score(y_true[unknown_mask], y_pred[unknown_mask]):.4f}')

    # 各類別 Precision / Recall / F1
    class_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    print('\n=== 各類別 Precision / Recall / F1 ===')
    print(classification_report(y_true, y_pred, labels=class_labels, zero_division=0))

    # 混淆矩陣（文字 + 圖）
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    print('=== 混淆矩陣, 列=真實, 欄=預測 ===')
    print(pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    heatmap = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Overall Acc = {overall_accuracy:.3f})')
    for row in range(len(class_labels)):
        for col in range(len(class_labels)):
            count = conf_matrix[row, col]
            text_color = 'white' if count > conf_matrix.max() / 2 else 'black'
            ax.text(col, row, count, ha='center', va='center', color=text_color, fontsize=8)
    fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)

    # 輸出最終預測
    output = rf_result[['Area', 'Perimeter', 'confidence_score', 'pred_label']].copy()
    output['Final_Predicted_Class'] = y_pred
    output['True_Class'] = y_true
    output.to_csv('final_prediction.csv', index=False, encoding='utf-8-sig')
    print('\n已輸出: final_prediction.csv, confusion_matrix.png')


if __name__ == '__main__':
    main()
