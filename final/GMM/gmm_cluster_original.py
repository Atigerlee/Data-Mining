"""
 第二階段: Clustering — 手刻 GMM 原始版 K=2

 本檔做的事：
   1. 讀取分類階段的 rf_classification_result.csv
      欄位 = 原始特徵... | confidence_score | pred_label
   2. 只取出未知樣本 'Unlabeled' 的原始特徵, 用 StandardScaler 標準化
   3. 用高斯混合模型分成 K=2 群
   4. 依各群原始 Area 平均做標籤對應: Unknown Type A / B
   5. 輸出 gmm_original_result.csv 給 Phase 3 整合

 GMM 的兩個步驟:
   E 步: 用目前的參數, 算每個點屬於每一群的機率
   M 步: 用責任值回頭更新每群的權重、平均、共變異數
   反覆做直到 log-likelihood 收斂
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 15
N_CLUSTERS = 2


class GMM:
    def __init__(self, n_clusters, max_iter=300, tol=1e-5,
                 reg_covar=1e-6, n_init=10, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar      # 加在共變異數對角線上的小數, 避免矩陣不可逆
        self.n_init = n_init            # 換不同起始點重跑幾次, 挑最好的
        self.random_state = random_state

    # 多維高斯分布的 log 機率密度
    def _gaussian_log_pdf(self, samples, mean, cov):
        n_features = samples.shape[1]
        centered = samples - mean
        inv_cov = np.linalg.inv(cov)
        _, log_det = np.linalg.slogdet(cov)
        # 馬氏距離平方：對每個樣本算 (x-mean)·inv_cov·(x-mean)
        mahalanobis_sq = np.sum((centered @ inv_cov) * centered, axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis_sq)

    # 穩定 log(sum(exp(...)))
    @staticmethod
    def _log_sum_exp(arr, axis):
        max_val = np.max(arr, axis=axis, keepdims=True)
        return max_val.squeeze(axis) + np.log(np.sum(np.exp(arr - max_val), axis=axis))

    # 隨機挑幾個樣本當群中心, 共變異數先用整體共變異數
    def _init_params(self, samples, rng):
        n_samples, n_features = samples.shape
        center_idx = rng.choice(n_samples, self.n_clusters, replace=False)
        means = samples[center_idx].copy()
        weights = np.full(self.n_clusters, 1.0 / self.n_clusters)
        base_cov = np.cov(samples, rowvar=False) + self.reg_covar * np.eye(n_features)
        covs = np.array([base_cov.copy() for _ in range(self.n_clusters)])
        return weights, means, covs

    # E 步：算責任值與這一輪的 log-likelihood
    def _e_step(self, samples, weights, means, covs):
        n_samples = samples.shape[0]
        log_weighted = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            log_weighted[:, k] = np.log(weights[k] + 1e-12) + \
                self._gaussian_log_pdf(samples, means[k], covs[k])
        log_total = self._log_sum_exp(log_weighted, axis=1)     # 每列的正規化常數
        responsibility = np.exp(log_weighted - log_total[:, None])
        return responsibility, log_total.sum()

    # M 步：用責任值更新權重、平均、共變異數
    def _m_step(self, samples, responsibility):
        n_samples, n_features = samples.shape
        cluster_size = responsibility.sum(axis=0) + 1e-12       # 每群的軟性樣本數
        weights = cluster_size / n_samples
        means = (responsibility.T @ samples) / cluster_size[:, None]
        covs = np.zeros((self.n_clusters, n_features, n_features))
        for k in range(self.n_clusters):
            centered = samples - means[k]
            covs[k] = (responsibility[:, [k]] * centered).T @ centered / cluster_size[k]
            covs[k] += self.reg_covar * np.eye(n_features)
        return weights, means, covs

    # 單次完整 EM
    def _fit_once(self, samples, rng):
        weights, means, covs = self._init_params(samples, rng)
        last_log_likelihood = -np.inf
        for _ in range(self.max_iter):
            responsibility, log_likelihood = self._e_step(samples, weights, means, covs)
            weights, means, covs = self._m_step(samples, responsibility)
            if abs(log_likelihood - last_log_likelihood) < self.tol:
                break
            last_log_likelihood = log_likelihood
        return weights, means, covs, log_likelihood

    # 多組起始點各跑一次 EM, 留下 log-likelihood 最高的結果
    def fit(self, samples):
        seed_maker = np.random.default_rng(self.random_state)
        best_log_likelihood = -np.inf
        best_params = None
        for _ in range(self.n_init):
            rng = np.random.default_rng(seed_maker.integers(1 << 31))
            try:
                weights, means, covs, log_likelihood = self._fit_once(samples, rng)
            except np.linalg.LinAlgError:
                continue                # 偶爾遇到不可逆矩陣, 跳過這次起始點
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (weights, means, covs)
        self.weights_, self.means_, self.covs_ = best_params
        self.log_likelihood_ = best_log_likelihood
        return self

    def predict_proba(self, samples):
        responsibility, _ = self._e_step(samples, self.weights_, self.means_, self.covs_)
        return responsibility

    def predict(self, samples):
        return self.predict_proba(samples).argmax(axis=1)

    # 模型參數量（給 BIC / AIC 用）
    def count_parameters(self, n_features):
        cov_params = self.n_clusters * n_features * (n_features + 1) / 2   # 對稱共變異數
        mean_params = self.n_clusters * n_features
        weight_params = self.n_clusters - 1                               # 權重總和為 1
        return int(cov_params + mean_params + weight_params)

    def score(self, samples):
        _, log_likelihood = self._e_step(samples, self.weights_, self.means_, self.covs_)
        return log_likelihood

    def bic(self, samples):
        n_samples, n_features = samples.shape
        return -2 * self.score(samples) + self.count_parameters(n_features) * np.log(n_samples)

    def aic(self, samples):
        n_features = samples.shape[1]
        return -2 * self.score(samples) + 2 * self.count_parameters(n_features)

def main():
    data = pd.read_csv('rf_classification_result.csv')

    # 1. 只取未知樣本 Unlabeled
    unknown_mask = (data['pred_label'] == 'Unlabeled').values
    unknown = data[unknown_mask].copy().reset_index(drop=True)
    print(f'未知(Unlabeled)樣本數: {len(unknown)}')

    # 2. 取原始特徵並標準化後分群
    non_features = ('Class', 'confidence_score', 'pred_label')
    feature_cols = [col for col in data.columns if col not in non_features]
    samples = StandardScaler().fit_transform(unknown[feature_cols].values)

    # 3. GMM, K=2
    gmm = GMM(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    gmm.fit(samples)
    clusters = gmm.predict(samples)
    print(f'收斂 log-likelihood: {gmm.log_likelihood_:.2f}')
    print(f'BIC: {gmm.bic(samples):.2f}   AIC: {gmm.aic(samples):.2f}')
    for k in range(N_CLUSTERS):
        print(f'  Cluster {k}: {int((clusters == k).sum())} 筆, 權重 {gmm.weights_[k]:.3f}')

    # 4. 群編號 -> 中性標籤
    #    分群本身並不知道豆子叫什麼名字, 不硬套 BOMBAY/BARBUNYA，
    #    只給中性代號 Unknown_Type_A / Unknown_Type_B, 真正的名字對應留到第三階段決定
    unknown['Cluster'] = clusters
    cluster_to_label = {k: f'Unknown_Type_{chr(ord("A") + k)}' for k in range(N_CLUSTERS)}
    unknown['Cluster_Label'] = unknown['Cluster'].map(cluster_to_label)
    print('\n各群大小與原始 Area 平均:')
    for k in range(N_CLUSTERS):
        size = int((clusters == k).sum())
        mean_area = unknown.loc[unknown['Cluster'] == k, 'Area'].mean()
        print(f'  {cluster_to_label[k]}: {size} 筆, Area 平均 {mean_area:.0f}')

    # 5. 輸出
    output_cols = ['Area', 'Perimeter', 'confidence_score', 'pred_label',
                   'Cluster', 'Cluster_Label']
    unknown[output_cols].to_csv('gmm_original_result.csv',
                                index=False, encoding='utf-8-sig')
    print('\n已輸出: gmm_original_result.csv')

    # 6. 看每一群實際上是哪些真實類別
    true_class = pd.read_csv('dry_bean_test.csv')['Class'].values[unknown_mask]
    unknown['True_Class'] = true_class
    print('\n各群真實類別組成: ')
    for k in range(N_CLUSTERS):
        composition = Counter(unknown[unknown['Cluster'] == k]['True_Class'])
        print(f'  {cluster_to_label[k]}: {dict(composition)}')


if __name__ == '__main__':
    main()
