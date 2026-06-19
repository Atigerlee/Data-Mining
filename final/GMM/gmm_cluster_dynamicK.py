"""
 GMM + Dynamic K

 與 original 差別: 讓 K 從 2 跑到 5, 計算每個 K 的 BIC 與 AIC, 
 並一併看每個 K 的表現，再決定最佳 K。

   BIC = -2 * logL + p * ln(N)     p = 模型參數量 N = 樣本數
   AIC = -2 * logL + 2 * p
   兩者都是越小越好
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 15
K_RANGE = range(2, 6)        # 嘗試 K = 2, 3, 4, 5


class GMM:
    def __init__(self, n_clusters, max_iter=300, tol=1e-5,
                 reg_covar=1e-6, n_init=10, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar     
        self.n_init = n_init            
        self.random_state = random_state

    # 多維高斯分布的 log 機率密度
    def _gaussian_log_pdf(self, samples, mean, cov):
        n_features = samples.shape[1]
        centered = samples - mean
        inv_cov = np.linalg.inv(cov)
        _, log_det = np.linalg.slogdet(cov)
        mahalanobis_sq = np.sum((centered @ inv_cov) * centered, axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis_sq)

    @staticmethod
    def _log_sum_exp(arr, axis):
        max_val = np.max(arr, axis=axis, keepdims=True)
        return max_val.squeeze(axis) + np.log(np.sum(np.exp(arr - max_val), axis=axis))

    def _init_params(self, samples, rng):
        n_samples, n_features = samples.shape
        center_idx = rng.choice(n_samples, self.n_clusters, replace=False)
        means = samples[center_idx].copy()
        weights = np.full(self.n_clusters, 1.0 / self.n_clusters)
        base_cov = np.cov(samples, rowvar=False) + self.reg_covar * np.eye(n_features)
        covs = np.array([base_cov.copy() for _ in range(self.n_clusters)])
        return weights, means, covs

    def _e_step(self, samples, weights, means, covs):
        n_samples = samples.shape[0]
        log_weighted = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            log_weighted[:, k] = np.log(weights[k] + 1e-12) + \
                self._gaussian_log_pdf(samples, means[k], covs[k])
        log_total = self._log_sum_exp(log_weighted, axis=1)
        responsibility = np.exp(log_weighted - log_total[:, None])
        return responsibility, log_total.sum()

    def _m_step(self, samples, responsibility):
        n_samples, n_features = samples.shape
        cluster_size = responsibility.sum(axis=0) + 1e-12
        weights = cluster_size / n_samples
        means = (responsibility.T @ samples) / cluster_size[:, None]
        covs = np.zeros((self.n_clusters, n_features, n_features))
        for k in range(self.n_clusters):
            centered = samples - means[k]
            covs[k] = (responsibility[:, [k]] * centered).T @ centered / cluster_size[k]
            covs[k] += self.reg_covar * np.eye(n_features)
        return weights, means, covs

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

    def fit(self, samples):
        seed_maker = np.random.default_rng(self.random_state)
        best_log_likelihood = -np.inf
        best_params = None
        for _ in range(self.n_init):
            rng = np.random.default_rng(seed_maker.integers(1 << 31))
            try:
                weights, means, covs, log_likelihood = self._fit_once(samples, rng)
            except np.linalg.LinAlgError:
                continue
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

    def count_parameters(self, n_features):
        cov_params = self.n_clusters * n_features * (n_features + 1) / 2
        mean_params = self.n_clusters * n_features
        weight_params = self.n_clusters - 1
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


def best_pool_accuracy(true_labels, clusters, n_clusters, new_names):
    best = 0.0
    for assignment in itertools.product(new_names, repeat=n_clusters):
        predicted = np.array([assignment[c] for c in clusters])
        best = max(best, (predicted == true_labels).mean())
    return best


def main():
    data = pd.read_csv('rf_classification_result.csv')
    unknown_mask = (data['pred_label'] == 'Unlabeled').values
    unknown = data[unknown_mask].copy().reset_index(drop=True)
    non_features = ('Class', 'confidence_score', 'pred_label')
    feature_cols = [col for col in data.columns if col not in non_features]
    samples = StandardScaler().fit_transform(unknown[feature_cols].values)

    # 分析每個 K 的表現
    unknown['True_Class'] = pd.read_csv('dry_bean_test.csv')['Class'].values[unknown_mask]
    print(f'未知(Unlabeled)樣本數: {len(samples)}')

    # K = 2..5，記錄 BIC / AIC
    new_names = ['BARBUNYA', 'BOMBAY']          # 測試集新增的 2 類
    true_labels = unknown['True_Class'].values
    records = []
    models = {}
    print('\n=== 動態 K 值評估 ===')
    print(' K     logL        BIC          AIC     未知池最佳對齊準確率')
    for n_clusters in K_RANGE:
        gmm = GMM(n_clusters=n_clusters, random_state=RANDOM_STATE)
        gmm.fit(samples)
        clusters = gmm.predict(samples)
        pool_accuracy = best_pool_accuracy(true_labels, clusters, n_clusters, new_names)
        bic, aic, log_likelihood = gmm.bic(samples), gmm.aic(samples), gmm.log_likelihood_
        models[n_clusters] = gmm
        records.append((n_clusters, log_likelihood, bic, aic, pool_accuracy))
        print(f' {n_clusters}   {log_likelihood:9.1f}   {bic:10.1f}   {aic:10.1f}'
              f'        {pool_accuracy:6.3f}')

    summary = pd.DataFrame(records, columns=['K', 'logL', 'BIC', 'AIC', 'PoolAcc'])
    bic_best_k = int(summary.loc[summary['BIC'].idxmin(), 'K'])
    
    selected_k = 2 # 作業規範說會多兩類新類別

    # 繪製 BIC / AIC 與 任務準確率 對 K 的關係
    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_left.plot(summary['K'], summary['BIC'], 'o-', color='tab:blue', label='BIC')
    ax_left.plot(summary['K'], summary['AIC'], 's--', color='tab:cyan', label='AIC')
    ax_left.set_xlabel('Number of components K')
    ax_left.set_ylabel('Information criterion (lower is better)')
    ax_left.set_xticks(list(K_RANGE))
    ax_right = ax_left.twinx()
    ax_right.plot(summary['K'], summary['PoolAcc'], '^-', color='tab:red',
                  label='Unknown-pool accuracy')
    ax_right.set_ylabel('Unknown-pool mapping accuracy')
    ax_right.set_ylim(0, 1)
    ax_left.axvline(selected_k, color='green', ls=':', label=f'selected K = {selected_k}')
    all_lines = ax_left.get_lines() + ax_right.get_lines()
    ax_left.legend(all_lines, [line.get_label() for line in all_lines], loc='center right')
    plt.title('GMM Model Selection: BIC/AIC vs. Task Accuracy')
    plt.tight_layout()
    plt.savefig('gmm_bic_aic.png', dpi=120)
    print('已輸出圖檔: gmm_bic_aic.png')

    # 用選定的 K 做最終分群, 給中性代號
    gmm = models[selected_k]
    clusters = gmm.predict(samples)
    unknown['Cluster'] = clusters
    cluster_to_label = {k: f'Unknown_Type_{chr(ord("A") + k)}' for k in range(selected_k)}
    unknown['Cluster_Label'] = unknown['Cluster'].map(cluster_to_label)
    print('\n各群大小與原始 Area 平均（僅供描述）:')
    for k in range(selected_k):
        size = int((clusters == k).sum())
        mean_area = unknown.loc[unknown['Cluster'] == k, 'Area'].mean()
        print(f'  {cluster_to_label[k]}: {size} 筆, Area 平均 {mean_area:.0f}')

    output_cols = ['Area', 'Perimeter', 'confidence_score', 'pred_label',
                   'Cluster', 'Cluster_Label']
    unknown[output_cols].to_csv('gmm_dynamicK_result.csv',
                                index=False, encoding='utf-8-sig')
    summary.to_csv('gmm_bic_aic_table.csv', index=False, encoding='utf-8-sig')
    print('已輸出: gmm_dynamicK_result.csv, gmm_bic_aic_table.csv')


if __name__ == '__main__':
    main()
