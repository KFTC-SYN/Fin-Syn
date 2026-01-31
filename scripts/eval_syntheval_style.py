"""
SynthEval 스타일의 합성 데이터 평가 스크립트
syntheval 라이브러리의 핵심 로직을 직접 구현
"""
import numpy as np
import pandas as pd
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from scipy.stats import sem, chi2_contingency, ks_2samp, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from collections import Counter
from typing import Literal
from scipy.spatial.distance import cdist
import lib
from lib import read_pure_data, read_changed_val


# ============================================================================
# 유틸리티 함수들 (syntheval에서 가져온 로직)
# ============================================================================

def get_cat_variables(df, threshold=10):
    """범주형 변수 자동 감지"""
    cat_variables = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_variables.append(col)
        elif (
            np.issubdtype(df[col].dtype, np.integer) or 
            np.issubdtype(df[col].dtype, np.floating)
        ) and df[col].nunique() < threshold:
            cat_variables.append(col)
    return cat_variables


class ConsistentLabelEncoding:
    """일관된 레이블 인코딩 클래스"""
    def __init__(self, real, fake, categorical_columns, numerical_columns, hout=None):
        assert (
            len(categorical_columns) > 0 or len(numerical_columns) > 0
        ), "Either categorical or numerical columns must be provided."

        joint_dataframe = pd.concat((real.reset_index(), fake.reset_index()), axis=0)
        if hout is not None:
            joint_dataframe = pd.concat(
                (joint_dataframe.reset_index(), hout.reset_index()), axis=0
            )

        if len(categorical_columns) > 0:
            self.encoder = OrdinalEncoder().fit(joint_dataframe[categorical_columns])
            self.cat_cols = categorical_columns
        else:
            self.cat_cols = None

        if len(numerical_columns) > 0:
            self.num_encoder = MinMaxScaler().fit(joint_dataframe[numerical_columns])
            self.num_cols = numerical_columns
        else:
            self.num_cols = None

    def encode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.transform(data[self.cat_cols]).astype("int")
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.transform(data[self.num_cols])
        return data

    def decode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.inverse_transform(data[self.cat_cols])
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.inverse_transform(data[self.num_cols])
        return data


def stack_dataframes(real, fake):
    """real과 fake 데이터프레임을 스택하고 구분 컬럼 추가"""
    real = pd.concat(
        (real.reset_index(), pd.DataFrame(np.ones(len(real)), columns=["real"])), axis=1
    )
    fake = pd.concat(
        (fake.reset_index(), pd.DataFrame(np.zeros(len(fake)), columns=["real"])),
        axis=1,
    )
    return pd.concat((real, fake), ignore_index=True)


# ============================================================================
# Gower Distance 구현 (NN 메트릭에 필요)
# ============================================================================

def _create_matrix_with_ones(indices, num_rows):
    """NN 인덱스를 위한 행렬 생성"""
    matrix = np.zeros((len(indices), num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i, index] = 1
    return matrix


def _gower_matrix_sklearn(data_x, data_y=None, cat_features: list = None, weights=None, 
                          num_attribute_ranges=None, nums_metric: Literal['L1', 'EXP_L2'] = 'L1'):
    """Gower distance 행렬 계산 (syntheval에서 가져온 로직)"""
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    out_shape = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    if cat_features is None:
        cat_features = np.zeros(x_n_cols, dtype=bool)
        for col in range(x_n_cols):
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    if weights is None:
        weights = np.ones(X.shape[1])

    weights_cat = weights[cat_features]
    weights_num = weights[np.logical_not(cat_features)]

    Z = np.concatenate((X, Y))
    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)]
    Z_cat = Z[:, cat_features]

    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(
            np.stack((np.array(np.ptp(Z_num, axis=0), dtype=np.float64), np.ones(len(weights_num)))),
            axis=0
        )

    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]

    if not np.array_equal(cat_features, np.ones(X.shape[1])):
        if nums_metric == 'L1':
            nums_sum = cdist(
                X_num.astype(float), Y_num.astype(float), 'minkowski', p=1,
                w=(weights_num / num_attribute_ranges)
            )
        elif nums_metric == 'EXP_L2':
            nums_sum = cdist(
                X_num.astype(float), Y_num.astype(float), 'minkowski', p=2,
                w=(weights_num / num_attribute_ranges**2)
            )
        else:
            raise NotImplementedError("The keyword literal is not valid!")
    else:
        nums_sum = out_shape

    if not np.array_equal(cat_features, np.zeros(X.shape[1])):
        Z_cat_enc = OrdinalEncoder().fit_transform(Z_cat)
        X_cat = Z_cat_enc[x_index,]
        Y_cat = Z_cat_enc[y_index,]
        cat_sum = cdist(X_cat.astype(int), Y_cat.astype(int), 'hamming', w=weights_cat) * len(weights_cat)
    else:
        cat_sum = out_shape

    return (nums_sum + cat_sum) / weights.sum()


# def _knn_distance(a, b, cat_cols, num, metric: Literal['gower', 'euclid'] = 'gower', weights=None):
#     """K-nearest neighbors 거리 계산 (weights 파라미터 지원)"""
#     def gower_knn(a, b, bool_cat_cols, gower_variant):
#         d = []
#         if np.array_equal(a, b):
#             matrix = _gower_matrix_sklearn(a, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant) + np.eye(len(a))
#             for _ in range(num):
#                 d.append(matrix.min(axis=1))
#                 matrix += _create_matrix_with_ones(matrix.argmin(axis=1, keepdims=True), len(a))
#         else:
#             matrix = _gower_matrix_sklearn(a, b, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)
#             for _ in range(num):
#                 d.append(matrix.min(axis=1))
#                 matrix += _create_matrix_with_ones(matrix.argmin(axis=1, keepdims=True), len(b))
#         return d

#     def euclidean_knn(a, b):
#         d = []
#         nn = NearestNeighbors(n_neighbors=num + 1, metric_params={'w': weights})
#         if np.array_equal(a, b):
#             nn.fit(a)
#             dists, _ = nn.kneighbors(a)
#             for i in range(num):
#                 d.append(dists[:, 1 + i])
#         else:
#             nn.fit(b)
#             dists, _ = nn.kneighbors(a)
#             for i in range(num):
#                 d.append(dists[:, i])
#         return d

#     if metric == 'gower':
#         bool_cat_cols = [col1 in cat_cols for col1 in a.columns]
#         num_cols = [col2 for col2 in a.columns if col2 not in cat_cols]
#         a[num_cols] = a[num_cols].astype("float")
#         b[num_cols] = b[num_cols].astype("float")
#         return gower_knn(a, b, bool_cat_cols, gower_variant='L1')
#     elif metric == 'euclid':
#         return euclidean_knn(a, b)
#     else:
#         raise Exception("Unknown metric; options are 'gower' or 'euclid'")
from typing import Literal
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _knn_distance(
    a,
    b,
    cat_cols,
    num,
    metric: Literal['gower', 'euclid'] = 'gower',
    weights=None,
    verbose: bool = True,
    log_every: int = 1000,
):
    """
    K-nearest neighbors distance computation.

    Returns
    -------
    list of np.ndarray
        Length = num, each array has shape (n_samples,)
    """

    def gower_knn(a, b, bool_cat_cols, gower_variant):
        d = []

        if verbose:
            print(f"[kNN:gower] computing distance matrix "
                  f"({len(a)} x {len(b)})")

        if np.array_equal(a, b):
            matrix = (
                _gower_matrix_sklearn(
                    a,
                    cat_features=bool_cat_cols,
                    weights=weights,
                    nums_metric=gower_variant
                ) + np.eye(len(a))
            )
        else:
            matrix = _gower_matrix_sklearn(
                a,
                b,
                cat_features=bool_cat_cols,
                weights=weights,
                nums_metric=gower_variant
            )

        row_idx = np.arange(matrix.shape[0])

        for i in range(num):
            min_vals = matrix.min(axis=1)
            min_pos = matrix.argmin(axis=1)

            d.append(min_vals)
            matrix[row_idx, min_pos] += 1.0  # exclude selected neighbor

            if verbose and ((i + 1) % log_every == 0 or i + 1 == num):
                print(f"[kNN:gower] neighbour {i + 1}/{num} selected")

        return d

    def euclidean_knn(a, b):
        d = []
        same = np.array_equal(a, b)
        n_neighbors = num + 1 if same else num

        if verbose:
            print(f"[kNN:euclid] fitting NearestNeighbors "
                  f"(n_neighbors={n_neighbors})")

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric_params={'w': weights} if weights is not None else None
        )

        if same:
            nn.fit(a)
            dists, _ = nn.kneighbors(a)
            for i in range(num):
                d.append(dists[:, 1 + i])
        else:
            nn.fit(b)
            dists, _ = nn.kneighbors(a)
            for i in range(num):
                d.append(dists[:, i])

        if verbose:
            print("[kNN:euclid] done")

        return d

    # ---- metric dispatch ----
    if metric == 'gower':
        if verbose:
            print("[kNN] metric = gower")

        cat_set = set(cat_cols)
        bool_cat_cols = [col in cat_set for col in a.columns]
        num_cols = [col for col in a.columns if col not in cat_set]

        a[num_cols] = a[num_cols].astype(float)
        b[num_cols] = b[num_cols].astype(float)

        return gower_knn(a, b, bool_cat_cols, gower_variant='L1')

    elif metric == 'euclid':
        if verbose:
            print("[kNN] metric = euclid")
        return euclidean_knn(a, b)

    else:
        raise ValueError("metric must be 'gower' or 'euclid'")



# ============================================================================
# 메트릭 베이스 클래스
# ============================================================================

class MetricBase(ABC):
    """메트릭 베이스 클래스"""
    def __init__(
        self,
        real_data: pd.DataFrame,
        synt_data: pd.DataFrame,
        hout_data: Optional[pd.DataFrame] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        do_preprocessing: bool = True,
        verbose: bool = True
    ):
        if do_preprocessing:
            if cat_cols is None:
                cat_cols = get_cat_variables(real_data, threshold=10)
                num_cols = [col for col in real_data.columns if col not in cat_cols]
            
            CLE = ConsistentLabelEncoding(real_data, synt_data, cat_cols, num_cols, hout_data)
            real_data = CLE.encode(real_data)
            synt_data = CLE.encode(synt_data)
            if hout_data is not None:
                hout_data = CLE.encode(hout_data)
            self.encoder = CLE
        else:
            self.encoder = None

        self.real_data = real_data
        self.synt_data = synt_data
        self.hout_data = hout_data
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.results = {}
        self.verbose = verbose

    @staticmethod
    @abstractmethod
    def name() -> str:
        """메트릭 이름"""
        pass

    @staticmethod
    @abstractmethod
    def type() -> str:
        """메트릭 타입: 'utility', 'privacy', 'fairness'"""
        pass

    @abstractmethod
    def evaluate(self) -> Union[float, dict]:
        """메트릭 평가"""
        pass

    def format_output(self) -> Optional[str]:
        """출력 포맷팅"""
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        """정규화된 출력"""
        return None


# ============================================================================
# 유틸리티 메트릭 구현
# ============================================================================

class DimensionWiseMeans(MetricBase):
    """Dimension-Wise Means 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'dwm'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self) -> dict:
        if len(self.num_cols) == 0:
            if self.verbose:
                print("Warning: No numerical attributes provided for dimensionwise means metric.")
            return {}
        
        real_data = self.real_data[self.num_cols]
        synt_data = self.synt_data[self.num_cols]

        dim_means = np.array([np.mean(real_data, axis=0), np.mean(synt_data, axis=0)]).T
        means_diff = dim_means[:, 0] - dim_means[:, 1]
        
        mean_errors = np.array([sem(real_data), sem(synt_data)]).T
        diff_error = np.sqrt(np.sum(mean_errors**2, axis=1))

        self.results = {
            'avg': np.mean(abs(means_diff)), 
            'err': np.sqrt(sum(diff_error**2)) / len(diff_error)
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average dimensionwise means diff. (nums) :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_dwm_diff', 
                'dim': 'u', 
                'val': self.results['avg'], 
                'err': self.results['err'], 
                'n_val': 1 - self.results['avg'], 
                'n_err': self.results['err']
            }]
        return None


class PropensityMSE(MetricBase):
    """Propensity Mean Squared Error 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'p_mse'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, k_folds=5, max_iter=100, solver='liblinear') -> dict:
        discriminator = LogisticRegression(max_iter=max_iter, solver=solver, random_state=42)
        Df = stack_dataframes(self.real_data, self.synt_data).drop(['index'], axis=1)

        if len(self.num_cols) > 0:
            Df[self.num_cols] = StandardScaler().fit_transform(Df[self.num_cols])
        Xs, ys = Df.drop(['real'], axis=1), Df['real']

        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        res, acc = [], []
        for train_index, test_index in kf.split(Xs, ys):
            x_train = Xs.iloc[train_index]
            x_test = Xs.iloc[test_index]
            y_train = ys.iloc[train_index]
            y_test = ys.iloc[test_index]

            mod = discriminator.fit(x_train, y_train)
            pred = mod.predict_proba(x_test)

            num_synths = len(y_test) - np.count_nonzero(y_test)
            res.append(np.mean((pred[:, 0] - num_synths / len(y_test))**2))
            acc.append(f1_score(y_test, mod.predict(x_test), average='macro'))

        self.results = {
            'avg pMSE': np.mean(res), 
            'pMSE err': np.std(res, ddof=1) / np.sqrt(len(res)),
            'avg acc': np.mean(acc), 
            'acc err': np.std(acc, ddof=1) / np.sqrt(len(acc))
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"""| Propensity mean squared error (pMSE)     :   {self.results['avg pMSE']:.4f}  {self.results['pMSE err']:.4f}   |
|   -> average pMSE classifier accuracy    :   {self.results['avg acc']:.4f}  {self.results['acc err']:.4f}   |"""
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_pMSE', 
                'dim': 'u', 
                'val': self.results['avg pMSE'], 
                'err': self.results['pMSE err'], 
                'n_val': 1 - 4 * self.results['avg pMSE'], 
                'n_err': 4 * self.results['pMSE err']
            }]
        return None


def _cramers_V(var1, var2):
    """Cramer's V 계산"""
    try:
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
        stat = chi2_contingency(crosstab)[0]
        obs = np.sum(crosstab)
        mini = min(crosstab.shape) - 1
        return np.sqrt(stat / (obs * mini + 1e-16))
    except (ValueError, MemoryError) as e:
        # crosstab이 너무 크거나 메모리 부족 시 기본값 반환
        # ValueError: "Unstacked DataFrame is too big, causing int32 overflow"
        return 0.0


def _correlation_ratio(categories, measurements):
    """Correlation ratio (eta^2) 계산"""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)


def _apply_mat(data, func, labs1, labs2):
    """행렬 생성 헬퍼 함수"""
    res = []
    for lab1 in labs1:
        row = []
        for lab2 in labs2:
            try:
                value = func(data[lab1], data[lab2])
                row.append(value)
            except (ValueError, MemoryError) as e:
                # crosstab 오류 등 발생 시 기본값 사용
                row.append(0.0)
        res.append(row)
    return pd.DataFrame(
        np.array(res, dtype=float), 
        columns=labs2, 
        index=labs1
    )


def mixed_correlation(data, num_cols, cat_cols):
    """혼합 상관관계 행렬 계산"""
    if len(num_cols) > 0:
        corr_num_num = data[num_cols].corr()
    else:
        corr_num_num = pd.DataFrame()
    
    if len(cat_cols) > 0:
        corr_cat_cat = _apply_mat(data, _cramers_V, cat_cols, cat_cols)
        corr_cat_num = _apply_mat(data, _correlation_ratio, cat_cols, num_cols)
    else:
        corr_cat_cat = pd.DataFrame()
        corr_cat_num = pd.DataFrame()
    
    if corr_cat_cat.empty:
        corr = corr_num_num
    elif corr_num_num.empty:
        corr = corr_cat_cat
    else:
        top_row = pd.concat([corr_cat_cat, corr_cat_num], axis=1)
        bot_row = pd.concat([corr_cat_num.transpose(), corr_num_num], axis=1)
        corr = pd.concat([top_row, bot_row], axis=0)
    
    return corr + np.diag(1 - np.diag(corr))


class MixedCorrelation(MetricBase):
    """Mixed Correlation 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'corr_diff'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, mixed_corr=True) -> dict:
        if mixed_corr:
            r_corr = mixed_correlation(self.real_data, self.num_cols, self.cat_cols)
            f_corr = mixed_correlation(self.synt_data, self.num_cols, self.cat_cols)
        else:
            r_corr = self.real_data[self.num_cols].corr()
            f_corr = self.synt_data[self.num_cols].corr()
        
        corr_mat = r_corr - f_corr
        self.results = {
            'corr_mat_diff': np.linalg.norm(corr_mat, ord='fro'), 
            'corr_mat_dims': len(corr_mat)
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            if self.num_cols and self.cat_cols:
                return f"| Mixed correlation matrix difference      :   {self.results['corr_mat_diff']:.4f}           |"
            else:
                return f"| Correlation difference (nums only)       :   {self.results['corr_mat_diff']:.4f}           |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            n_elements = int(self.results['corr_mat_dims'] * (self.results['corr_mat_dims'] - 1) / 2)
            return [{
                'metric': 'corr_mat_diff', 
                'dim': 'u', 
                'val': self.results['corr_mat_diff'], 
                'n_val': 1 - self.results['corr_mat_diff'] / max(n_elements, 1)
            }]
        return None


class KolmogorovSmirnovTest(MetricBase):
    """Kolmogorov-Smirnov Test 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'ks_test'

    @staticmethod
    def type() -> str:
        return 'utility'

    def _total_variation_distance(self, x, y):
        """Total Variation Distance 계산"""
        X, Y = Counter(x), Counter(y)
        merged = X + Y
        return np.round(0.5 * sum([abs(X[key]/len(x) - Y[key]/len(y)) for key in merged.keys()]), 4)

    def _discrete_ks(self, x, y, n_perms=1000):
        """이산값에 대한 permutation test"""
        # scipy < 1.8.0에서는 permutation_test가 없으므로 간단한 TVD만 반환
        tvd = self._total_variation_distance(x, y)
        return tvd, 0.5  # 기본 p-value

    def evaluate(self, sig_lvl=0.05, n_perms=1000) -> dict:
        n_dists, c_dists = [], []
        pvals = []
        sig_cols = []
        self.sig_lvl = sig_lvl

        for category in self.real_data.columns:
            R = self.real_data[category]
            F = self.synt_data[category]

            if category in self.cat_cols:
                statistic, pvalue = self._discrete_ks(F, R, n_perms)
                c_dists.append(statistic)
                pvals.append(pvalue)
            else:
                KstestResult = ks_2samp(R, F)
                statistic, pvalue = KstestResult.statistic, KstestResult.pvalue
                n_dists.append(statistic)
                pvals.append(pvalue)
            
            if pvalue < sig_lvl:
                sig_cols.append(category)

        if n_dists == []:
            avg_ks = np.nan
            err_ks = np.nan
        else:
            avg_ks = np.mean(n_dists)
            err_ks = np.std(n_dists, ddof=1) / np.sqrt(len(n_dists))

        if c_dists == []:
            avg_tvd = np.nan
            err_tvd = np.nan
        else:
            avg_tvd = np.mean(c_dists)
            err_tvd = np.std(c_dists, ddof=1) / np.sqrt(len(c_dists))

        self.results = {
            'avg stat': np.mean(n_dists + c_dists),
            'stat err': np.std(n_dists + c_dists, ddof=1) / np.sqrt(len(n_dists + c_dists)),
            'avg ks': avg_ks,
            'ks err': err_ks,
            'avg tvd': avg_tvd,
            'tvd err': err_tvd,
            'avg pval': np.mean(pvals),
            'pval err': np.std(pvals, ddof=1) / np.sqrt(len(pvals)),
            'num sigs': len(sig_cols),
            'frac sigs': len(sig_cols) / len(pvals) if len(pvals) > 0 else 0,
            'sigs cols': sig_cols
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            R = self.results
            return f"""| Kolmogorov–Smirnov / Total Variation Distance test            |
|   -> average combined statistic          :   {R['avg stat']:.4f}  {R['stat err']:.4f}   |
|       -> avg. Kolmogorov–Smirnov dist.   :   {R['avg ks']:.4f}  {R['ks err']:.4f}   |
|       -> avg. Total Variation Distance   :   {R['avg tvd']:.4f}  {R['tvd err']:.4f}   |
|   -> average combined p-value            :   {R['avg pval']:.4f}  {R['pval err']:.4f}   |
|       -> # significant tests at a={self.sig_lvl:.2f}   :   {R['num sigs']:2d}               |
|       -> fraction of significant tests   :   {R['frac sigs']:.4f}           |"""
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            R = self.results
            return [
                {
                    'metric': 'ks_tvd_stat', 'dim': 'u',
                    'val': R['avg stat'],
                    'err': R['stat err'],
                    'n_val': 1 - R['avg stat'],
                    'n_err': R['stat err']
                },
                {
                    'metric': 'frac_ks_sigs', 'dim': 'u',
                    'val': R['frac sigs'],
                    'n_val': 1 - R['frac sigs']
                }
            ]
        return None


class HellingerDistance(MetricBase):
    """Hellinger Distance 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'h_dist'

    @staticmethod
    def type() -> str:
        return 'utility'

    def _scott_ref_rule(self, set1, set2):
        """Scott reference rule로 bin 개수 계산"""
        samples = np.concatenate((set1, set2))
        std = np.std(samples)
        n = len(samples)
        bin_width = np.ceil(
            n**(1/3) * std / (3.5 * (np.percentile(samples, 75) - np.percentile(samples, 25)))
        ).astype(int)
        min_edge = min(samples)
        max_edge = max(samples)
        N = min(abs(int((max_edge - min_edge) / bin_width)), 10000)
        Nplus1 = N + 1
        return np.linspace(min_edge, max_edge, Nplus1)

    def _hellinger(self, p, q):
        """Hellinger distance 계산"""
        sqrt_pdf1 = np.sqrt(p)
        sqrt_pdf2 = np.sqrt(q)
        diff = sqrt_pdf1 - sqrt_pdf2
        return 1 / np.sqrt(2) * np.linalg.norm(diff)

    def evaluate(self) -> dict:
        H_dist = []

        for category in self.cat_cols:
            class_num = len(np.unique(self.real_data[category]))
            pdfR = np.histogram(self.real_data[category], bins=class_num)[0]
            pdfF = np.histogram(self.synt_data[category], bins=class_num)[0]
            H_dist.append(self._hellinger(pdfR / sum(pdfR), pdfF / sum(pdfF)))

        for category in self.num_cols:
            n_bins = self._scott_ref_rule(self.real_data[category], self.synt_data[category])
            pdfR = np.histogram(self.real_data[category], bins=n_bins)[0]
            pdfF = np.histogram(self.synt_data[category], bins=n_bins)[0]
            H_dist.append(self._hellinger(pdfR / sum(pdfR), pdfF / sum(pdfF)))

        self.results = {
            'avg': np.mean(H_dist),
            'err': np.std(H_dist, ddof=1) / np.sqrt(len(H_dist))
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average empirical Hellinger distance     :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_h_dist', 'dim': 'u',
                'val': self.results['avg'],
                'err': self.results['err'],
                'n_val': 1 - self.results['avg'],
                'n_err': self.results['err']
            }]
        return None


class ConfidenceIntervalOverlap(MetricBase):
    """Confidence Interval Overlap 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'cio'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, confidence=95) -> dict:
        confidence_table = {80: 1.28, 90: 1.645, 95: 1.96, 98: 2.33, 99: 2.58}
        
        if len(self.num_cols) == 0:
            if self.verbose:
                print("Warning: No numerical attributes provided for confidence interval overlap metric.")
            return {}
        
        if confidence not in confidence_table.keys():
            if self.verbose:
                print("Error: Confidence level not recognized, choose 80, 90, 95, 98 or 99.")
            return {}
        
        self.confidence = confidence
        z_value = confidence_table[confidence]
        
        mus = np.array([
            np.mean(self.real_data[self.num_cols], axis=0),
            np.mean(self.synt_data[self.num_cols], axis=0)
        ]).T
        sems = np.array([
            sem(self.real_data[self.num_cols]),
            sem(self.synt_data[self.num_cols])
        ]).T

        CI = sems * z_value
        us = mus + CI
        ls = mus - CI

        Jk = []
        for i in range(len(CI)):
            top = (min(us[i][0], us[i][1]) - max(ls[i][0], ls[i][1]))
            Jk.append(max(0, 0.5 * (top / (us[i][0] - ls[i][0]) + top / (us[i][1] - ls[i][1]))))

        num = sum([j == 0 for j in Jk])
        frac = num / len(Jk) if len(Jk) > 0 else 0
        
        self.results = {
            'avg overlap': np.mean(Jk),
            'overlap err': np.std(Jk, ddof=1) / np.sqrt(len(Jk)) if len(Jk) > 0 else 0,
            'num non-overlaps': num,
            'frac non-overlaps': frac
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"""| Average confidence interval overlap      :   {self.results['avg overlap']:.4f}  {self.results['overlap err']:.4f}   |
|   -> # non-overlapping COIs at {self.confidence:2d}%%       :   {self.results['num non-overlaps']:2d}               |
|   -> fraction of non-overlapping CIs     :   {self.results['frac non-overlaps']:.4f}           |"""
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_cio', 'dim': 'u',
                'val': self.results['avg overlap'],
                'err': self.results['overlap err'],
                'n_val': self.results['avg overlap'],
                'n_err': self.results['overlap err']
            }]
        return None


class MutualInformation(MetricBase):
    """Mutual Information 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'mi_diff'

    @staticmethod
    def type() -> str:
        return 'utility'

    def _pairwise_attributes_mutual_information(self, data):
        """모든 변수 쌍에 대한 normalized mutual information 계산"""
        labs = sorted(data.columns)
        res = (
            normalized_mutual_info_score(
                data[cat1].astype(str), data[cat2].astype(str),
                average_method='arithmetic'
            )
            for cat1 in labs for cat2 in labs
        )
        return pd.DataFrame(
            np.fromiter(res, dtype=float).reshape(len(labs), len(labs)),
            columns=labs, index=labs
        )

    def evaluate(self) -> dict:
        r_mi = self._pairwise_attributes_mutual_information(self.real_data)
        f_mi = self._pairwise_attributes_mutual_information(self.synt_data)
        mi_mat = r_mi - f_mi

        self.results = {
            'mutual_inf_diff': np.linalg.norm(mi_mat, ord='fro'),
            'mi_mat_dims': len(mi_mat)
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Pairwise mutual information difference   :   {self.results['mutual_inf_diff']:.4f}           |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            n_elements = int(self.results['mi_mat_dims'] * (self.results['mi_mat_dims'] - 1) / 2)
            return [{
                'metric': 'mutual_inf_diff', 'dim': 'u',
                'val': self.results['mutual_inf_diff'],
                'n_val': 1 - self.results['mutual_inf_diff'] / max(n_elements, 1)
            }]
        return None


class QuantileMSE(MetricBase):
    """Quantile Mean Squared Error 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'q_mse'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, num_quants=10, cat_mse=False) -> dict:
        if len(self.num_cols) == 0 and not cat_mse:
            if self.verbose:
                print('Error: Quantile mse did not run, no numerical attributes, or cat_mse not enabled!')
            return {}

        qMSE_lst = []
        for category in self.real_data.columns:
            if category in self.cat_cols and cat_mse:
                # 범주형 데이터
                real_items = self.real_data[category].unique()
                synth_frac = np.array([
                    np.sum(self.synt_data[category] == item) for item in real_items
                ]) / len(self.synt_data)
                real_frac = np.array([
                    np.sum(self.real_data[category] == item) for item in real_items
                ]) / len(self.real_data)
                qMSE_lst.append(np.mean((synth_frac - real_frac)**2))
            elif category not in self.cat_cols:
                # 수치형 데이터
                quantiles = np.quantile(self.real_data[category], np.linspace(0, 1, num_quants + 1))
                bin_edges = quantiles.tolist()
                synth_hist, _ = np.histogram(self.synt_data[category], bins=bin_edges)
                synth_frac = synth_hist / len(self.synt_data)
                qMSE_lst.append(np.mean((synth_frac - 1 / num_quants)**2))

        self.results = {
            'avg qMSE': np.mean(qMSE_lst),
            'qMSE err': np.std(qMSE_lst, ddof=1) / np.sqrt(len(qMSE_lst)) if len(qMSE_lst) > 0 else 0
        }
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Quantile mean squared error (qMSE)       :   {self.results['avg qMSE']:.4f}  {self.results['qMSE err']:.4f}   |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_qMSE', 'dim': 'u',
                'val': self.results['avg qMSE'],
                'err': self.results['qMSE err'],
                'n_val': 1 - self.results['avg qMSE'],
                'n_err': self.results['qMSE err']
            }]
        return None


class NearestNeighbourDistanceRatio(MetricBase):
    """Nearest Neighbour Distance Ratio 메트릭 (Privacy)"""
    
    @staticmethod
    def name() -> str:
        return 'nndr'

    @staticmethod
    def type() -> str:
        return 'privacy'

    # def evaluate(self, nn_dist='gower') -> dict:
    #     try:
    #         dist = _knn_distance(self.real_data, self.synt_data, self.cat_cols, 2, nn_dist)
    #         dr = list(map(lambda x: x[0] / (x[1] + 1e-16), np.transpose(dist)))

    #         self.results = {
    #             'avg': np.mean(dr),
    #             'err': np.std(dr, ddof=1) / np.sqrt(len(dr)) if len(dr) > 0 else 0
    #         }

    #         if self.hout_data is not None:
    #             dist_h = _knn_distance(self.hout_data, self.synt_data, self.cat_cols, 2, nn_dist)
    #             dr_h = list(map(lambda x: x[0] / (x[1] + 1e-16), np.transpose(dist_h)))
    #             diff = np.mean(dr_h) - self.results['avg']
    #             err_diff = np.sqrt(
    #                 (np.std(dr_h, ddof=1) / np.sqrt(len(dr_h)))**2 + self.results['err']**2
    #             )
    #             self.results['priv_loss'] = diff
    #             self.results['priv_loss_err'] = err_diff
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Warning: NNDR calculation failed: {e}")
    #         self.results = {}
        
    #     return self.results

    def evaluate(self, nn_dist: str = 'gower') -> dict:
        try:
            t0 = time.time()
            print("[NNDR] Start evaluation")

            print("[NNDR] Computing kNN distance (real → synt)...")
            t1 = time.time()
            dist = _knn_distance(
                self.real_data, self.synt_data, self.cat_cols, 2, nn_dist
            )
            # dist는 [dist_1nn, dist_2nn] 형태의 리스트
            dist_1nn = np.asarray(dist[0])
            dist_2nn = np.asarray(dist[1])
            print(f"[NNDR] kNN(real) done in {time.time() - t1:.2f}s, n_samples={len(dist_1nn)}")

            print("[NNDR] Computing distance ratios (real)...")
            dr = dist_1nn / (dist_2nn + 1e-16)
            print(f"[NNDR] dr stats: mean={dr.mean():.4f}, std={dr.std():.4f}")

            # mean and standard error of mean
            n = len(dr)
            avg = float(np.mean(dr))
            err = float(np.std(dr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            self.results = {'avg': avg, 'err': err}

            if self.hout_data is not None:
                print("[NNDR] Computing kNN distance (holdout → synt)...")
                t2 = time.time()
                dist_h = _knn_distance(
                    self.hout_data, self.synt_data, self.cat_cols, 2, nn_dist
                )
                dist_h_1nn = np.asarray(dist_h[0])
                dist_h_2nn = np.asarray(dist_h[1])
                print(f"[NNDR] kNN(holdout) done in {time.time() - t2:.2f}s, n_samples={len(dist_h_1nn)}")

                dr_h = dist_h_1nn / (dist_h_2nn + 1e-16)
                n_h = len(dr_h)
                avg_h = float(np.mean(dr_h))
                err_h = float(np.std(dr_h, ddof=1) / np.sqrt(n_h)) if n_h > 1 else 0.0

                diff = avg_h - avg
                err_diff = (err_h**2 + err**2) ** 0.5

                self.results['priv_loss'] = diff
                self.results['priv_loss_err'] = err_diff

            print(f"[NNDR] Total time: {time.time() - t0:.2f}s")

        except Exception as e:
            if self.verbose:
                print(f"[NNDR] Failed: {e}")
                import traceback
                traceback.print_exc()
            self.results = {}

        return self.results


    def format_output(self) -> Optional[str]:
        if self.results:
            string = f"| Nearest neighbour distance ratio         :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
            if self.hout_data is not None and 'priv_loss' in self.results:
                string += f"\n| Privacy loss (diff. in NNDR)             :   {self.results['priv_loss']:.4f}  {self.results['priv_loss_err']:.4f}   |"
            return string
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            output = [{
                'metric': 'avg_nndr', 'dim': 'p',
                'val': self.results['avg'],
                'err': self.results['err'],
                'n_val': self.results['avg'],
                'n_err': self.results['err']
            }]
            if self.hout_data is not None and 'priv_loss' in self.results:
                output.append({
                    'metric': 'priv_loss_nndr', 'dim': 'p',
                    'val': self.results['priv_loss'],
                    'err': self.results['priv_loss_err'],
                    'n_val': 1 - abs(self.results['priv_loss']),
                    'n_err': self.results['err']
                })
            return output
        return None


class HittingRate(MetricBase):
    """Hitting Rate 메트릭 (Privacy)"""
    
    @staticmethod
    def name() -> str:
        return 'hit_rate'

    @staticmethod
    def type() -> str:
        return 'privacy'

    def evaluate(self, thres_percent=1/30) -> dict:
        self.thres_percent = thres_percent
        thres = thres_percent * (self.real_data.max() - self.real_data.min())
        thres[self.cat_cols] = 0

        hit = 0
        for i, r in self.real_data.iterrows():
            hit += any((abs(r - self.synt_data) <= thres).all(axis='columns'))
        
        hit_rate = hit / len(self.real_data) if len(self.real_data) > 0 else 0
        self.results = {'hit rate': hit_rate}
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Hitting rate ({self.thres_percent:.2f} x range(att))         :   {self.results['hit rate']:.4f}           |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'hit_rate', 'dim': 'p',
                'val': self.results['hit rate'],
                'n_val': 1 - self.results['hit rate']
            }]
        return None


class MedianDistanceToClosestRecord(MetricBase):
    """Median Distance to Closest Record 메트릭 (Privacy)"""
    
    @staticmethod
    def name() -> str:
        return 'dcr'

    @staticmethod
    def type() -> str:
        return 'privacy'

    def evaluate(self, nn_dist='gower') -> dict:
        try:
            distances = _knn_distance(self.synt_data, self.real_data, self.cat_cols, 1, nn_dist)
            in_dists = _knn_distance(self.real_data, self.real_data, self.cat_cols, 1, nn_dist)

            int_nn = np.median(in_dists[0]) if len(in_dists) > 0 else 0
            mut_nn = np.median(distances[0]) if len(distances) > 0 else 0

            if int_nn == 0 and mut_nn == 0:
                dcr = 1
            elif int_nn == 0 and mut_nn != 0:
                dcr = 0
            else:
                dcr = mut_nn / int_nn
            
            self.results = {'mDCR': dcr}
        except Exception as e:
            if self.verbose:
                print(f"Warning: DCR calculation failed: {e}")
            self.results = {}
        
        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Median distance to closest record        :   {self.results['mDCR']:.4f}           |"
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'median_DCR', 'dim': 'p',
                'val': self.results['mDCR'],
                'n_val': np.tanh(self.results['mDCR'])
            }]
        return None


class PrincipalComponentAnalysis(MetricBase):
    """Principal Component Analysis 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'pca'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, num_components=2, preprocess='mean', use_cats=False) -> dict:
        if len(self.num_cols) < 2 and not use_cats:
            if self.verbose:
                print('Error: Principal component analysis did not run, too few attributes!')
            return {}

        if use_cats:
            select_cols = self.num_cols + self.cat_cols
        else:
            select_cols = self.num_cols

        try:
            if preprocess == 'mean':
                r_scaled = self.real_data[select_cols] - self.real_data[select_cols].mean()
                f_scaled = self.synt_data[select_cols] - self.synt_data[select_cols].mean()
            else:
                scaler = StandardScaler()
                r_scaled = pd.DataFrame(
                    scaler.fit_transform(self.real_data[select_cols]),
                    columns=select_cols
                )
                f_scaled = pd.DataFrame(
                    scaler.transform(self.synt_data[select_cols]),
                    columns=select_cols
                )

            pca_real = PCA(n_components=min(num_components, len(select_cols)))
            pca_synt = PCA(n_components=min(num_components, len(select_cols)))

            r_proj = pca_real.fit_transform(r_scaled)
            f_proj = pca_synt.fit_transform(f_scaled)

            # Eigenvalue difference (explained variance)
            exp_var_diff = np.abs(
                np.sum(pca_real.explained_variance_ratio_) - 
                np.sum(pca_synt.explained_variance_ratio_)
            )

            # Eigenvector angle difference
            comp_angle_diff = 0.0
            for i in range(min(len(pca_real.components_), len(pca_synt.components_))):
                cos_angle = np.dot(pca_real.components_[i], pca_synt.components_[i]) / (
                    np.linalg.norm(pca_real.components_[i]) * np.linalg.norm(pca_synt.components_[i])
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                comp_angle_diff += angle

            comp_angle_diff /= min(len(pca_real.components_), len(pca_synt.components_))

            self.results = {
                'exp_var_diff': exp_var_diff,
                'comp_angle_diff': comp_angle_diff
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: PCA calculation failed: {e}")
            self.results = {}

        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            return f"""| PCA difference in eigenvalues (exp. var.):   {self.results['exp_var_diff']:.4f}           |
| PCA angle diff. between eigenvectors     :   {self.results['comp_angle_diff']:.4f}           |"""
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [
                {
                    'metric': 'pca_eigval_diff', 'dim': 'u',
                    'val': self.results['exp_var_diff'],
                    'n_val': 1 - self.results['exp_var_diff']
                },
                {
                    'metric': 'pca_eigvec_ang', 'dim': 'u',
                    'val': self.results['comp_angle_diff'],
                    'n_val': 1 - self.results['comp_angle_diff']
                }
            ]
        return None


def _adversarial_score(real, fake, cat_cols, metric):
    """Adversarial score 계산"""
    left = np.mean(_knn_distance(real, fake, cat_cols, 1, metric)[0] > _knn_distance(real, real, cat_cols, 1, metric)[0])
    right = np.mean(_knn_distance(fake, real, cat_cols, 1, metric)[0] > _knn_distance(fake, fake, cat_cols, 1, metric)[0])
    return 0.5 * (left + right)


def _evaluate_dataset_nnaa(real, fake, num_cols, cat_cols, metric, n_resample):
    """NNAA 평가 헬퍼 함수"""
    real_fake = len(real) / len(fake)
    fake_real = len(fake) / len(real)

    if any([real_fake >= 2, fake_real >= 2]):
        aa_lst = []
        for _ in range(n_resample):
            temp_r = real if real_fake < 2 else real.sample(n=len(fake))
            temp_f = fake if fake_real < 2 else fake.sample(n=len(real))
            aa_lst.append(_adversarial_score(temp_r, temp_f, cat_cols, metric))

        avg = np.mean(aa_lst)
        err = np.std(aa_lst, ddof=1) / np.sqrt(len(aa_lst))
    else:
        avg = _adversarial_score(real, fake, cat_cols, metric)
        err = 0.0

    return avg, err


class NearestNeighbourAdversarialAccuracy(MetricBase):
    """Nearest Neighbour Adversarial Accuracy 메트릭"""
    
    @staticmethod
    def name() -> str:
        return 'nnaa'

    @staticmethod
    def type() -> str:
        return 'utility'

    def evaluate(self, nn_dist='gower', n_resample=30) -> dict:
        try:
            avg, err = _evaluate_dataset_nnaa(
                self.real_data, self.synt_data,
                self.num_cols, self.cat_cols, nn_dist, n_resample
            )

            self.results = {'avg': avg, 'err': err}

            if self.hout_data is not None:
                avg_h, err_h = _evaluate_dataset_nnaa(
                    self.hout_data, self.synt_data,
                    self.num_cols, self.cat_cols, nn_dist, n_resample
                )
                diff = avg_h - avg
                err_diff = np.sqrt(err_h**2 + err**2)
                self.results['priv_loss'] = diff
                self.results['priv_loss_err'] = err_diff
        except Exception as e:
            if self.verbose:
                print(f"Warning: NNAA calculation failed: {e}")
            self.results = {}

        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            string = f"| Nearest neighbour adversarial accuracy   :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
            if self.hout_data is not None and 'priv_loss' in self.results:
                string += f"\n| Privacy loss (diff. in NNAA)             :   {self.results['priv_loss']:.4f}  {self.results['priv_loss_err']:.4f}   |"
            return string
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            output = [{
                'metric': 'nnaa', 'dim': 'u',
                'val': self.results['avg'],
                'err': self.results['err'],
                'n_val': 1 - self.results['avg'],
                'n_err': self.results['err']
            }]
            if self.hout_data is not None and 'priv_loss' in self.results:
                output.append({
                    'metric': 'priv_loss_nnaa', 'dim': 'p',
                    'val': self.results['priv_loss'],
                    'err': self.results['priv_loss_err'],
                    'n_val': 1 - abs(self.results['priv_loss']),
                    'n_err': self.results['priv_loss_err']
                })
            return output
        return None


def _column_entropy(labels):
    """컬럼의 엔트로피 계산"""
    value, counts = np.unique(np.round(labels), return_counts=True)
    return entropy(counts)


class EpsilonIdentifiability(MetricBase):
    """Epsilon Identifiability Risk 메트릭 (Privacy)"""
    
    @staticmethod
    def name() -> str:
        return 'eps_risk'

    @staticmethod
    def type() -> str:
        return 'privacy'

    def evaluate(self, nn_dist='gower') -> dict:
        try:
            if nn_dist == 'euclid':
                real_data = self.real_data[self.num_cols]
                synt_data = self.synt_data[self.num_cols]
            else:
                real_data = self.real_data
                synt_data = self.synt_data

            real = np.asarray(real_data)
            no, x_dim = np.shape(real)
            W = [_column_entropy(real[:, i]) for i in range(x_dim)]
            W_adjust = 1 / (np.array(W) + 1e-16)

            in_dists = _knn_distance(self.real_data, self.real_data, self.cat_cols, 1, nn_dist, W_adjust)[0]
            ext_distances = _knn_distance(self.real_data, self.synt_data, self.cat_cols, 1, nn_dist, W_adjust)[0]

            R_Diff = ext_distances - in_dists
            identifiability_value = np.sum(R_Diff < 0) / float(no)

            self.results = {'eps_risk': identifiability_value}

            if self.hout_data is not None:
                in_dists_h = _knn_distance(self.hout_data, self.hout_data, self.cat_cols, 1, nn_dist, W_adjust)[0]
                ext_distances_h = _knn_distance(self.hout_data, self.synt_data, self.cat_cols, 1, nn_dist, W_adjust)[0]
                R_Diff_h = ext_distances_h - in_dists_h
                identifiability_value_h = np.sum(R_Diff_h < 0) / float(len(self.hout_data))
                self.results['priv_loss'] = self.results['eps_risk'] - identifiability_value_h
        except Exception as e:
            if self.verbose:
                print(f"Warning: Epsilon identifiability calculation failed: {e}")
            self.results = {}

        return self.results

    def format_output(self) -> Optional[str]:
        if self.results:
            string = f"| Epsilon identifiability risk             :   {self.results['eps_risk']:.4f}           |"
            if self.hout_data is not None and 'priv_loss' in self.results:
                string += f"\n| Privacy loss (diff. in eps. risk)        :   {self.results['priv_loss']:.4f}           |"
            return string
        return None

    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            output = [{
                'metric': 'eps_identif_risk', 'dim': 'p',
                'val': self.results['eps_risk'],
                'n_val': 1 - self.results['eps_risk']
            }]
            if self.hout_data is not None and 'priv_loss' in self.results:
                output.append({
                    'metric': 'priv_loss_eps', 'dim': 'p',
                    'val': self.results['priv_loss'],
                    'n_val': 1 - abs(self.results['priv_loss'])
                })
            return output
        return None


# ============================================================================
# 일차원 분포 유사성 메트릭 (Univariate Distribution Similarity)
# ============================================================================

class JensenShannonDivergence(MetricBase):
    """Jensen-Shannon Divergence 메트릭 (일차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'jsd'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def _kl_divergence(self, p, q):
        """Kullback-Leibler Divergence 계산"""
        # 0으로 나누기 방지
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p + 1e-10  # 작은 값 추가하여 0 방지
        q = q + 1e-10
        p = p / p.sum()  # 정규화
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    def _jsd(self, p, q):
        """Jensen-Shannon Divergence 계산: JSD(p||q) = 0.5 * KL(p||M) + 0.5 * KL(q||M), M = 0.5 * (p + q)"""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        M = 0.5 * (p + q)
        return 0.5 * self._kl_divergence(p, M) + 0.5 * self._kl_divergence(q, M)
    
    def evaluate(self, num_bins=50) -> dict:
        """각 변수에 대해 JSD 계산"""
        jsd_values = []
        
        for col in self.real_data.columns:
            real_col = self.real_data[col]
            synt_col = self.synt_data[col]
            
            if col in self.cat_cols:
                # 범주형 변수: 빈도 분포 사용
                real_counts = Counter(real_col)
                synt_counts = Counter(synt_col)
                all_values = set(real_counts.keys()) | set(synt_counts.keys())
                
                p = np.array([real_counts.get(v, 0) for v in all_values])
                q = np.array([synt_counts.get(v, 0) for v in all_values])
                
            else:
                # 수치형 변수: 히스토그램 사용
                min_val = min(real_col.min(), synt_col.min())
                max_val = max(real_col.max(), synt_col.max())
                bins = np.linspace(min_val, max_val, num_bins + 1)
                
                p, _ = np.histogram(real_col, bins=bins)
                q, _ = np.histogram(synt_col, bins=bins)
            
            jsd_val = self._jsd(p, q)
            jsd_values.append(jsd_val)
        
        self.results = {
            'avg': np.mean(jsd_values),
            'err': np.std(jsd_values, ddof=1) / np.sqrt(len(jsd_values)) if len(jsd_values) > 0 else 0,
            'values': jsd_values
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average Jensen-Shannon Divergence (JSD)  :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            # JSD는 0~1 범위로 정규화 (log base 2 사용 시 최대값은 1)
            return [{
                'metric': 'avg_jsd', 'dim': 'u',
                'val': self.results['avg'],
                'err': self.results['err'],
                'n_val': 1 - self.results['avg'],  # 낮을수록 좋음
                'n_err': self.results['err']
            }]
        return None


class KLDivergence(MetricBase):
    """Kullback-Leibler Divergence 메트릭 (일차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'kl_div'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def _kl_divergence(self, p, q):
        """Kullback-Leibler Divergence 계산: KL(P||Q) = Σ p(x) log(p(x)/q(x))"""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p + 1e-10  # 0으로 나누기 방지
        q = q + 1e-10
        p = p / p.sum()  # 정규화
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    def evaluate(self, num_bins=50) -> dict:
        """각 변수에 대해 KL Divergence 계산"""
        kl_values = []
        
        for col in self.real_data.columns:
            real_col = self.real_data[col]
            synt_col = self.synt_data[col]
            
            if col in self.cat_cols:
                # 범주형 변수: 빈도 분포 사용
                real_counts = Counter(real_col)
                synt_counts = Counter(synt_col)
                all_values = set(real_counts.keys()) | set(synt_counts.keys())
                
                p = np.array([real_counts.get(v, 0) for v in all_values])
                q = np.array([synt_counts.get(v, 0) for v in all_values])
                
            else:
                # 수치형 변수: 히스토그램 사용
                min_val = min(real_col.min(), synt_col.min())
                max_val = max(real_col.max(), synt_col.max())
                bins = np.linspace(min_val, max_val, num_bins + 1)
                
                p, _ = np.histogram(real_col, bins=bins)
                q, _ = np.histogram(synt_col, bins=bins)
            
            kl_val = self._kl_divergence(p, q)
            kl_values.append(kl_val)
        
        self.results = {
            'avg': np.mean(kl_values),
            'err': np.std(kl_values, ddof=1) / np.sqrt(len(kl_values)) if len(kl_values) > 0 else 0,
            'values': kl_values
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average KL Divergence (KL)              :   {self.results['avg']:.4f}  {self.results['err']:.4f}   |"
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            # KL Divergence는 0 이상의 값 (낮을수록 좋음)
            return [{
                'metric': 'avg_kl_div', 'dim': 'u',
                'val': self.results['avg'],
                'err': self.results['err'],
                'n_val': 1 / (1 + self.results['avg']),  # 낮을수록 좋음, tanh로 정규화
                'n_err': self.results['err']
            }]
        return None


# ============================================================================
# 다차원 분포 유사성 메트릭 (Multivariate Distribution Similarity)
# ============================================================================

class TheilsU(MetricBase):
    """Theil's U 통계량 메트릭 (다차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'theils_u'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def _theils_u(self, x, y):
        """Theil's U 계산: U(X|Y) = sqrt((H(X) - H(X|Y)) / H(X))"""
        # 엔트로피 계산
        def calc_entropy(data):
            value, counts = np.unique(data, return_counts=True)
            probs = counts / counts.sum()
            return entropy(probs, base=2)
        
        # 조건부 엔트로피 계산
        def calc_conditional_entropy(x, y):
            # 결합 분포
            xy_df = pd.DataFrame({'x': x, 'y': y})
            joint_counts = xy_df.groupby(['x', 'y']).size()
            joint_probs = joint_counts / len(xy_df)
            
            # y의 주변 분포
            y_counts = xy_df.groupby('y').size()
            y_probs = y_counts / len(xy_df)
            
            # 조건부 엔트로피: H(X|Y) = -Σ p(x,y) log(p(x|y))
            cond_entropy = 0
            for (x_val, y_val), joint_prob in joint_probs.items():
                y_prob = y_probs.get(y_val, 1e-10)
                cond_prob = joint_prob / y_prob if y_prob > 0 else 1e-10
                cond_entropy -= joint_prob * np.log2(cond_prob + 1e-10)
            
            return cond_entropy
        
        H_X = calc_entropy(x)
        H_X_given_Y = calc_conditional_entropy(x, y)
        
        if H_X == 0:
            return 0.0
        
        # Theil's U: U(X|Y) = sqrt((H(X) - H(X|Y)) / H(X))
        u_value = np.sqrt((H_X - H_X_given_Y) / H_X)
        return u_value
    
    def evaluate(self) -> dict:
        """모든 변수 쌍에 대해 Theil's U 계산"""
        u_values = []
        columns = self.real_data.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                try:
                    real_u = self._theils_u(
                        self.real_data[col1].astype(str),
                        self.real_data[col2].astype(str)
                    )
                    synt_u = self._theils_u(
                        self.synt_data[col1].astype(str),
                        self.synt_data[col2].astype(str)
                    )
                    
                    # 원본과 합성 데이터의 U 값 차이
                    u_diff = abs(real_u - synt_u)
                    u_values.append(u_diff)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Theil's U calculation failed for ({col1}, {col2}): {e}")
                    continue
        
        self.results = {
            'avg_diff': np.mean(u_values) if len(u_values) > 0 else 0,
            'err': np.std(u_values, ddof=1) / np.sqrt(len(u_values)) if len(u_values) > 0 else 0,
            'values': u_values
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average Theil's U difference            :   {self.results['avg_diff']:.4f}  {self.results['err']:.4f}   |"
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_theils_u_diff', 'dim': 'u',
                'val': self.results['avg_diff'],
                'err': self.results['err'],
                'n_val': 1 - self.results['avg_diff'],  # 낮을수록 좋음
                'n_err': self.results['err']
            }]
        return None


class ConditionalEntropy(MetricBase):
    """조건부 엔트로피 메트릭 (다차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'cond_entropy'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def _conditional_entropy(self, x, y):
        """조건부 엔트로피 계산: H(X|Y) = -Σ p(x,y) log(p(x|y))"""
        xy_df = pd.DataFrame({'x': x.astype(str), 'y': y.astype(str)})
        joint_counts = xy_df.groupby(['x', 'y']).size()
        joint_probs = joint_counts / len(xy_df)
        
        y_counts = xy_df.groupby('y').size()
        y_probs = y_counts / len(xy_df)
        
        cond_entropy = 0
        for (x_val, y_val), joint_prob in joint_probs.items():
            y_prob = y_probs.get(y_val, 1e-10)
            cond_prob = joint_prob / y_prob if y_prob > 0 else 1e-10
            cond_entropy -= joint_prob * np.log2(cond_prob + 1e-10)
        
        return cond_entropy
    
    def evaluate(self) -> dict:
        """모든 변수 쌍에 대해 조건부 엔트로피 차이 계산"""
        cond_entropy_diffs = []
        columns = self.real_data.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                try:
                    real_cond_ent = self._conditional_entropy(
                        self.real_data[col1],
                        self.real_data[col2]
                    )
                    synt_cond_ent = self._conditional_entropy(
                        self.synt_data[col1],
                        self.synt_data[col2]
                    )
                    
                    diff = abs(real_cond_ent - synt_cond_ent)
                    cond_entropy_diffs.append(diff)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Conditional entropy calculation failed for ({col1}, {col2}): {e}")
                    continue
        
        self.results = {
            'avg_diff': np.mean(cond_entropy_diffs) if len(cond_entropy_diffs) > 0 else 0,
            'err': np.std(cond_entropy_diffs, ddof=1) / np.sqrt(len(cond_entropy_diffs)) if len(cond_entropy_diffs) > 0 else 0,
            'values': cond_entropy_diffs
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average Conditional Entropy difference   :   {self.results['avg_diff']:.4f}  {self.results['err']:.4f}   |"
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_cond_entropy_diff', 'dim': 'u',
                'val': self.results['avg_diff'],
                'err': self.results['err'],
                'n_val': 1 / (1 + self.results['avg_diff']),  # 낮을수록 좋음
                'n_err': self.results['err']
            }]
        return None


class EntropyMetric(MetricBase):
    """엔트로피 메트릭 (다차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'entropy'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def _entropy(self, data):
        """엔트로피 계산: H(X) = -Σ p(x) log(p(x))"""
        value, counts = np.unique(data.astype(str), return_counts=True)
        probs = counts / counts.sum()
        return entropy(probs, base=2)
    
    def evaluate(self) -> dict:
        """각 변수에 대해 엔트로피 차이 계산"""
        entropy_diffs = []
        
        for col in self.real_data.columns:
            try:
                real_ent = self._entropy(self.real_data[col])
                synt_ent = self._entropy(self.synt_data[col])
                
                diff = abs(real_ent - synt_ent)
                entropy_diffs.append(diff)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Entropy calculation failed for {col}: {e}")
                continue
        
        self.results = {
            'avg_diff': np.mean(entropy_diffs) if len(entropy_diffs) > 0 else 0,
            'err': np.std(entropy_diffs, ddof=1) / np.sqrt(len(entropy_diffs)) if len(entropy_diffs) > 0 else 0,
            'values': entropy_diffs
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"| Average Entropy difference                :   {self.results['avg_diff']:.4f}  {self.results['err']:.4f}   |"
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'avg_entropy_diff', 'dim': 'u',
                'val': self.results['avg_diff'],
                'err': self.results['err'],
                'n_val': 1 / (1 + self.results['avg_diff']),  # 낮을수록 좋음
                'n_err': self.results['err']
            }]
        return None


class PearsonCorrelation(MetricBase):
    """피어슨 상관계수 메트릭 (다차원 분포 유사성)"""
    
    @staticmethod
    def name() -> str:
        return 'pearson_corr'
    
    @staticmethod
    def type() -> str:
        return 'utility'
    
    def evaluate(self) -> dict:
        """수치형 변수들 간의 피어슨 상관계수 차이 계산"""
        if len(self.num_cols) < 2:
            if self.verbose:
                print("Warning: Need at least 2 numerical columns for Pearson correlation.")
            return {}
        
        real_corr = self.real_data[self.num_cols].corr()
        synt_corr = self.synt_data[self.num_cols].corr()
        
        # 대각선 제외한 상관계수 차이
        corr_diff = real_corr - synt_corr
        # 상삼각 행렬만 사용 (중복 제거)
        mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
        corr_diff_values = corr_diff.values[mask]
        
        self.results = {
            'avg_diff': np.mean(np.abs(corr_diff_values)),
            'err': np.std(np.abs(corr_diff_values), ddof=1) / np.sqrt(len(corr_diff_values)) if len(corr_diff_values) > 0 else 0,
            'frobenius_norm': np.linalg.norm(corr_diff.values, ord='fro'),
            'values': corr_diff_values.tolist()
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"""| Average Pearson Correlation difference  :   {self.results['avg_diff']:.4f}  {self.results['err']:.4f}   |
| Frobenius norm of correlation diff.    :   {self.results['frobenius_norm']:.4f}           |"""
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [
                {
                    'metric': 'avg_pearson_corr_diff', 'dim': 'u',
                    'val': self.results['avg_diff'],
                    'err': self.results['err'],
                    'n_val': 1 - self.results['avg_diff'],  # 낮을수록 좋음
                    'n_err': self.results['err']
                },
                {
                    'metric': 'pearson_corr_fro_norm', 'dim': 'u',
                    'val': self.results['frobenius_norm'],
                    'n_val': 1 / (1 + self.results['frobenius_norm'])
                }
            ]
        return None


# ============================================================================
# 안정성 메트릭 (Stability / Disclosure Risk)
# ============================================================================

class IdentificationRisk(MetricBase):
    """구별 위험도 (Identification Risk) 메트릭: 1/n Σ I(S_i = R_j)"""
    
    @staticmethod
    def name() -> str:
        return 'ident_risk'
    
    @staticmethod
    def type() -> str:
        return 'privacy'
    
    def evaluate(self, exact_match=True) -> dict:
        """
        구별 위험도 계산: 1/n Σ I(S_i = R_j)
        
        Args:
            exact_match: True면 정확히 일치하는 경우만, False면 거리 기반
        """
        n_synt = len(self.synt_data)
        n_real = len(self.real_data)
        matches = 0
        close_matches = 0
        
        if exact_match:
            # 정확히 일치하는 레코드 수 계산
            synt_array = self.synt_data.values
            real_array = self.real_data.values
            
            # 각 합성 레코드에 대해 실제 데이터와 일치하는지 확인
            for synt_row in synt_array:
                # 모든 실제 레코드와 비교
                for real_row in real_array:
                    if np.array_equal(synt_row, real_row):
                        matches += 1
                        break  # 한 번만 매칭되면 됨
            
            ident_risk = matches / n_synt if n_synt > 0 else 0
        else:
            # 거리 기반: 매우 가까운 레코드도 고려
            # Gower 거리를 사용하여 임계값 이내의 레코드 수 계산
            try:
                distances = _knn_distance(
                    self.synt_data, 
                    self.real_data, 
                    self.cat_cols, 
                    1, 
                    'gower'
                )
                # 거리가 0에 가까운 경우 (임계값 0.01)
                threshold = 0.01
                close_matches = np.sum(distances[0] < threshold)
                ident_risk = close_matches / n_synt if n_synt > 0 else 0
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Distance-based identification risk calculation failed: {e}")
                ident_risk = 0
        
        self.results = {
            'ident_risk': ident_risk,
            'n_synt': n_synt,
            'n_real': n_real,
            'n_matches': matches if exact_match else close_matches
        }
        return self.results
    
    def format_output(self) -> Optional[str]:
        if self.results:
            return f"""| Identification Risk                      :   {self.results['ident_risk']:.4f}           |
|   -> # synthetic records: {self.results['n_synt']:6d}                    |
|   -> # real records:      {self.results['n_real']:6d}                    |"""
        return None
    
    def normalize_output(self) -> Optional[List[dict]]:
        if self.results:
            return [{
                'metric': 'ident_risk', 'dim': 'p',
                'val': self.results['ident_risk'],
                'n_val': 1 - self.results['ident_risk']  # 낮을수록 좋음 (프라이버시)
            }]
        return None


# ============================================================================
# 메트릭 실행 프레임워크
# ============================================================================

class SynthevalEvaluator:
    """합성 데이터 평가 프레임워크"""
    
    def __init__(
        self,
        real_dataframe: pd.DataFrame,
        holdout_dataframe: Optional[pd.DataFrame] = None,
        cat_cols: Optional[List[str]] = None,
        unique_threshold: int = 10,
        nn_distance: str = 'gower',  # 'gower' or 'euclid'
        verbose: bool = True
    ):
        self.real = real_dataframe
        self.verbose = verbose
        self.nn_dist = nn_distance

        if holdout_dataframe is not None:
            if len(real_dataframe.columns) == len(holdout_dataframe.columns):
                holdout_dataframe = holdout_dataframe[real_dataframe.columns.tolist()]
            assert real_dataframe.columns.tolist() == holdout_dataframe.columns.tolist(), \
                'Columns in train and test dataframe are not the same'
            self.hold_out = holdout_dataframe
        else:
            self.hold_out = None

        if cat_cols is None:
            cat_cols = get_cat_variables(real_dataframe, unique_threshold)
            if self.verbose:
                print(f"Inferred categorical columns (unique threshold: {unique_threshold}):\n{cat_cols}")
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [col for col in real_dataframe.columns if col not in cat_cols]
        self._raw_results = {}

    def evaluate(
        self,
        synthetic_dataframe: pd.DataFrame,
        analysis_target_var: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> dict:
        """합성 데이터 평가"""
        self.synt = synthetic_dataframe
        
        if len(self.real.columns) == len(synthetic_dataframe.columns):
            synthetic_dataframe = synthetic_dataframe[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synthetic_dataframe.columns.tolist(), \
            'Columns in real and synthetic dataframe are not the same'
        
        if self.verbose:
            print('Syntheval: synthetic data read successfully')

        # 사용 가능한 메트릭
        available_metrics = {
            # Utility 메트릭
            'dwm': DimensionWiseMeans,
            'p_mse': PropensityMSE,
            'corr_diff': MixedCorrelation,
            'ks_test': KolmogorovSmirnovTest,
            'h_dist': HellingerDistance,
            'cio': ConfidenceIntervalOverlap,
            'mi_diff': MutualInformation,
            'q_mse': QuantileMSE,
            'pca': PrincipalComponentAnalysis,
            'nnaa': NearestNeighbourAdversarialAccuracy,
            # 일차원 분포 유사성 메트릭
            'jsd': JensenShannonDivergence,
            'kl_div': KLDivergence,
            # 다차원 분포 유사성 메트릭
            'theils_u': TheilsU,
            'cond_entropy': ConditionalEntropy,
            'entropy': EntropyMetric,
            'pearson_corr': PearsonCorrelation,
            # Privacy 메트릭
            'nndr': NearestNeighbourDistanceRatio,
            'hit_rate': HittingRate,
            'dcr': MedianDistanceToClosestRecord,
            'eps_risk': EpsilonIdentifiability,
            'ident_risk': IdentificationRisk
        }

        # 기본 메트릭 설정 (전체 메트릭 실행)
        if metrics is None:
            metrics = ['dwm', 'p_mse', 'corr_diff', 'ks_test', 'h_dist', 'cio', 'mi_diff', 'q_mse', 'pca', 'nnaa', 
                      'jsd', 'kl_div', 'theils_u', 'cond_entropy', 'entropy', 'pearson_corr',
                      'nndr', 'hit_rate', 'dcr', 'eps_risk', 'ident_risk']

        # 전처리 설정
        CLE = ConsistentLabelEncoding(
            self.real, 
            self.synt, 
            self.categorical_columns, 
            self.numerical_columns, 
            self.hold_out
        )
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        hout_data = CLE.encode(self.hold_out) if self.hold_out is not None else None

        # 메트릭 실행
        results = {}
        utility_output = []
        privacy_output = []
        metric_times = {}  # 메트릭별 소요 시간 저장
        
        for metric_name in metrics:
            print(f"Running metric: {metric_name}")
            if metric_name not in available_metrics:
                if self.verbose:
                    print(f"Warning: Unknown metric '{metric_name}', skipping...")
                continue
            
            try:
                start_time = time.time()
                
                MetricClass = available_metrics[metric_name]
                metric = MetricClass(
                    real_data, 
                    synt_data, 
                    hout_data,
                    self.categorical_columns,
                    self.numerical_columns,
                    do_preprocessing=False,  # 이미 전처리됨
                    verbose=self.verbose
                )
                
                # 메트릭별 특수 파라미터 처리
                if metric_name in ['nndr', 'dcr', 'nnaa', 'eps_risk']:
                    metric_result = metric.evaluate(nn_dist=self.nn_dist)
                else:
                    metric_result = metric.evaluate()
                
                elapsed_time = time.time() - start_time
                metric_times[metric_name] = elapsed_time
                
                # 소요 시간 출력
                if elapsed_time < 60:
                    print(f"  ✓ {metric_name} completed in {elapsed_time:.2f} seconds")
                else:
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    print(f"  ✓ {metric_name} completed in {minutes}m {seconds:.2f}s")
                
                results[metric_name] = metric_result
                
                # normalize_output이 있으면 호출하여 추가 메트릭 생성
                normalized = metric.normalize_output()
                if normalized:
                    for norm_result in normalized:
                        norm_metric_name = norm_result.get('metric')
                        if norm_metric_name and norm_metric_name != metric_name:
                            # 정규화된 결과를 별도 메트릭으로 저장
                            results[norm_metric_name] = norm_result
                            if self.verbose:
                                print(f"    → {norm_metric_name}: {norm_result.get('val', 'N/A')}")
                
                # 출력 포맷팅
                output_str = metric.format_output()
                if output_str:
                    metric_type = MetricClass.type()
                    if metric_type == 'utility':
                        utility_output.append(output_str)
                    elif metric_type == 'privacy':
                        privacy_output.append(output_str)
                
            except Exception as e:
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                if self.verbose:
                    print(f"  ✗ {metric_name} failed after {elapsed_time:.2f}s. Exception: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 결과 출력
        if self.verbose:
            print("\n" + "="*100)
            if utility_output:
                print("UTILITY METRICS:")
                print("-"*100)
                for out in utility_output:
                    print(out)
            if privacy_output:
                print("\nPRIVACY METRICS:")
                print("-"*100)
                for out in privacy_output:
                    print(out)
            print("="*100)
            
            # 메트릭별 소요 시간 요약 출력
            if metric_times:
                print("\nMETRIC EXECUTION TIMES:")
                print("-"*100)
                total_time = sum(metric_times.values())
                # 시간 순으로 정렬
                sorted_times = sorted(metric_times.items(), key=lambda x: x[1], reverse=True)
                for metric_name, elapsed_time in sorted_times:
                    percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0
                    if elapsed_time < 60:
                        print(f"  {metric_name:20s}: {elapsed_time:8.2f}s ({percentage:5.1f}%)")
                    else:
                        minutes = int(elapsed_time // 60)
                        seconds = elapsed_time % 60
                        print(f"  {metric_name:20s}: {minutes:3d}m {seconds:5.2f}s ({percentage:5.1f}%)")
                print("-"*100)
                if total_time < 60:
                    print(f"  {'TOTAL':20s}: {total_time:8.2f}s")
                else:
                    total_minutes = int(total_time // 60)
                    total_seconds = total_time % 60
                    print(f"  {'TOTAL':20s}: {total_minutes:3d}m {total_seconds:5.2f}s")
            print("="*100)

        self._raw_results = results
        # 메트릭별 시간 정보도 결과에 포함
        if metric_times:
            results['_execution_times'] = metric_times
        return results


# ============================================================================
# 데이터 변환 및 메인 함수
# ============================================================================

def numpy_to_dataframe(X_num, X_cat, y, num_cols=None, cat_cols=None, target_col='target'):
    """numpy 배열을 pandas DataFrame으로 변환"""
    dfs = []
    
    if X_num is not None:
        if num_cols is None:
            num_cols = [f'num_{i}' for i in range(X_num.shape[1])]
        df_num = pd.DataFrame(X_num, columns=num_cols)
        dfs.append(df_num)
    
    if X_cat is not None:
        if cat_cols is None:
            cat_cols = [f'cat_{i}' for i in range(X_cat.shape[1])]
        df_cat = pd.DataFrame(X_cat, columns=cat_cols)
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(str)
        dfs.append(df_cat)
    
    if len(dfs) > 0:
        df = pd.concat(dfs, axis=1)
    else:
        df = pd.DataFrame()
    
    df[target_col] = y
    return df


def get_column_names(real_data_path):
    """실제 데이터 경로에서 컬럼 이름 정보 가져오기"""
    info_path = Path(real_data_path) / 'info.json'
    if info_path.exists():
        info = lib.load_json(info_path)
        num_cols = info.get('num_cols', None)
        cat_cols = info.get('cat_cols', None)
        return num_cols, cat_cols
    return None, None


def train_syntheval(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict=None,
    seed=0,
    change_val=True,
    metrics=None,  # 사용할 메트릭 리스트
    device=None
):
    """
    Syntheval 스타일 평가 실행
    
    Args:
        parent_dir: 합성 데이터가 저장된 디렉토리
        real_data_path: 실제 데이터 경로
        eval_type: 평가 타입 ('synthetic', 'real', 'merged')
        T_dict: 변환 설정 (사용하지 않음)
        seed: 랜덤 시드
        change_val: validation split 변경 여부
        metrics: 사용할 메트릭 리스트 (None이면 기본 메트릭 사용)
        device: 디바이스 (사용하지 않음)
    
    Returns:
        dict: 평가 결과
    """
    print('='*100)
    print('SynthEval 스타일 평가 시작')
    print('='*100)
    
    # 실제 데이터 로드
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(
            real_data_path, val_size=0.2
        )
        if X_num_real is not None and X_num_val is not None:
            X_num_real = np.concatenate([X_num_real, X_num_val], axis=0)
        if X_cat_real is not None and X_cat_val is not None:
            X_cat_real = np.concatenate([X_cat_real, X_cat_val], axis=0)
        y_real = np.concatenate([y_real, y_val], axis=0)
    else:
        X_num_real, X_cat_real, y_real = read_pure_data(real_data_path, 'train')
    
    # Test 데이터 로드 (holdout으로 사용)
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')
    
    # 컬럼 이름 가져오기
    num_cols, cat_cols = get_column_names(real_data_path)
    
    # 실제 데이터를 DataFrame으로 변환
    df_real = numpy_to_dataframe(
        X_num_real, X_cat_real, y_real, 
        num_cols=num_cols, cat_cols=cat_cols, target_col='target'
    )
    
    # Test 데이터를 DataFrame으로 변환 (holdout)
    df_test = numpy_to_dataframe(
        X_num_test, X_cat_test, y_test,
        num_cols=num_cols, cat_cols=cat_cols, target_col='target'
    )
    
    # Evaluator 초기화
    evaluator = SynthevalEvaluator(
        real_dataframe=df_real,
        holdout_dataframe=df_test,
        cat_cols=cat_cols,
        nn_distance='gower',  # 'gower' or 'euclid'
        verbose=True
    )
    
    # 합성 데이터 로드 및 평가
    if eval_type == 'synthetic':
        print(f'합성 데이터 로드: {parent_dir}')
        X_num_synt, X_cat_synt, y_synt = read_pure_data(parent_dir)
        
        df_synt = numpy_to_dataframe(
            X_num_synt, X_cat_synt, y_synt,
            num_cols=num_cols, cat_cols=cat_cols, target_col='target'
        )
        
        results = evaluator.evaluate(
            df_synt,
            analysis_target_var='target',
            metrics=metrics
        )
        
    elif eval_type == 'real':
        print('실제 데이터 평가 (baseline)')
        results = evaluator.evaluate(
            df_real,
            analysis_target_var='target',
            metrics=metrics
        )
        
    elif eval_type == 'merged':
        print('Merged 데이터 평가')
        X_num_synt, X_cat_synt, y_synt = read_pure_data(parent_dir)
        
        if X_num_real is not None and X_num_synt is not None:
            X_num_merged = np.concatenate([X_num_real, X_num_synt], axis=0)
        else:
            X_num_merged = X_num_real if X_num_real is not None else X_num_synt
            
        if X_cat_real is not None and X_cat_synt is not None:
            X_cat_merged = np.concatenate([X_cat_real, X_cat_synt], axis=0)
        else:
            X_cat_merged = X_cat_real if X_cat_real is not None else X_cat_synt
            
        y_merged = np.concatenate([y_real, y_synt], axis=0)
        
        df_merged = numpy_to_dataframe(
            X_num_merged, X_cat_merged, y_merged,
            num_cols=num_cols, cat_cols=cat_cols, target_col='target'
        )
        
        results = evaluator.evaluate(
            df_merged,
            analysis_target_var='target',
            metrics=metrics
        )
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    
    # 결과를 JSON으로 저장 가능한 형태로 변환
    def convert_to_serializable(obj):
        """재귀적으로 numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    results_serializable = convert_to_serializable(results)
    
    # 결과 저장 (기존 파일이 있으면 병합)
    output_path = Path(parent_dir) / 'eval_syntheval_style.json'
    
    if output_path.exists():
        try:
            existing_results = lib.load_json(output_path)
            print(f'\n기존 결과 파일 발견: {output_path}')
            print(f'  기존 메트릭: {list(existing_results.keys())}')
            
            # 기존 결과에 새로운 결과 병합 (새로운 결과가 우선)
            merged_results = existing_results.copy()
            for key, value in results_serializable.items():
                if key in merged_results:
                    print(f'  갱신: {key}')
                else:
                    print(f'  추가: {key}')
                merged_results[key] = value
            
            results_serializable = merged_results
            print(f'  병합 후 메트릭: {list(results_serializable.keys())}')
        except Exception as e:
            print(f'\n기존 결과 파일 로드 실패: {e}')
            print('새로운 결과만 저장합니다.')
    
    lib.dump_json(results_serializable, output_path)
    print(f'\n결과 저장: {output_path}')
    
    return results_serializable


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SynthEval 스타일 평가 스크립트')
    parser.add_argument('--config', metavar='FILE', required=True,
                       help='설정 파일 경로 (config.toml)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       choices=['dwm', 'p_mse', 'corr_diff', 'ks_test', 'h_dist', 'cio', 
                               'mi_diff', 'q_mse', 'pca', 'nnaa',
                               'jsd', 'kl_div', 'theils_u', 'cond_entropy', 'entropy', 'pearson_corr',
                               'nndr', 'hit_rate', 'dcr', 'eps_risk', 'ident_risk'],
                       help='사용할 메트릭 (기본값: 전체 메트릭 사용)')
    parser.add_argument('--change_val', action='store_true', default=False,
                       help='validation split 변경 여부')
    parser.add_argument('--sizes', type=float, nargs='+', default=None,
                       help='크기별 평가 (예: --sizes 1.0 1.5 2.0). 지정하면 size_Xx 디렉토리들을 평가')
    
    args = parser.parse_args()
    config = lib.load_config(args.config)
    
    # 설정 추출
    parent_dir = config.get('parent_dir', '')
    real_data_path = config.get('real_data_path', '')
    
    eval_config = config.get('eval', {})
    eval_type_config = eval_config.get('type', {})
    eval_type = eval_type_config.get('eval_type', 'synthetic')
    
    T_dict = eval_config.get('T', {})
    seed = config.get('seed', 0)
    
    # 크기별 평가 모드
    if args.sizes:
        print('='*100)
        print(f'크기별 SynthEval 스타일 평가 시작: {args.sizes}')
        print('='*100)
        
        base_parent_dir = Path(parent_dir)
        results_summary = {}
        
        for size in args.sizes:
            size_dir = base_parent_dir / f'size_{size}x'
            
            if not size_dir.exists():
                print(f'\n경고: 디렉토리를 찾을 수 없습니다: {size_dir}')
                print('건너뜁니다...\n')
                continue
            
            print('\n' + '='*100)
            print(f'평가 중: Size {size}x')
            print(f'디렉토리: {size_dir}')
            print('='*100)
            
            # 평가 실행
            try:
                result = train_syntheval(
                    parent_dir=str(size_dir),
                    real_data_path=real_data_path,
                    eval_type=eval_type,
                    T_dict=T_dict,
                    seed=seed,
                    change_val=args.change_val,
                    metrics=args.metrics
                )
                results_summary[f'size_{size}x'] = {
                    'status': 'success',
                    'result_file': str(size_dir / 'eval_syntheval_style.json')
                }
                print(f'\n✅ 완료: Size {size}x')
            except Exception as e:
                print(f'\n❌ 오류 발생 (Size {size}x): {e}')
                import traceback
                traceback.print_exc()
                results_summary[f'size_{size}x'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 요약 출력
        print('\n' + '='*100)
        print('크기별 평가 완료 요약')
        print('='*100)
        for size_name, info in results_summary.items():
            if info['status'] == 'success':
                print(f'  ✓ {size_name}: {info["result_file"]}')
            else:
                print(f'  ✗ {size_name}: 실패 - {info["error"]}')
        print('='*100)
    
    else:
        # 단일 디렉토리 평가 (기존 동작)
        train_syntheval(
            parent_dir=parent_dir,
            real_data_path=real_data_path,
            eval_type=eval_type,
            T_dict=T_dict,
            seed=seed,
            change_val=args.change_val,
            metrics=args.metrics
        )
