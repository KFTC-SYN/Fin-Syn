"""
eval_privacy.py

Data-only privacy evaluation for synthetic tabular data
Metrics:
- Identification Risk (IR)
- Distance to Closest Record (DCR)
- Nearest Neighbour Distance Ratio (NNDR)
- Targeted Correct Attribution Probability (TCAP)

Style aligned with eval_syntheval.py
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path
import lib

# ============================================================================
# Utility
# ============================================================================

def get_cat_variables(df, threshold=10):
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif df[col].nunique() < threshold:
            cat_cols.append(col)
    return cat_cols


class ConsistentLabelEncoding:
    def __init__(self, real, fake, cat_cols, num_cols):
        joint = pd.concat([real, fake], axis=0)

        self.cat_cols = cat_cols
        self.num_cols = num_cols

        if cat_cols:
            self.cat_enc = OrdinalEncoder().fit(joint[cat_cols])
        else:
            self.cat_enc = None

        if num_cols:
            self.num_enc = MinMaxScaler().fit(joint[num_cols])
        else:
            self.num_enc = None

    def encode(self, df):
        df = df.copy()
        if self.cat_enc:
            df[self.cat_cols] = self.cat_enc.transform(df[self.cat_cols]).astype(int)
        if self.num_enc:
            df[self.num_cols] = self.num_enc.transform(df[self.num_cols])
        return df


# ============================================================================
# Distance utilities (Gower + Euclidean)
# ============================================================================

def _gower_distance(X, Y, cat_mask):
    num_mask = ~cat_mask
    X_num, Y_num = X[:, num_mask], Y[:, num_mask]
    X_cat, Y_cat = X[:, cat_mask], Y[:, cat_mask]

    dist = np.zeros((len(X), len(Y)))

    if X_num.shape[1] > 0:
        ranges = np.ptp(np.vstack([X_num, Y_num]), axis=0)
        ranges[ranges == 0] = 1
        dist += cdist(X_num, Y_num, metric='cityblock') / ranges.sum()

    if X_cat.shape[1] > 0:
        dist += cdist(X_cat, Y_cat, metric='hamming')

    return dist / X.shape[1]


def knn_distances(df_a, df_b, cat_cols, k=2):
    cat_mask = np.array([c in cat_cols for c in df_a.columns])
    A, B = df_a.values, df_b.values
    D = _gower_distance(A, B, cat_mask)
    return np.sort(D, axis=1)[:, :k]


# ============================================================================
# Metric base
# ============================================================================

class MetricBase(ABC):
    def __init__(self, real, synt, cat_cols, num_cols):
        self.real = real
        self.synt = synt
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.results = {}

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def type():
        pass

    @abstractmethod
    def evaluate(self):
        pass


# ============================================================================
# Privacy Metrics
# ============================================================================

class IdentificationRisk(MetricBase):
    """
    IR = fraction of synthetic records exactly matching any real record
    """

    @staticmethod
    def name():
        return "ir"

    @staticmethod
    def type():
        return "privacy"

    def evaluate(self):
        real_set = set(map(tuple, self.real.values))
        matches = sum(tuple(r) in real_set for r in self.synt.values)
        self.results = {
            "IR": matches / len(self.synt)
        }
        return self.results


class DistanceToClosestRecord(MetricBase):
    """
    DCR = median(d(synth, real)) / median(d(real, real))
    """

    @staticmethod
    def name():
        return "dcr"

    @staticmethod
    def type():
        return "privacy"

    def evaluate(self):
        d_sr = knn_distances(self.synt, self.real, self.cat_cols, k=1).flatten()
        d_rr = knn_distances(self.real, self.real, self.cat_cols, k=2)[:, 1]

        dcr = np.median(d_sr) / (np.median(d_rr) + 1e-12)
        self.results = {"DCR": dcr}
        return self.results


class NearestNeighbourDistanceRatio(MetricBase):
    """
    NNDR = d1 / d2
    """

    @staticmethod
    def name():
        return "nndr"

    @staticmethod
    def type():
        return "privacy"

    def evaluate(self):
        dists = knn_distances(self.synt, self.real, self.cat_cols, k=2)
        ratios = dists[:, 0] / (dists[:, 1] + 1e-12)
        self.results = {
            "NNDR_mean": ratios.mean(),
            "NNDR_std": ratios.std()
        }
        return self.results


class TCAP(MetricBase):
    """
    Data-only proxy for attribute inference risk
    """

    @staticmethod
    def name():
        return "tcap"

    @staticmethod
    def type():
        return "privacy"

    def evaluate(self, sensitive_cols: List[str], quasi_cols: List[str]):
        def cond_acc(df):
            correct, total = 0, 0
            for _, g in df.groupby(quasi_cols):
                for c in sensitive_cols:
                    m = g[c].mode()
                    if len(m) == 0:
                        continue
                    correct += (g[c] == m.iloc[0]).sum()
                    total += len(g)
            return correct / total if total > 0 else 0

        acc_r = cond_acc(self.real)
        acc_s = cond_acc(self.synt)

        self.results = {
            "TCAP_real": acc_r,
            "TCAP_synth": acc_s,
            "TCAP_gap": abs(acc_r - acc_s)
        }
        return self.results


# ============================================================================
# Evaluator
# ============================================================================

class PrivacyEvaluator:
    def __init__(
        self,
        real_df: pd.DataFrame,
        synt_df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None
    ):
        if cat_cols is None:
            cat_cols = get_cat_variables(real_df)

        num_cols = [c for c in real_df.columns if c not in cat_cols]

        enc = ConsistentLabelEncoding(real_df, synt_df, cat_cols, num_cols)
        self.real = enc.encode(real_df)
        self.synt = enc.encode(synt_df)

        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def evaluate(
        self,
        sensitive_cols: List[str],
        quasi_cols: List[str]
    ):
        metrics = [
            IdentificationRisk,
            DistanceToClosestRecord,
            NearestNeighbourDistanceRatio
        ]

        results = {}

        for M in metrics:
            m = M(self.real, self.synt, self.cat_cols, self.num_cols)
            results[M.name()] = m.evaluate()

        # TCAP (needs args)
        tcap = TCAP(self.real, self.synt, self.cat_cols, self.num_cols)
        results["tcap"] = tcap.evaluate(
            sensitive_cols=sensitive_cols,
            quasi_cols=quasi_cols
        )

        return results


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Data-only Privacy Evaluation for Synthetic Data")
    parser.add_argument(
        '--config',
        metavar='FILE',
        required=True,
        help='Path to config file (json/yaml)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        choices=['ir', 'dcr', 'nndr', 'tcap'],
        help='Privacy metrics to run (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='eval_privacy.json',
        help='Output filename for privacy evaluation results'
    )

    args = parser.parse_args()
    config = lib.load_config(args.config)

    # ---------------------------------------------------------------------
    # Config parsing (syntheval-style)
    # ---------------------------------------------------------------------
    real_path = Path(config['real_data_path'])
    synt_path = Path(config['parent_dir'])

    # 안전하게 privacy 설정 가져오기 (키가 없을 경우 빈 딕셔너리 반환)
    privacy_config = config['eval'].get('privacy', {})
    sensitive_cols = privacy_config.get('sensitive_cols', [])
    quasi_cols = privacy_config.get('quasi_cols', [])

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    print('=' * 100)
    print('Privacy Evaluation (data-only)')
    print('=' * 100)

    # 실제 데이터 로드 (npy 형식)
    X_num_real, X_cat_real, y_real = lib.read_pure_data(real_path, 'train')
    real_df = lib.concat_to_pd(X_num_real, X_cat_real, y_real)
    # 컬럼 이름을 모두 문자열로 변환 (sklearn 호환성)
    real_df.columns = real_df.columns.astype(str)
    
    # 합성 데이터 로드 (npy 형식)
    X_num_synt, X_cat_synt, y_synt = lib.read_pure_data(synt_path, 'train')
    synt_df = lib.concat_to_pd(X_num_synt, X_cat_synt, y_synt)
    # 컬럼 이름을 모두 문자열로 변환 (sklearn 호환성)
    synt_df.columns = synt_df.columns.astype(str)

    # ---------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------
    evaluator = PrivacyEvaluator(
        real_df=real_df,
        synt_df=synt_df,
        cat_cols=config.get('cat_cols', None)
    )

    results = {}

    selected_metrics = args.metrics or ['ir', 'dcr', 'nndr', 'tcap']

    # TCAP 메트릭이 선택된 경우에만 sensitive_cols와 quasi_cols 검증
    if 'tcap' in selected_metrics:
        if len(sensitive_cols) == 0 or len(quasi_cols) == 0:
            raise ValueError(
                "TCAP requires 'sensitive_cols' and 'quasi_cols' "
                "to be specified under config['eval']['privacy']"
            )

    if 'ir' in selected_metrics:
        results['ir'] = IdentificationRisk(
            evaluator.real, evaluator.synt,
            evaluator.cat_cols, evaluator.num_cols
        ).evaluate()

    if 'dcr' in selected_metrics:
        results['dcr'] = DistanceToClosestRecord(
            evaluator.real, evaluator.synt,
            evaluator.cat_cols, evaluator.num_cols
        ).evaluate()

    if 'nndr' in selected_metrics:
        results['nndr'] = NearestNeighbourDistanceRatio(
            evaluator.real, evaluator.synt,
            evaluator.cat_cols, evaluator.num_cols
        ).evaluate()

    if 'tcap' in selected_metrics:
        results['tcap'] = TCAP(
            evaluator.real, evaluator.synt,
            evaluator.cat_cols, evaluator.num_cols
        ).evaluate(
            sensitive_cols=sensitive_cols,
            quasi_cols=quasi_cols
        )

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    output_path = Path(config['parent_dir']) / args.output
    lib.dump_json(results, output_path)

    print('-' * 100)
    print('Privacy metrics completed:')
    for k, v in results.items():
        print(f'  - {k}: {v}')
    print('-' * 100)
    print(f'Results saved to: {output_path}')
    print('=' * 100)
