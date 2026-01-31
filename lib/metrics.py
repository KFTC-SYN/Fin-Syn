import enum
from typing import Any, Optional, Tuple, Dict, Union, cast
from functools import partial

import numpy as np
import scipy.special
import sklearn.metrics as skm

from . import util
from .util import TaskType


class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'

class MetricsReport:
    def __init__(self, report: dict, task_type: TaskType):
        self._res = {k: {} for k in report.keys()}
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            self._metrics_names = ["acc", "f1"]
            # 첫 번째 split에서 클래스별 f1 score 이름 결정 (중복 방지)
            first_split = list(report.keys())[0]
            class_keys = [key for key in report[first_split].keys() if isinstance(key, str) and key.isdigit()]
            
            if task_type == TaskType.BINCLASS:
                # 이진 분류: 클래스 0 (negative), 클래스 1 (positive)
                if "0" in report[first_split]:
                    self._metrics_names.append("f1_0")
                if "1" in report[first_split]:
                    self._metrics_names.append("f1_1")
            elif task_type == TaskType.MULTICLASS:
                # 다중 분류: 각 클래스별 f1 score
                for class_key in sorted(class_keys, key=int):
                    self._metrics_names.append(f"f1_{class_key}")
            
            # 클래스 불균형 평가 지표 이름 추가
            if "weighted avg" in report[first_split]:
                self._metrics_names.append("f1_weighted")
            if "balanced_acc" in report[first_split]:
                self._metrics_names.append("balanced_acc")
            if "mcc" in report[first_split]:
                self._metrics_names.append("mcc")
            if "kappa" in report[first_split]:
                self._metrics_names.append("kappa")
            if task_type == TaskType.BINCLASS:
                self._metrics_names.append("roc_auc")
            
            # 각 split에 대해 지표 값 저장
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                
                # 2025.11.24 각 클래스별 f1 score 추가
                if task_type == TaskType.BINCLASS:
                    if "0" in report[k]:
                        self._res[k]["f1_0"] = report[k]["0"]["f1-score"]
                    if "1" in report[k]:
                        self._res[k]["f1_1"] = report[k]["1"]["f1-score"]
                elif task_type == TaskType.MULTICLASS:
                    for class_key in sorted(class_keys, key=int):
                        if class_key in report[k]:
                            self._res[k][f"f1_{class_key}"] = report[k][class_key]["f1-score"]
                
                # 2025.11.24 클래스 불균형 평가 지표 추가
                if "weighted avg" in report[k]:
                    self._res[k]["f1_weighted"] = report[k]["weighted avg"]["f1-score"]
                if "balanced_acc" in report[k]:
                    self._res[k]["balanced_acc"] = report[k]["balanced_acc"]
                if "mcc" in report[k]:
                    self._res[k]["mcc"] = report[k]["mcc"]
                if "kappa" in report[k]:
                    self._res[k]["kappa"] = report[k]["kappa"]
                if task_type == TaskType.BINCLASS:
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]

        elif task_type == TaskType.REGRESSION:
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list[str]:
        return self._res.keys()

    def get_metrics_names(self) -> list[str]:
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        return self._res[split][metric]

    def get_val_score(self) -> float:
        # return self._res["val"]["r2"] if "r2" in self._res["val"] else self._res["val"]["f1"]
        return self._res["val"]["r2"] if "r2" in self._res["val"] else self._res["val"]["f1_1"]
    
    def get_test_score(self) -> float:
        # return self._res["test"]["r2"] if "r2" in self._res["test"] else self._res["test"]["f1"]
        return self._res["test"]["r2"] if "r2" in self._res["test"] else self._res["test"]["f1_1"]
    
    def get_val_score_for_fraud(self) -> float:
        return self._res["val"]["f1_1"]
    
    def get_test_score_for_fraud(self) -> float:
        return self._res["test"]["f1_1"]

    def print_metrics(self) -> None:
        res = {
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }
    
        print("*"*100)
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])

        return res

class SeedsMetricsReport:
    def __init__(self):
        self._reports = []

    def add_report(self, report: MetricsReport) -> None:
        self._reports.append(report)
    
    def get_mean_std(self) -> dict:
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self) -> dict:
        res = {split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]} for split in ["val", "test"]}
        print("="*100)
        print("EVAL RESULTS:")
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])
        print("="*100)
        return res

def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        util.raise_unknown('prediction_type', prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        r2 = skm.r2_score(y_true, y_pred)
        result = {'rmse': rmse, 'r2': r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        # zero_division=0으로 설정하여 UndefinedMetricWarning 억제
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True, zero_division=0)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
        
        # 2025.11.24 클래스 불균형 평가 지표 추가
        # Balanced Accuracy: 클래스 불균형에 강건한 정확도
        result['balanced_acc'] = skm.balanced_accuracy_score(y_true, labels)
        
        # Matthews Correlation Coefficient: 불균형 데이터에 좋은 지표
        result['mcc'] = skm.matthews_corrcoef(y_true, labels)
        
        # Cohen's Kappa: 일치도 측정
        result['kappa'] = skm.cohen_kappa_score(y_true, labels)
        
    return result
