"""模型评估：逐图概率 → 各类 F1 / 宏平均 F1 + 患者整组 bootstrap 置信区间。

业务约束（与页面/服务层配合）：
- 真值只取当前已发布清单中验证侧图像的类别；
- 每个重采样单元是“患者”：同一患者的多张验证图整组一起进/出，
  避免同患者图像相关导致置信区间被低估（虚窄）；
- 每次计算（含每次 bootstrap 重采样）都沿用完整类别集合，
  F1 分母为 0 时该类别记 0；
- bootstrap 使用固定种子并随记录入库，同样输入可复现同样的区间。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20240601
CI_LOW_Q = 0.025
CI_HIGH_Q = 0.975


@dataclass(frozen=True)
class EvalRow:
    """一张验证图：真值类别 + 各类别预测概率。"""

    image_no: str
    patient_no: str
    truth: str
    probs: dict[str, float]


@dataclass(frozen=True)
class ClassMetric:
    class_name: str
    tp: int
    fp: int
    fn: int
    f1: float


@dataclass(frozen=True)
class PatientF1:
    """单个患者内部的逐图（含重复图）对宏平均 F1 的贡献。"""

    patient_no: str
    image_count: int
    tp: list[int]                     # 与 classes 同序
    fp: list[int]
    fn: list[int]


@dataclass(frozen=True)
class EvaluationResult:
    class_metrics: list[ClassMetric]
    macro_f1: float
    ci_lower: float
    ci_upper: float
    n_images: int
    n_patients: int
    bootstrap_iters: int
    seed: int
    patient_f1: list[PatientF1]


def predict_label(probs: dict[str, float], classes: list[str]) -> str:
    """概率最大者胜出；并列时取类别集合中靠前的类别（确定性）。

    缺失类别按 0 处理（解析层会拒绝缺列的预测 CSV，这里只是防御）。
    """
    best = classes[0]
    best_p = probs.get(best, 0.0)
    for cls in classes[1:]:
        p = probs.get(cls, 0.0)
        if p > best_p:
            best, best_p = cls, p
    return best


def _f1_from_counts(classes: list[str], tp: list[int], fp: list[int], fn: list[int]) -> tuple[list[float], float]:
    f1s: list[float] = []
    for k in range(len(classes)):
        denom = 2 * tp[k] + fp[k] + fn[k]
        f1s.append((2 * tp[k] / denom) if denom else 0.0)
    macro = sum(f1s) / len(classes) if classes else 0.0
    return f1s, macro


def _percentile(sorted_values: list[float], q: float) -> float:
    """线性插值分位数（numpy percentile 默认口径），确定性。"""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def evaluate(
    rows: list[EvalRow],
    classes: list[str],
    iterations: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> EvaluationResult:
    """计算指标与患者整组 bootstrap 95% 区间。

    rows 必须非空、覆盖且只覆盖验证侧图像；classes 为完整类别集合。
    重采样单位是患者：每次有放回地抽 n_patients 个患者，抽中患者的所有
    验证图整组进入样本，因此同患者多图不会被拆散、不会让区间虚窄。
    """
    if not rows:
        raise ValueError("评估样本为空")
    if not classes:
        raise ValueError("类别集合为空")
    if iterations <= 0:
        raise ValueError("bootstrap 次数必须为正")

    by_patient: dict[str, list[EvalRow]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_no, []).append(row)
    patients = sorted(by_patient)

    # 每个患者预先算好计数，重采样时只做向量相加，1000 次也很快。
    patient_counts: list[PatientF1] = []
    total_tp = [0] * len(classes)
    total_fp = [0] * len(classes)
    total_fn = [0] * len(classes)
    for patient in patients:
        tp = [0] * len(classes)
        fp = [0] * len(classes)
        fn = [0] * len(classes)
        for row in by_patient[patient]:
            truth_idx = classes.index(row.truth)
            pred_idx = classes.index(predict_label(row.probs, classes))
            if truth_idx == pred_idx:
                tp[truth_idx] += 1
            else:
                fp[pred_idx] += 1
                fn[truth_idx] += 1
        for k in range(len(classes)):
            total_tp[k] += tp[k]
            total_fp[k] += fp[k]
            total_fn[k] += fn[k]
        patient_counts.append(PatientF1(patient, len(by_patient[patient]), tp, fp, fn))

    f1s, macro = _f1_from_counts(classes, total_tp, total_fp, total_fn)
    class_metrics = [
        ClassMetric(classes[k], total_tp[k], total_fp[k], total_fn[k], f1s[k])
        for k in range(len(classes))
    ]

    rng = random.Random(seed)
    boot_macros: list[float] = []
    n_patients = len(patients)
    for _ in range(iterations):
        tp = [0] * len(classes)
        fp = [0] * len(classes)
        fn = [0] * len(classes)
        for idx in rng.choices(range(n_patients), k=n_patients):
            pc = patient_counts[idx]
            for k in range(len(classes)):
                tp[k] += pc.tp[k]
                fp[k] += pc.fp[k]
                fn[k] += pc.fn[k]
        # 每次重采样都沿用完整类别集合；某类分母为 0 时该类记 0。
        _, sampled_macro = _f1_from_counts(classes, tp, fp, fn)
        boot_macros.append(sampled_macro)

    boot_macros.sort()
    return EvaluationResult(
        class_metrics=class_metrics,
        macro_f1=macro,
        ci_lower=_percentile(boot_macros, CI_LOW_Q),
        ci_upper=_percentile(boot_macros, CI_HIGH_Q),
        n_images=len(rows),
        n_patients=n_patients,
        bootstrap_iters=iterations,
        seed=seed,
        patient_f1=patient_counts,
    )
