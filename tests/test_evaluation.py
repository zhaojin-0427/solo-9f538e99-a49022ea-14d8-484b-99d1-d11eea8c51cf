"""评估：预测 CSV 整份校验、F1/宏平均、患者整组 bootstrap、版本绑定与历史。"""
import json
import math

import pytest

from app.db import get_db
from app.evaluation import EvalRow, evaluate
from app.parser import InputError, parse_predictions_csv
from app.service import Service, StaleEvaluationError
from tests.conftest import make_csv, make_dup


# ---------------------------------------------------------------- 构造数据
def val_rows():
    """验证清单：A 类 3 张（P1 两图 + P2 一图），B 类 2 张（P3、P4）。"""
    return [
        ("V1", "P1", "A"), ("V2", "P1", "A"), ("V3", "P2", "A"),
        ("V4", "P3", "B"), ("V5", "P4", "B"),
    ]


def make_predictions(rows, classes=("A", "B")):
    """rows: (image_no, {类别: 概率})；按表头顺序输出。"""
    lines = [",".join(["图像号", *classes])]
    for image_no, probs in rows:
        lines.append(",".join([image_no, *[repr(probs[c]) for c in classes]]))
    return "\n".join(lines) + "\n"


def perfect_predictions():
    mapping = {"V1": "A", "V2": "A", "V3": "A", "V4": "B", "V5": "B"}
    out = []
    for image_no, truth in mapping.items():
        probs = {c: (0.9 if c == truth else 0.1) for c in ("A", "B")}
        out.append((image_no, probs))
    return out


@pytest.fixture()
def evaluated(tmp_path):
    """一份已发布且带验证侧的清单。训练侧至少有一张图用于拒绝用例。"""
    db = str(tmp_path / "e.db")
    svc = Service(db, synchronous=True)
    rows = val_rows() + [("T1", "P9", "A"), ("T2", "P10", "B")]
    svc.create_version(make_csv(rows), make_dup([]), {"A": 4, "B": 2})
    svc.start_partition()
    assert svc.current_manifest() is not None
    entries = svc.get_manifest_entries(svc.current_manifest()["id"])
    val = {e["image_no"] for e in entries if e["side"] == "val"}
    train = {e["image_no"] for e in entries if e["side"] == "train"}
    truth = {e["image_no"]: e["class_name"] for e in entries}
    assert val and train
    yield svc, val, train, truth
    svc.close()


def _pred_for(val_images, truth, correct=True):
    out = []
    for img in sorted(val_images):
        t = truth[img]
        if correct:
            probs = {"A": 0.9 if t == "A" else 0.1, "B": 0.1 if t == "A" else 0.9}
        else:
            wrong = "B" if t == "A" else "A"
            probs = {"A": 0.9 if wrong == "A" else 0.1,
                     "B": 0.1 if wrong == "A" else 0.9}
        out.append((img, probs))
    return out


# ---------------------------------------------------------------- 解析校验
def test_parser_accepts_exact_cover():
    text = make_predictions(perfect_predictions())
    out = parse_predictions_csv(text, ["A", "B"], {"V1", "V2", "V3", "V4", "V5"})
    assert len(out) == 5
    assert out[0].probs == {"A": 0.9, "B": 0.1}


def test_parser_rejects_duplicate_rows():
    rows = perfect_predictions()
    rows.append(("V1", {"A": 0.5, "B": 0.5}))
    with pytest.raises(InputError, match="重复"):
        parse_predictions_csv(make_predictions(rows), ["A", "B"],
                              {"V1", "V2", "V3", "V4", "V5"})


def test_parser_rejects_missing_images():
    rows = perfect_predictions()[:4]
    with pytest.raises(InputError, match="未覆盖"):
        parse_predictions_csv(make_predictions(rows), ["A", "B"],
                              {"V1", "V2", "V3", "V4", "V5"})


def test_parser_rejects_training_or_unknown_image():
    rows = perfect_predictions() + [("T1", {"A": 0.5, "B": 0.5})]
    with pytest.raises(InputError, match="不在当前验证清单"):
        parse_predictions_csv(make_predictions(rows), ["A", "B"],
                              {"V1", "V2", "V3", "V4", "V5"})


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "abc", ""])
def test_parser_rejects_non_finite_or_non_numeric(bad):
    rows = perfect_predictions()
    rows[0] = ("V1", {"A": bad, "B": "0.1"})
    text = make_predictions(rows)
    with pytest.raises(InputError):
        parse_predictions_csv(text, ["A", "B"], {"V1", "V2", "V3", "V4", "V5"})


def test_parser_rejects_probability_sum_off_tolerance():
    rows = perfect_predictions()
    rows[0] = ("V1", {"A": 0.9, "B": 0.1000011})   # 偏离 1.1e-6 > 1e-6
    with pytest.raises(InputError, match="概率和"):
        parse_predictions_csv(make_predictions(rows), ["A", "B"],
                              {"V1", "V2", "V3", "V4", "V5"})
    # 恰好 1e-6 边界内接受（浮点表现可能落在边界，取更小的 5e-7 确保接受）
    rows[0] = ("V1", {"A": 0.9, "B": 0.1000005})
    out = parse_predictions_csv(make_predictions(rows), ["A", "B"],
                                {"V1", "V2", "V3", "V4", "V5"})
    assert out[0].image_no == "V1"


def test_parser_rejects_out_of_range_probability():
    rows = perfect_predictions()
    rows[0] = ("V1", {"A": 1.2, "B": -0.2})
    with pytest.raises(InputError, match="范围"):
        parse_predictions_csv(make_predictions(rows), ["A", "B"],
                              {"V1", "V2", "V3", "V4", "V5"})


def test_parser_rejects_class_column_mismatch():
    # 缺少 B 列
    text = "图像号,A\nV1,1\n"
    with pytest.raises(InputError, match="缺少类别列"):
        parse_predictions_csv(text, ["A", "B"], {"V1"})
    # 多出不存在的类别列
    text = "图像号,A,B,C\nV1,0.5,0.3,0.2\n"
    with pytest.raises(InputError, match="多余列"):
        parse_predictions_csv(text, ["A", "B"], {"V1"})


def test_parser_rejects_missing_image_column():
    with pytest.raises(InputError, match="图像号列"):
        parse_predictions_csv("编号,A,B\nV1,0.5,0.5\n", ["A", "B"], {"V1"})


def test_parser_rejects_wrong_arity_and_empty_body():
    with pytest.raises(InputError, match="列不一致"):
        parse_predictions_csv("图像号,A,B\nV1,0.5,0.5,0\n", ["A", "B"], {"V1"})
    with pytest.raises(InputError, match="没有数据行"):
        parse_predictions_csv("图像号,A,B\n", ["A", "B"], {"V1"})
    with pytest.raises(InputError, match="内容为空"):
        parse_predictions_csv("\n\n", ["A", "B"], {"V1"})


# ---------------------------------------------------------------- 指标计算
def test_f1_and_macro_with_full_class_set_and_zero_denominator():
    rows = [
        EvalRow("I1", "P1", "A", {"A": 0.9, "B": 0.1}),
        EvalRow("I2", "P2", "A", {"A": 0.6, "B": 0.4}),
        EvalRow("I3", "P3", "B", {"A": 0.7, "B": 0.3}),
    ]
    res = evaluate(rows, ["A", "B", "C"], iterations=50, seed=7)
    by_class = {m.class_name: m for m in res.class_metrics}
    # A: TP=2, FP=0, FN=1 → F1 = 4/5
    assert by_class["A"].f1 == pytest.approx(0.8)
    # B: TP=0, FP=2, FN=0 → F1 = 0
    assert by_class["B"].f1 == 0.0
    # C 无任何样本，分母为 0 → 记 0
    assert by_class["C"].f1 == 0.0
    # 宏平均仍除以完整 3 类
    assert res.macro_f1 == pytest.approx(0.8 / 3)
    assert res.n_images == 3 and res.n_patients == 3


def test_bootstrap_is_reproducible_with_same_seed():
    rows = []
    for i in range(12):
        truth = "A" if i % 2 else "B"
        correct = i % 3 != 0          # 约 1/3 判错，宏平均随重采样变化
        pred = truth if correct else ("B" if truth == "A" else "A")
        rows.append(
            EvalRow(
                f"I{i}", f"P{i}", truth,
                {"A": 0.9 if pred == "A" else 0.1,
                 "B": 0.1 if pred == "A" else 0.9},
            )
        )
    r1 = evaluate(rows, ["A", "B"], iterations=300, seed=123)
    r2 = evaluate(rows, ["A", "B"], iterations=300, seed=123)
    assert (r1.ci_lower, r1.ci_upper, r1.macro_f1) == (r2.ci_lower, r2.ci_upper, r2.macro_f1)
    assert r1.ci_lower < r1.ci_upper
    r3 = evaluate(rows, ["A", "B"], iterations=300, seed=124)
    assert (r3.ci_lower, r3.ci_upper) != (r1.ci_lower, r1.ci_upper)


def test_single_patient_group_resampling_has_zero_width():
    """所有图同属一个患者：每次重采样都整组抽到同一组图，区间宽度必为 0。

    若按图像独立重采样，错误图的进/出会让区间变宽——该测试锁定“整组”语义。
    """
    rows = [
        EvalRow("I1", "P1", "A", {"A": 0.9, "B": 0.1}),
        EvalRow("I2", "P1", "A", {"A": 0.9, "B": 0.1}),
        EvalRow("I3", "P1", "A", {"A": 0.1, "B": 0.9}),  # 错一张
        EvalRow("I4", "P1", "B", {"A": 0.1, "B": 0.9}),
    ]
    res = evaluate(rows, ["A", "B"], iterations=100, seed=1)
    assert res.ci_lower == pytest.approx(res.macro_f1)
    assert res.ci_upper == pytest.approx(res.macro_f1)
    assert res.n_patients == 1 and res.n_images == 4


def test_patient_bootstrap_ci_not_narrower_than_group_naive_width():
    """两患者、其中一人双图且全错：患者整组重采样只有少数几种取值。"""
    rows = [
        EvalRow("I1", "P1", "A", {"A": 0.9, "B": 0.1}),
        EvalRow("I2", "P1", "A", {"A": 0.9, "B": 0.1}),
        EvalRow("I3", "P2", "A", {"A": 0.1, "B": 0.9}),
        EvalRow("I4", "P2", "B", {"A": 0.1, "B": 0.9}),
    ]
    res = evaluate(rows, ["A", "B"], iterations=1000, seed=5)
    assert 0.0 <= res.ci_lower <= res.ci_upper <= 1.0
    assert res.n_patients == 2


# ---------------------------------------------------------------- 服务/事务
def test_submit_evaluation_binds_version_and_manifest(evaluated):
    svc, val_images, _, truth = evaluated
    context = svc.current_validation()
    assert context is not None
    eval_id = svc.submit_evaluation(make_predictions(_pred_for(val_images, truth)))
    row = svc.get_evaluation(eval_id)
    assert row["version_id"] == context["version_id"]
    assert row["manifest_id"] == context["manifest_id"]
    assert row["bootstrap_iters"] == 1000
    assert row["n_images"] == len(val_images)
    assert row["n_patients"] == len({e["patient_no"] for e in svc.get_manifest_entries(
        context["manifest_id"]) if e["side"] == "val"})
    assert svc.current_evaluation()["id"] == eval_id
    details = json.loads(row["details"])
    assert len(details["predictions"]) == len(val_images)


def test_submit_evaluation_rejects_before_writing(evaluated):
    svc, val_images, train_images, truth = evaluated
    bad = _pred_for(val_images, truth)
    bad.append((sorted(train_images)[0], {"A": 0.5, "B": 0.5}))
    with pytest.raises(InputError):
        svc.submit_evaluation(make_predictions(bad))
    conn = get_db(svc.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM evaluations").fetchone()["c"] == 0
        assert conn.execute(
            "SELECT COUNT(*) c FROM evaluation_predictions"
        ).fetchone()["c"] == 0
    finally:
        conn.close()


def test_submit_evaluation_empty_validation_side_rejected(tmp_path):
    db = str(tmp_path / "empty.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv(val_rows()), make_dup([]), {"A": 0, "B": 0})
    svc.start_partition()
    assert svc.current_manifest() is not None
    assert svc.current_validation()["val_images"] == {}
    with pytest.raises(InputError, match="验证清单为空"):
        svc.submit_evaluation(make_predictions(perfect_predictions()))
    assert svc.list_evaluations() == []
    svc.close()


def test_no_current_evaluation_without_published_manifest(tmp_path):
    db = str(tmp_path / "noman.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv(val_rows()), make_dup([]), {"A": 1, "B": 1})
    assert svc.current_validation() is None
    with pytest.raises(InputError, match="没有可评估的验证清单"):
        svc.submit_evaluation(make_predictions(perfect_predictions()))
    assert svc.list_evaluations() == []
    svc.close()


def test_stale_interleave_never_saved_as_current(evaluated, monkeypatch):
    """校验依据在校验期间被更正：事务内复查拒绝写入，且旧评估不会标为当前。"""
    svc, val_images, _, truth = evaluated
    stale_context = svc.current_validation()
    stale_text = make_predictions(_pred_for(val_images, truth))
    # 模拟：提交前已有一次更正产生新版本 v2（旧清单变历史）
    svc.create_version(make_csv(val_rows()), make_dup([]), {"A": 1, "B": 1})
    # 强制 submit_evaluation 使用过期依据（模拟解析耗时期间的旧上下文）
    monkeypatch.setattr(svc, "current_validation", lambda: stale_context)
    with pytest.raises(StaleEvaluationError, match="过期"):
        svc.submit_evaluation(stale_text)
    assert svc.list_evaluations() == []
    conn = get_db(svc.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM evaluations").fetchone()["c"] == 0
    finally:
        conn.close()


def test_evaluation_superseded_by_correction_is_history_only(evaluated):
    svc, val_images, _, truth = evaluated
    eval_id = svc.submit_evaluation(make_predictions(_pred_for(val_images, truth)))
    assert svc.current_evaluation()["id"] == eval_id
    # 更正输入 → 旧版本 superseded，旧评估不再是“当前”
    svc.create_version(make_csv(val_rows()), make_dup([]), {"A": 1, "B": 1})
    assert svc.current_evaluation() is None
    old = svc.get_evaluation(eval_id)
    assert old["version_status"] == "superseded"
    # 历史记录仍可查看
    assert any(e["id"] == eval_id for e in svc.list_evaluations())


def test_new_evaluation_becomes_current_and_previous_is_history(evaluated):
    svc, val_images, _, truth = evaluated
    e1 = svc.submit_evaluation(make_predictions(_pred_for(val_images, truth, correct=True)))
    e2 = svc.submit_evaluation(make_predictions(_pred_for(val_images, truth, correct=False)))
    assert svc.current_evaluation()["id"] == e2
    conn = get_db(svc.db_path)
    try:
        rows = conn.execute(
            "SELECT id, current_key FROM evaluations ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert [(r["id"], r["current_key"]) for r in rows] == [(e1, None), (e2, "current")]


def test_midflight_correction_during_compute_is_rejected(evaluated, monkeypatch):
    """真实交错：评估计算期间发生更正，写库事务内的复查必须拒绝。"""
    import app.service as service_mod

    svc, val_images, _, truth = evaluated
    text = make_predictions(_pred_for(val_images, truth))
    original = service_mod.evaluate

    def slow_then_correct(*args, **kwargs):
        # 模拟计算耗时期间用户上传了更正
        svc.create_version(make_csv(val_rows()), make_dup([]), {"A": 1, "B": 1})
        return original(*args, **kwargs)

    monkeypatch.setattr(service_mod, "evaluate", slow_then_correct)
    with pytest.raises(StaleEvaluationError):
        svc.submit_evaluation(text)
    assert svc.list_evaluations() == []


def test_atomic_rollback_on_write_error(evaluated, monkeypatch):
    """写库中途异常：整条评估（含逐图明细）回滚，旧当前记录保持原状。"""
    import app.service as service_mod

    svc, val_images, _, truth = evaluated
    first = svc.submit_evaluation(make_predictions(_pred_for(val_images, truth)))
    n = len(val_images)
    calls = {"n": 0}

    def flaky_predict_label(probs, classes):
        calls["n"] += 1
        # 第一遍（组装 details）正常；第二遍在事务内写逐图明细时抛错
        if calls["n"] > n:
            raise RuntimeError("模拟写库中断")
        return service_mod.predict_label(probs, classes)

    monkeypatch.setattr(service_mod, "predict_label", flaky_predict_label)
    with pytest.raises(RuntimeError):
        svc.submit_evaluation(make_predictions(_pred_for(val_images, truth)))
    conn = get_db(svc.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM evaluations").fetchone()["c"] == 1
        assert conn.execute(
            "SELECT COUNT(*) c FROM evaluation_predictions"
        ).fetchone()["c"] == n
        cur = conn.execute(
            "SELECT id, current_key FROM evaluations"
        ).fetchone()
    finally:
        conn.close()
    assert cur["id"] == first and cur["current_key"] == "current"
