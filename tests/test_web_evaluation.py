"""网页端评估流程：上传、查看、拒绝路径、更正后历史。"""
from io import BytesIO

import pytest

from app.db import get_db
from app.web import create_app
from tests.conftest import make_csv, make_dup, upload


@pytest.fixture()
def client(tmp_db, monkeypatch):
    """模块内同步执行划分，避免依赖外部 PYX_SYNCHRONOUS 环境变量。"""
    monkeypatch.setenv("PYX_SYNCHRONOUS", "1")
    app = create_app(tmp_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, app.extensions["service"]
    app.extensions["service"].close()


def rows_basic():
    return [
        ("I1", "P1", "A"), ("I2", "P2", "A"), ("I3", "P3", "A"),
        ("I4", "P4", "B"), ("I5", "P5", "B"), ("I6", "P6", "B"),
    ]


def setup_published(client, quota=None):
    c, svc = client
    upload(c, rows_basic(), quota or {"A": 2, "B": 1})
    c.post("/partition", follow_redirects=True)
    manifest = svc.current_manifest()
    assert manifest is not None
    entries = svc.get_manifest_entries(manifest["id"])
    val = {e["image_no"]: e["class_name"] for e in entries if e["side"] == "val"}
    return c, svc, manifest, val


def pred_csv(val, reverse=False):
    lines = ["图像号,A,B"]
    for image_no, truth in sorted(val.items()):
        if reverse:
            predicted = "B" if truth == "A" else "A"
        else:
            predicted = truth
        a = 0.9 if predicted == "A" else 0.1
        lines.append(f"{image_no},{a},{1 - a}")
    return "\n".join(lines) + "\n"


def post_pred(client, text):
    return client.post(
        "/evaluations/upload",
        data={"prediction_csv": (BytesIO(text.encode("utf-8")), "pred.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def test_web_upload_and_view_evaluation(client):
    c, svc, manifest, val = setup_published(client)
    assert len(val) == 3
    resp = post_pred(c, pred_csv(val))
    body = resp.data.decode("utf-8")
    assert resp.status_code == 200
    assert "评估" in body and "宏平均 F1" in body and "95%" in body
    assert "1.0000" in body   # 全部预测正确 → 各类 F1 与宏平均均为 1
    current = svc.current_evaluation()
    assert current is not None
    assert current["manifest_id"] == manifest["id"]
    # 首页展示当前评估摘要
    home = c.get("/").data.decode("utf-8")
    assert f"当前评估 #{current['id']}" in home
    # 评估列表页
    evals = c.get("/evaluations").data.decode("utf-8")
    assert f"#{current['id']}" in evals
    # 历史页
    hist = c.get("/history").data.decode("utf-8")
    assert "模型评估历史" in hist


def test_web_rejects_bad_prediction_without_record(client):
    c, svc, _, val = setup_published(client)
    bad = pred_csv(val) + "TRAINX,0.5,0.5\n"
    resp = post_pred(c, bad)
    page = resp.data.decode("utf-8")
    assert "整份拒绝" in page
    assert svc.list_evaluations() == []
    conn = get_db(svc.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM evaluations").fetchone()["c"] == 0
    finally:
        conn.close()


def test_web_correction_marks_old_evaluation_history(client):
    c, svc, _, val = setup_published(client)
    post_pred(c, pred_csv(val))
    first = svc.current_evaluation()
    # 更正输入（新版本、新图像）→ 旧评估仅历史
    upload(c, [("J1", "Q1", "A"), ("J2", "Q2", "A"), ("J3", "Q3", "B")],
           {"A": 1, "B": 1})
    assert svc.current_evaluation() is None
    page = c.get(f"/evaluations/{first['id']}").data.decode("utf-8")
    assert "历史" in page and "仅作历史查看" in page
    home = c.get("/").data.decode("utf-8")
    assert "当前评估" not in home
    # 新版本划分后重新评估 → 新记录为当前，旧记录仍历史
    c.post("/partition", follow_redirects=True)
    val2 = {
        e["image_no"]: e["class_name"]
        for e in svc.get_manifest_entries(svc.current_manifest()["id"])
        if e["side"] == "val"
    }
    post_pred(c, pred_csv(val2))
    second = svc.current_evaluation()
    assert second is not None and second["id"] != first["id"]
    old_page = c.get(f"/evaluations/{first['id']}").data.decode("utf-8")
    assert "仅作历史查看" in old_page
    evals = c.get("/evaluations").data.decode("utf-8")
    assert "当前" in evals and "历史" in evals


def test_web_upload_without_manifest_shows_clear_error(client):
    c, svc = client
    upload(c, rows_basic(), {"A": 2, "B": 1})   # 未划分，无清单
    resp = post_pred(c, "图像号,A,B\nI1,0.5,0.5\n")
    page = resp.data.decode("utf-8")
    assert "没有可评估的验证清单" in page
    assert svc.list_evaluations() == []


def test_web_empty_validation_side_rejected(client):
    c, svc = client
    upload(c, rows_basic(), {"A": 0, "B": 0})
    c.post("/partition", follow_redirects=True)
    assert svc.current_manifest() is not None
    resp = post_pred(c, "图像号,A,B\nI1,0.5,0.5\n")
    assert "验证清单为空" in resp.data.decode("utf-8")
    assert svc.list_evaluations() == []


def test_web_evaluation_detail_404(client):
    c, _ = client
    assert c.get("/evaluations/999").status_code == 404
