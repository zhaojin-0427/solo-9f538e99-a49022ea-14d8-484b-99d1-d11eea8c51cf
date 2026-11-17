"""CSV 解析、输入校验、并查集连通分量构建。

规则：
- 全量图像 CSV：图像号、患者号、类别（支持中英表头，也支持无表头按列顺序）。
- 重复图像关系 CSV：图像号 A、图像号 B。
- 同患者的图像天然连通；重复关系边把图像连通；连通分量是不可分割的分配单元。
- 限 40 张图、4 类。
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

from .db import MAX_CLASSES, MAX_IMAGES

IMAGE_HEADERS = [
    ("图像号", "患者号", "类别"),
    ("image_no", "patient_no", "class_name"),
    ("image", "patient", "class"),
    ("image_id", "patient_id", "class"),
]
DUP_HEADERS = [
    ("图像号a", "图像号b"),
    ("image_a", "image_b"),
    ("image1", "image2"),
    ("a", "b"),
]


class InputError(ValueError):
    """输入数据本身的问题（格式/越界/引用缺失等）。"""


@dataclass
class ImageRow:
    image_no: str
    patient_no: str
    class_name: str


@dataclass
class Component:
    cid: int
    image_nos: list[str] = field(default_factory=list)
    classes: set[str] = field(default_factory=set)
    patients: set[str] = field(default_factory=set)
    # 各类别图像数向量，例如 {"肿瘤": 2, "正常": 1}
    class_counts: dict[str, int] = field(default_factory=dict)


def _read_rows(text: str) -> list[list[str]]:
    if text is None or not text.strip():
        raise InputError("CSV 内容为空")
    # utf-8-sig 自动去掉 BOM
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    rows = [[cell.strip() for cell in row] for row in reader]
    rows = [r for r in rows if any(c for c in r)]  # 丢弃空行
    if not rows:
        raise InputError("CSV 没有有效行")
    return rows


def _match_header(header: list[str], candidates: list[tuple[str, ...]]) -> list[int] | None:
    low = [h.strip().lower() for h in header]
    for cand in candidates:
        idx = []
        ok = True
        for name in cand:
            if name.lower() in low:
                idx.append(low.index(name.lower()))
            else:
                ok = False
                break
        if ok:
            return idx
    return None


def parse_images_csv(text: str) -> list[ImageRow]:
    rows = _read_rows(text)
    col_idx = _match_header(rows[0], IMAGE_HEADERS)
    if col_idx is not None:
        data_rows = rows[1:]
    else:
        col_idx = [0, 1, 2]
        data_rows = rows
    out: list[ImageRow] = []
    seen: set[str] = set()
    for n, row in enumerate(data_rows, start=2 if col_idx else 1):
        if len(row) < 3:
            raise InputError(f"图像 CSV 第 {n} 行列数不足，需要 3 列（图像号、患者号、类别）")
        image_no, patient_no, class_name = row[0], row[1], row[2]
        if not image_no or not patient_no or not class_name:
            raise InputError(f"图像 CSV 第 {n} 行存在空值（图像号、患者号、类别均必填）")
        if image_no in seen:
            raise InputError(f"图像号重复：{image_no}")
        seen.add(image_no)
        out.append(ImageRow(image_no, patient_no, class_name))
    if not out:
        raise InputError("图像 CSV 没有数据行")
    return out


def parse_dups_csv(text: str) -> list[tuple[str, str]]:
    if text is None or not text.strip():
        return []
    rows = _read_rows(text)
    col_idx = _match_header(rows[0], DUP_HEADERS)
    if col_idx is not None:
        data_rows = rows[1:]
    else:
        col_idx = [0, 1]
        data_rows = rows
    edges: list[tuple[str, str]] = []
    for n, row in enumerate(data_rows, start=2 if col_idx else 1):
        if len(row) < 2:
            raise InputError(f"重复关系 CSV 第 {n} 行列数不足，需要 2 列（图像号 A、图像号 B）")
        a, b = row[0], row[1]
        if not a or not b:
            raise InputError(f"重复关系 CSV 第 {n} 行存在空值")
        edges.append((a, b))
    return edges


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def validate_and_build(
    images: list[ImageRow],
    dup_edges: list[tuple[str, str]],
) -> tuple[list[Component], list[str]]:
    """校验全局约束并构建连通分量。返回 (分量列表, 固定顺序的类别列表)。"""
    if len(images) > MAX_IMAGES:
        raise InputError(f"图像总数 {len(images)} 超过上限 {MAX_IMAGES} 张")
    classes = sorted({r.class_name for r in images})
    if len(classes) > MAX_CLASSES:
        raise InputError(f"类别数 {len(classes)} 超过上限 {MAX_CLASSES} 类")

    image_set = {r.image_no for r in images}
    for n, (a, b) in enumerate(dup_edges, start=1):
        if a not in image_set:
            raise InputError(f"重复关系第 {n} 行引用了不存在的图像号：{a}")
        if b not in image_set:
            raise InputError(f"重复关系第 {n} 行引用了不存在的图像号：{b}")

    uf = UnionFind(list(image_set))
    # 同患者连通
    by_patient: dict[str, list[str]] = {}
    for r in images:
        by_patient.setdefault(r.patient_no, []).append(r.image_no)
    for members in by_patient.values():
        first = members[0]
        for other in members[1:]:
            uf.union(first, other)
    # 重复关系连通
    for a, b in dup_edges:
        uf.union(a, b)

    groups: dict[str, Component] = {}
    for r in images:
        root = uf.find(r.image_no)
        comp = groups.get(root)
        if comp is None:
            comp = Component(cid=len(groups))
            groups[root] = comp
        comp.image_nos.append(r.image_no)
        comp.patients.add(r.patient_no)
        comp.classes.add(r.class_name)
        comp.class_counts[r.class_name] = comp.class_counts.get(r.class_name, 0) + 1

    components = sorted(groups.values(), key=lambda c: c.cid)
    # 重新按稳定顺序编号
    for i, comp in enumerate(components):
        comp.cid = i
        comp.image_nos.sort()
    return components, classes


def parse_quota(raw: dict[str, int], classes: list[str]) -> dict[str, int]:
    """校验配额：非负整数、不超过该类图像总数。"""
    quota: dict[str, int] = {}
    for cls in classes:
        val = raw.get(cls)
        if val is None or val == "":
            raise InputError(f"类别「{cls}」缺少验证张数")
        try:
            n = int(val)
        except (TypeError, ValueError):
            raise InputError(f"类别「{cls}」的验证张数必须是整数，收到：{val!r}")
        if n < 0:
            raise InputError(f"类别「{cls}」的验证张数不能为负数")
        quota[cls] = n
    return quota


# ------------------------------------------------------------------ 预测 CSV
PROB_HEADERS = [
    ("图像号",),
    ("image_no",),
    ("image",),
    ("image_id",),
    ("图像",),
]
PROB_SUM_TOLERANCE = 1e-6


@dataclass
class PredictionRow:
    image_no: str
    probs: dict[str, float]


def parse_predictions_csv(
    text: str,
    classes: list[str],
    val_images: set[str],
) -> list[PredictionRow]:
    """解析逐图概率 CSV，整份校验，任何一行不合格即整体拒绝。

    格式：首行表头必须含图像号列与当前输入版本的每一个类别列（列名完全一致），
    不接受多余列。每行概率必须为有限实数且落在 [0,1]，同一行概率和与 1 的
    偏差不得超过 1e-6。数据行必须与验证清单图像恰好一一对应：
    不得重复、不得缺少、不得包含训练侧或清单外图像。
    """
    if not classes:
        raise InputError("当前输入版本没有类别，无法评估")
    if not val_images:
        raise InputError("当前验证清单为空（没有验证侧图像），无法评估分类效果")

    rows = _read_rows(text)
    header = rows[0]
    if len({h.lower() for h in header}) != len(header):
        raise InputError("预测 CSV 表头存在重复列名")

    image_col = None
    for cand in PROB_HEADERS:
        hits = [i for i, h in enumerate(header) if h.lower() == cand[0].lower()]
        if hits:
            image_col = hits[0]
            break
    if image_col is None:
        raise InputError("预测 CSV 缺少图像号列（可用表头：图像号 / image_no）")

    class_cols: dict[str, int] = {}
    for cls in classes:
        hits = [i for i, h in enumerate(header) if h == cls]
        if not hits:
            raise InputError(
                f"预测 CSV 缺少类别列「{cls}」；类别列必须与当前输入版本完全一致："
                f"{'、'.join(classes)}"
            )
        class_cols[cls] = hits[0]

    expected = {image_col, *class_cols.values()}
    extras = [h for i, h in enumerate(header) if i not in expected]
    if extras:
        raise InputError(
            f"预测 CSV 存在不属于当前类别的多余列：{'、'.join(extras)}；"
            f"类别列必须与当前输入版本完全一致：{'、'.join(classes)}"
        )

    out: list[PredictionRow] = []
    seen: set[str] = set()
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise InputError(
                f"预测 CSV 第 {n} 行有 {len(row)} 列，与表头 {len(header)} 列不一致"
            )
        image_no = row[image_col]
        if not image_no:
            raise InputError(f"预测 CSV 第 {n} 行图像号为空")
        if image_no in seen:
            raise InputError(f"预测 CSV 图像号重复：{image_no}")
        if image_no not in val_images:
            # 训练侧图像与清单外图像都会走到这里，分别给清晰说明
            raise InputError(
                f"预测 CSV 第 {n} 行的图像 {image_no} 不在当前验证清单中"
                "（训练侧图像或已过期/清单外图像不得提交）"
            )
        seen.add(image_no)
        probs: dict[str, float] = {}
        for cls, idx in class_cols.items():
            raw = row[idx]
            try:
                p = float(raw)
            except (TypeError, ValueError):
                raise InputError(
                    f"预测 CSV 第 {n} 行类别「{cls}」的概率不是数字：{raw!r}"
                )
            if not math.isfinite(p):
                raise InputError(
                    f"预测 CSV 第 {n} 行类别「{cls}」的概率不是有限值：{raw!r}"
                )
            if p < 0.0 or p > 1.0:
                raise InputError(
                    f"预测 CSV 第 {n} 行类别「{cls}」的概率 {p} 超出 [0,1] 范围"
                )
            probs[cls] = p
        total = sum(probs.values())
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise InputError(
                f"预测 CSV 第 {n} 行概率和为 {total:.12g}，偏离 1 超过 "
                f"{PROB_SUM_TOLERANCE:g}"
            )
        out.append(PredictionRow(image_no=image_no, probs=probs))

    if not out:
        raise InputError("预测 CSV 没有数据行")
    missing = sorted(val_images - seen)
    if missing:
        preview = "、".join(missing[:5]) + ("…" if len(missing) > 5 else "")
        raise InputError(
            f"预测 CSV 未覆盖全部验证图像，缺少 {len(missing)} 张：{preview}"
        )
    return out
