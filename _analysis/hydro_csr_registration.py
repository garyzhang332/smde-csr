from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegistrationConfig:
    min_segment_points: int = 4
    min_component_segments: int = 3
    min_storage_span_mm: float = 0.08
    min_storage_overlap_mm: float = 0.12
    min_overlap_fraction: float = 0.25
    max_loss_relative_difference: float = 1.25
    loss_floor_mm_h: float = 0.02
    component_gap_h: float = 1.0


def segment_loss_rate(segment: pd.DataFrame) -> np.ndarray:
    t = segment["segment_t_h"].to_numpy(float)
    y = segment["moisture_mm"].to_numpy(float)
    if len(segment) < 2 or np.nanmax(t) <= np.nanmin(t):
        return np.full(len(segment), np.nan)
    loss = -np.gradient(y, t, edge_order=1)
    finite = loss[np.isfinite(loss)]
    if len(finite):
        hi = max(float(np.nanpercentile(finite, 95)), 0.0)
        loss = np.clip(loss, 0.0, hi)
    return loss


def _segment_object(segment_id: str, segment: pd.DataFrame, cfg: RegistrationConfig) -> dict[str, object] | None:
    segment = segment.sort_values("segment_t_h").copy()
    if len(segment) < cfg.min_segment_points:
        return None
    t = segment["segment_t_h"].to_numpy(float)
    y = segment["moisture_mm"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < cfg.min_segment_points:
        return None
    t = t[ok]
    y = y[ok]
    if np.nanmax(t) <= np.nanmin(t):
        return None
    storage_span = float(np.nanmax(y) - np.nanmin(y))
    if storage_span < cfg.min_storage_span_mm:
        return None
    segment = segment.loc[ok].copy()
    loss = segment_loss_rate(segment)
    return {
        "event_id": str(segment_id),
        "t": t,
        "s": y,
        "loss": loss,
        "min_s": float(np.nanmin(y)),
        "max_s": float(np.nanmax(y)),
        "span_s": storage_span,
        "duration_h": float(np.nanmax(t) - np.nanmin(t)),
        "start_s": float(y[0]),
        "end_s": float(y[-1]),
    }


def _interp_by_storage(obj: dict[str, object], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    frame = pd.DataFrame(
        {
            "storage": np.asarray(obj["s"], dtype=float),
            "t": np.asarray(obj["t"], dtype=float),
            "loss": np.asarray(obj["loss"], dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame["storage"].nunique() < 3:
        return None
    anchor = (
        frame.groupby("storage", as_index=False)
        .agg(t=("t", "median"), loss=("loss", "median"))
        .sort_values("storage")
    )
    return (
        np.interp(grid, anchor["storage"].to_numpy(float), anchor["t"].to_numpy(float)),
        np.interp(grid, anchor["storage"].to_numpy(float), anchor["loss"].to_numpy(float)),
    )


def _pairwise_edges(objects: list[dict[str, object]], cfg: RegistrationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(len(objects)):
        a = objects[i]
        for j in range(i + 1, len(objects)):
            b = objects[j]
            lo = max(float(a["min_s"]), float(b["min_s"]))
            hi = min(float(a["max_s"]), float(b["max_s"]))
            overlap = hi - lo
            min_span = max(min(float(a["span_s"]), float(b["span_s"])), 1e-9)
            overlap_fraction = overlap / min_span
            if overlap < cfg.min_storage_overlap_mm or overlap_fraction < cfg.min_overlap_fraction:
                continue
            grid = np.linspace(lo + 0.05 * overlap, hi - 0.05 * overlap, 17)
            ia = _interp_by_storage(a, grid)
            ib = _interp_by_storage(b, grid)
            if ia is None or ib is None:
                continue
            t_a, loss_a = ia
            t_b, loss_b = ib
            loss_scale = np.nanmedian((loss_a + loss_b) / 2.0) + cfg.loss_floor_mm_h
            if not np.isfinite(loss_scale) or loss_scale <= 0:
                continue
            loss_relative_difference = float(np.nanmedian(np.abs(loss_a - loss_b)) / loss_scale)
            if not np.isfinite(loss_relative_difference) or loss_relative_difference > cfg.max_loss_relative_difference:
                continue
            offset_j_minus_i = float(np.nanmedian(t_a - t_b))
            if not np.isfinite(offset_j_minus_i):
                continue
            weight = float(overlap_fraction / (1.0 + loss_relative_difference))
            rows.append(
                {
                    "event_i": str(a["event_id"]),
                    "event_j": str(b["event_id"]),
                    "offset_j_minus_i_h": offset_j_minus_i,
                    "storage_overlap_mm": float(overlap),
                    "overlap_fraction": float(overlap_fraction),
                    "loss_relative_difference": loss_relative_difference,
                    "edge_weight": weight,
                }
            )
    return pd.DataFrame(rows)


def _connected_components(nodes: list[str], edges: pd.DataFrame) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    if not edges.empty:
        for row in edges.itertuples(index=False):
            adjacency[str(row.event_i)].add(str(row.event_j))
            adjacency[str(row.event_j)].add(str(row.event_i))
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        comp: list[str] = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _solve_component_offsets(component: list[str], edges: pd.DataFrame) -> dict[str, float]:
    if len(component) == 1:
        return {component[0]: 0.0}
    node_index = {node: i for i, node in enumerate(component)}
    rows = []
    rhs = []
    weights = []
    use = edges[edges["event_i"].isin(component) & edges["event_j"].isin(component)]
    for row in use.itertuples(index=False):
        event_i = str(row.event_i)
        event_j = str(row.event_j)
        if event_i not in node_index or event_j not in node_index:
            continue
        a = np.zeros(len(component) - 1, dtype=float)
        if node_index[event_j] > 0:
            a[node_index[event_j] - 1] += 1.0
        if node_index[event_i] > 0:
            a[node_index[event_i] - 1] -= 1.0
        rows.append(a)
        rhs.append(float(row.offset_j_minus_i_h))
        weights.append(np.sqrt(max(float(row.edge_weight), 1e-6)))
    if not rows:
        return {node: 0.0 for node in component}
    design = np.vstack(rows)
    target = np.asarray(rhs, dtype=float)
    w = np.asarray(weights, dtype=float)
    solution, *_ = np.linalg.lstsq(design * w[:, None], target * w, rcond=None)
    offsets = {component[0]: 0.0}
    for node, idx in node_index.items():
        if idx == 0:
            continue
        offsets[node] = float(solution[idx - 1])
    return offsets


def hydrologically_constrained_registration(
    segment_points: pd.DataFrame,
    cfg: RegistrationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or RegistrationConfig()
    objects: list[dict[str, object]] = []
    for segment_id, segment in segment_points.groupby("event_id", sort=False):
        obj = _segment_object(str(segment_id), segment, cfg)
        if obj is not None:
            objects.append(obj)
    if len(objects) < cfg.min_component_segments:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    nodes = [str(obj["event_id"]) for obj in objects]
    edges = _pairwise_edges(objects, cfg)
    components = _connected_components(nodes, edges)
    object_by_id = {str(obj["event_id"]): obj for obj in objects}
    kept_components = [comp for comp in components if len(comp) >= cfg.min_component_segments]
    if not kept_components:
        return pd.DataFrame(), edges, pd.DataFrame()

    component_meta = []
    offsets: dict[str, tuple[int, float, float]] = {}
    display_offset = 0.0
    for comp_id, comp in enumerate(
        sorted(kept_components, key=lambda c: np.nanmedian([object_by_id[x]["start_s"] for x in c]), reverse=True),
        start=1,
    ):
        local_edges = edges[edges["event_i"].isin(comp) & edges["event_j"].isin(comp)].copy()
        solved = _solve_component_offsets(comp, local_edges)
        local_min = np.inf
        local_max = -np.inf
        for node in comp:
            obj = object_by_id[node]
            x = np.asarray(obj["t"], dtype=float) + solved.get(node, 0.0)
            local_min = min(local_min, float(np.nanmin(x)))
            local_max = max(local_max, float(np.nanmax(x)))
        local_span = max(local_max - local_min, 0.0)
        for node in comp:
            offsets[node] = (comp_id, solved.get(node, 0.0) - local_min + display_offset, display_offset)
        component_meta.append(
            {
                "registration_component": comp_id,
                "segments": len(comp),
                "pairwise_edges": int(len(local_edges)),
                "component_start_x_h": float(display_offset),
                "component_span_h": float(local_span),
                "median_start_mm": float(np.nanmedian([object_by_id[x]["start_s"] for x in comp])),
                "median_end_mm": float(np.nanmedian([object_by_id[x]["end_s"] for x in comp])),
            }
        )
        display_offset += local_span + cfg.component_gap_h

    aligned_parts = []
    for segment_id, segment in segment_points.groupby("event_id", sort=False):
        key = str(segment_id)
        if key not in offsets:
            continue
        component, offset, component_start = offsets[key]
        piece = segment.sort_values("segment_t_h").copy()
        piece["loss_rate_mm_h"] = segment_loss_rate(piece)
        piece["csr_x_h"] = piece["segment_t_h"].to_numpy(float) + offset
        piece["registration_component"] = int(component)
        piece["registration_component_start_x_h"] = float(component_start)
        aligned_parts.append(piece)
    aligned = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pd.DataFrame()
    components_out = pd.DataFrame(component_meta)
    if not edges.empty:
        component_lookup = {node: offsets[node][0] for node in offsets}
        edges["registration_component"] = edges["event_i"].map(component_lookup)
        edges = edges[edges["event_j"].map(component_lookup).eq(edges["registration_component"])].copy()
    return aligned, edges.reset_index(drop=True), components_out


def align_to_registered_template(
    test_points: pd.DataFrame,
    train_aligned: pd.DataFrame,
    cfg: RegistrationConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or RegistrationConfig()
    if test_points.empty or train_aligned.empty:
        return pd.DataFrame()
    template = train_aligned.dropna(subset=["csr_x_h", "moisture_mm"]).copy()
    if "loss_rate_mm_h" not in template.columns:
        template["loss_rate_mm_h"] = template.groupby("event_id", group_keys=False).apply(
            lambda g: pd.Series(segment_loss_rate(g.sort_values("segment_t_h")), index=g.sort_values("segment_t_h").index)
        )
    by_storage = (
        template[["moisture_mm", "csr_x_h", "loss_rate_mm_h"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("moisture_mm", as_index=False)
        .agg(csr_x_h=("csr_x_h", "median"), loss_rate_mm_h=("loss_rate_mm_h", "median"))
        .sort_values("moisture_mm")
    )
    if by_storage["moisture_mm"].nunique() < 4:
        return pd.DataFrame()
    anchor_s = by_storage["moisture_mm"].to_numpy(float)
    anchor_x = by_storage["csr_x_h"].to_numpy(float)
    anchor_l = by_storage["loss_rate_mm_h"].to_numpy(float)
    parts = []
    for segment_id, segment in test_points.groupby("event_id", sort=False):
        obj = _segment_object(str(segment_id), segment, cfg)
        if obj is None:
            continue
        lo = max(float(obj["min_s"]), float(np.nanmin(anchor_s)))
        hi = min(float(obj["max_s"]), float(np.nanmax(anchor_s)))
        overlap = hi - lo
        overlap_fraction = overlap / max(float(obj["span_s"]), 1e-9)
        if overlap < cfg.min_storage_overlap_mm or overlap_fraction < cfg.min_overlap_fraction:
            continue
        grid = np.linspace(lo + 0.05 * overlap, hi - 0.05 * overlap, 17)
        interp = _interp_by_storage(obj, grid)
        if interp is None:
            continue
        test_t, test_loss = interp
        template_x = np.interp(grid, anchor_s, anchor_x)
        template_loss = np.interp(grid, anchor_s, anchor_l)
        scale = np.nanmedian((test_loss + template_loss) / 2.0) + cfg.loss_floor_mm_h
        loss_relative_difference = float(np.nanmedian(np.abs(test_loss - template_loss)) / scale)
        if not np.isfinite(loss_relative_difference) or loss_relative_difference > cfg.max_loss_relative_difference:
            continue
        offset = float(np.nanmedian(template_x - test_t))
        piece = segment.sort_values("segment_t_h").copy()
        piece["loss_rate_mm_h"] = segment_loss_rate(piece)
        piece["csr_x_h"] = piece["segment_t_h"].to_numpy(float) + offset
        piece["registration_component"] = -1
        piece["template_storage_overlap_mm"] = float(overlap)
        piece["template_loss_relative_difference"] = loss_relative_difference
        parts.append(piece)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
