"""Landmark validation and CSV/legacy .points interchange.

All pairs are ordered (fixed_xy, cleared_xy), both in *working image* pixels.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import cv2


def validate_pairs(fixed: np.ndarray, cleared: np.ndarray,
                   fixed_size: tuple[int, int] | None = None,
                   cleared_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(fixed, dtype=np.float64), np.asarray(cleared, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1] != 2:
        raise ValueError("Landmarks must have matching N x 2 coordinate arrays.")
    if len(a) < 4:
        raise ValueError("At least four corresponding landmarks are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Landmark coordinates must be finite.")
    for label, points, size in (("FIXED", a, fixed_size), ("CLEARED", b, cleared_size)):
        if size is not None:
            w, h = size
            if np.any(points[:, 0] < 0) or np.any(points[:, 1] < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
                raise ValueError(f"{label} landmark outside image bounds ({w} x {h}).")
        diffs = points[:, None] - points[None, :]
        distances2 = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(distances2, np.inf)
        if np.min(distances2) < 1e-8:
            raise ValueError(f"Duplicate or nearly duplicate {label} landmarks.")
        hull = cv2.convexHull(points.astype(np.float32))
        area = abs(cv2.contourArea(hull))
        box_span = np.ptp(points, axis=0)
        if area < 1e-5 * float(np.sum(box_span**2)):
            raise ValueError(f"{label} landmarks are collinear or nearly collinear.")
    return a, b


def write_csv(path: str | Path, fixed: np.ndarray, cleared: np.ndarray) -> None:
    fixed, cleared = validate_pairs(fixed, cleared)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fixed_x", "fixed_y", "cleared_x", "cleared_y"])
        for p, q in zip(fixed, cleared):
            writer.writerow([f"{p[0]:.9g}", f"{p[1]:.9g}", f"{q[0]:.9g}", f"{q[1]:.9g}"])


def read_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or not any(col.strip() for col in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) == 1 and ";" in row[0]:
                row = row[0].split(";")
            if row[0].strip().lower() == "fixed_x":
                if [x.strip().lower() for x in row] != ["fixed_x", "fixed_y", "cleared_x", "cleared_y"]:
                    raise ValueError("Unexpected CSV header.")
                continue
            if len(row) != 4:
                raise ValueError(f"Line {line_number}: expected four columns.")
            try:
                rows.append([float(x.strip()) for x in row])
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: invalid numeric coordinate.") from exc
    array = np.asarray(rows, dtype=np.float64).reshape((-1, 4))
    return validate_pairs(array[:, :2], array[:, 2:])


def read_legacy_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Historical layout: cleared coordinates, fixed coordinates, then two GUI-label groups.

    The image-space coordinate groups are used; label-space groups are ignored.
    """
    tokens = Path(path).read_text(encoding="utf-8-sig").split()
    index = 0
    groups = []
    for label in ("cleared image", "fixed image", "cleared GUI label", "fixed GUI label"):
        try:
            n = int(tokens[index]); index += 1
            if n < 0 or n > 100000:
                raise ValueError("count outside limits")
            vals = [float(v) for v in tokens[index:index + 2*n]]
            if len(vals) != 2*n:
                raise ValueError("truncated data")
            index += 2*n
            groups.append(np.array(vals, dtype=np.float64).reshape((-1, 2)))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid legacy .points section: {label}.") from exc
    if index != len(tokens) or len(set(map(len, groups))) != 1:
        raise ValueError("Unexpected or inconsistent data in legacy .points file.")
    return validate_pairs(groups[1], groups[0])


def read_landmarks(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return read_legacy_points(path) if Path(path).suffix.lower() == ".points" else read_csv(path)
