"""Numerical thin-plate spline registration and research output generation.

The TPS basis U(r) = r^2 log(r^2) follows the classic 2D TPS family used by
OpenCV's ThinPlateSplineShapeTransformer. This implementation uses a NumPy
linear solver rather than OpenCV's C++ estimator: compare with experimental
reference outputs before claiming numerical identity with the legacy program.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import platform
import sys
from typing import Callable

import cv2
import numpy as np

from . import __version__
from .landmarks import validate_pairs, write_csv

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path: str | Path) -> np.ndarray:
    # imdecode handles Unicode paths on Windows more reliably than cv2.imread.
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"Empty or missing image file: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError(f"Cannot read supported 1/3/4-channel image: {path}")
    if image.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
        raise ValueError(f"Unsupported image dtype {image.dtype}: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    suffix = Path(path).suffix.lower()
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("Downsample factor must be an integer >= 1.")
    if factor == 1:
        return image.copy()
    width = image.shape[1] // factor
    height = int(width * (image.shape[0] / image.shape[1]))
    if width < 2 or height < 2:
        raise ValueError("Downsample factor too large for this image.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def prepare_images(fixed: np.ndarray, cleared: np.ndarray, factor: int = 1,
                   match_fixed_width: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if match_fixed_width:
        w = cleared.shape[1]
        h = max(1, int(w * fixed.shape[0] / fixed.shape[1]))
        fixed = cv2.resize(fixed, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return downsample(fixed, factor), downsample(cleared, factor)


def _kernel(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    diff = points_a[:, None, :] - points_b[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    out = np.zeros_like(r2, dtype=np.float64)
    np.log(r2, out=out, where=r2 > 0)
    out *= r2
    return out


@dataclass(frozen=True)
class TPS:
    control: np.ndarray
    weights: np.ndarray
    affine: np.ndarray
    beta: float

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, beta: float = 0.0) -> "TPS":
        source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
            raise ValueError("TPS requires >=4 paired Nx2 landmark arrays.")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite landmark coordinates.")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("TPS regularization must be nonnegative and finite.")
        n = len(source)
        p = np.column_stack((np.ones(n), source))
        matrix = np.zeros((n + 3, n + 3), dtype=np.float64)
        matrix[:n, :n] = _kernel(source, source) + beta * np.eye(n)
        matrix[:n, n:] = p
        matrix[n:, :n] = p.T
        target_aug = np.vstack((target, np.zeros((3, 2))))
        try:
            coefficients = np.linalg.solve(matrix, target_aug)
        except np.linalg.LinAlgError as exc:
            raise ValueError("TPS solve failed: check landmark distribution and duplicates.") from exc
        return cls(source.copy(), coefficients[:n].copy(), coefficients[n:].copy(), beta)

    def transform(self, points: np.ndarray, chunk: int = 15000) -> np.ndarray:
        coords = np.asarray(points, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Transform coordinates must be Nx2.")
        out = np.empty((len(coords), 2), dtype=np.float64)
        for i in range(0, len(coords), chunk):
            q = coords[i:i+chunk]
            out[i:i+chunk] = (np.column_stack((np.ones(len(q)), q)) @ self.affine
                              + _kernel(q, self.control) @ self.weights)
        if not np.isfinite(out).all():
            raise ValueError("TPS produced nonfinite coordinates.")
        return out


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    corners: np.ndarray  # 4 x 2 points in FIXED space, clockwise
    stretch_x: float
    stretch_y: float


def grid_cells(forward: TPS, size_cleared: tuple[int, int], step: int) -> list[GridCell]:
    width, height = size_cleared
    if step < 1 or width < 2 or height < 2:
        raise ValueError("Invalid grid dimensions or grid step.")
    records = []
    control = []
    for row, y in enumerate(range(0, height-1, step)):
        y2 = min(height-1, y+step)
        for col, x in enumerate(range(0, width-1, step)):
            x2 = min(width-1, x+step)
            records.append((row, col, x2-x, y2-y))
            control.extend(((x, y), (x2, y), (x2, y2), (x, y2)))
    warped = forward.transform(np.asarray(control, dtype=np.float64)).reshape((-1, 4, 2))
    result = []
    for (row,col,dx,dy), q in zip(records, warped):
        sx = (np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/(2*dx)
        sy = (np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/(2*dy)
        result.append(GridCell(row, col, q, float(sx), float(sy)))
    return result


def warp_image(image: np.ndarray, backward: TPS, out_width: int, out_height: int,
               interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """Sample CLEARED source using FIXED->CLEARED coordinates, memory-bounded by rows."""
    if out_width < 1 or out_height < 1:
        raise ValueError("Invalid output dimensions.")
    result = np.zeros((out_height, out_width, *image.shape[2:]), dtype=image.dtype)
    # Number of pixels per remap chunk ~200k (including coordinate temporaries).
    chunk_rows = max(1, min(256, 200_000 // out_width))
    x = np.arange(out_width, dtype=np.float64)
    for y0 in range(0, out_height, chunk_rows):
        y1 = min(y0 + chunk_rows, out_height)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float64), x, indexing="ij")
        coords = np.column_stack((xx.ravel(), yy.ravel()))
        transformed = backward.transform(coords)
        map_x = transformed[:,0].reshape((y1-y0, out_width)).astype(np.float32)
        map_y = transformed[:,1].reshape((y1-y0, out_width)).astype(np.float32)
        result[y0:y1] = cv2.remap(image, map_x, map_y, interpolation,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result


def preview_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        data = image
    else:
        values = image.astype(np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("Image contains no finite pixel values.")
        lo, hi = float(np.min(values[finite])), float(np.max(values[finite]))
        data = np.clip((values-lo) * (255.0/(hi-lo) if hi > lo else 0), 0, 255).astype(np.uint8)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data.copy()


def _write_csv_matrix(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for row in values:
            writer.writerow([f"{v:.9g}" for v in row])


def _cell_maps(cells: list[GridCell]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = max(c.row for c in cells)+1, max(c.col for c in cells)+1
    mx, my = np.full((ny,nx), np.nan), np.full((ny,nx), np.nan)
    for cell in cells:
        mx[cell.row, cell.col] = cell.stretch_x
        my[cell.row, cell.col] = cell.stretch_y
    return mx, my


def pixel_stretch_maps(cells: list[GridCell], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    mx = np.zeros((height, width), dtype=np.float32)
    my = np.zeros_like(mx)
    for cell in cells:
        corners = np.rint(cell.corners).astype(np.int32)
        x0 = max(0, int(corners[:,0].min()))
        y0 = max(0, int(corners[:,1].min()))
        x1 = min(width, int(corners[:,0].max())+1)
        y1 = min(height, int(corners[:,1].max())+1)
        if x0 >= x1 or y0 >= y1:
            continue
        poly = corners - np.array([x0,y0], dtype=np.int32)
        mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        mx[y0:y1, x0:x1][mask == 255] = cell.stretch_x
        my[y0:y1, x0:x1][mask == 255] = cell.stretch_y
    return mx, my


def export_visuals(output: Path, fixed: np.ndarray, restored: np.ndarray,
                   cells: list[GridCell]) -> None:
    fixed8, restored8 = preview_bgr(fixed), preview_bgr(restored)
    grid = restored8.copy()
    direction = np.zeros_like(grid)
    for c in cells:
        poly = np.rint(c.corners).astype(np.int32)
        cv2.polylines(grid, [poly], isClosed=True, color=(0,220,255), thickness=1, lineType=cv2.LINE_AA)
        if c.stretch_x > 1 and c.stretch_y > 1:
            color = (0, 175, 0)
        elif c.stretch_x < 1 and c.stretch_y < 1:
            color = (0, 0, 200)
        elif c.stretch_x > 1 and c.stretch_y < 1:
            color = (200, 0, 0)
        elif c.stretch_y > 1 and c.stretch_x < 1:
            color = (120, 120, 120)
        else:
            color = (0, 215, 215)
        cv2.fillConvexPoly(direction, poly, color)
    save_image(output / "deformed_grid_overlay.png", grid)
    save_image(output / "stretch_direction_classes.png", direction)
    save_image(output / "fixed_restored_overlay.png", cv2.addWeighted(fixed8, .5, restored8, .5, 0))
    save_image(output / "fixed_vs_restored.png", np.hstack((fixed8, restored8)))


def _errors(model: TPS, source: np.ndarray, target: np.ndarray) -> dict:
    d = np.linalg.norm(model.transform(source) - target, axis=1)
    return {"rms_px": float(np.sqrt(np.mean(d*d))), "max_px": float(np.max(d))}


@dataclass
class Registration:
    fixed: np.ndarray
    cleared: np.ndarray
    fixed_points: np.ndarray
    cleared_points: np.ndarray
    beta: float = 0.0

    def __post_init__(self) -> None:
        self.fixed_points, self.cleared_points = validate_pairs(
            self.fixed_points, self.cleared_points,
            (self.fixed.shape[1], self.fixed.shape[0]),
            (self.cleared.shape[1], self.cleared.shape[0]),
        )
        self.forward = TPS.fit(self.cleared_points, self.fixed_points, self.beta)
        self.backward = TPS.fit(self.fixed_points, self.cleared_points, self.beta)

    def warp(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.cleared.shape[:2]:
            raise ValueError("Input image must match the CLEARED working-image dimensions.")
        return warp_image(image, self.backward, self.fixed.shape[1], self.fixed.shape[0])

    def export(self, output: str | Path, grid_step: int | None = None,
               pixel_maps: bool = False, text_maps: bool = False,
               preprocess_info: dict | None = None) -> dict:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        step = grid_step or max(1, max(self.fixed.shape[1], self.cleared.shape[1]) // 20)
        cells = grid_cells(self.forward, (self.cleared.shape[1], self.cleared.shape[0]), step)
        restored = self.warp(self.cleared)
        save_image(output / "restored.tif", restored)
        write_csv(output / "landmarks_used.csv", self.fixed_points, self.cleared_points)
        sx, sy = _cell_maps(cells)
        _write_csv_matrix(output / "stretch_x_cells.csv", sx)
        _write_csv_matrix(output / "stretch_y_cells.csv", sy)
        with (output / "stretch_cells_detailed.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["row", "col", "stretch_x", "stretch_y"])
            for c in cells:
                writer.writerow([c.row, c.col, f"{c.stretch_x:.9g}", f"{c.stretch_y:.9g}"])
        export_visuals(output, self.fixed, restored, cells)
        if pixel_maps or text_maps:
            xmap, ymap = pixel_stretch_maps(cells, self.fixed.shape[1], self.fixed.shape[0])
            save_image(output / "stretch_x_pixels.tif", xmap)
            save_image(output / "stretch_y_pixels.tif", ymap)
            if text_maps:
                np.savetxt(output / "stretch_x_pixels.txt", xmap, fmt="%.9g")
                np.savetxt(output / "stretch_y_pixels.txt", ymap, fmt="%.9g")
        report = {
            "software": "Porcine Tissue TPS Registration", "version": __version__,
            "method": "2D TPS (r² log(r²), NumPy solve), OpenCV inverse-map resampling",
            "python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
            "platform": platform.platform(),
            "fixed_working_image_shape": list(self.fixed.shape),
            "cleared_working_image_shape": list(self.cleared.shape),
            "fixed_original_pixels_retained": bool((preprocess_info or {}).get("downsample_factor", 1) == 1 and not (preprocess_info or {}).get("match_fixed_width", False)),
            "landmark_pairs": len(self.fixed_points), "regularization_beta": self.beta,
            "grid_step_cleared_pixels": step, "preprocessing": preprocess_info or {},
            "forward_grid_map": "cleared -> fixed", "image_sampling_map": "fixed -> cleared",
            "interpolation": "cv2.INTER_LANCZOS4", "border_mode": "constant_black",
            "forward_landmark_error": _errors(self.forward, self.cleared_points, self.fixed_points),
            "inverse_landmark_error": _errors(self.backward, self.fixed_points, self.cleared_points),
            "stretch_definition": "transformed cell edge length / original cell edge length; not Cartesian Jacobian",
            "pixel_stretch_coverage": "uncovered pixels are zero; overlapping cells last-write-wins",
        }
        (output / "registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def warp_folder(self, input_folder: str | Path, output_folder: str | Path,
                    downsample_factor: int = 1) -> dict:
        source, destination = Path(input_folder).resolve(), Path(output_folder).resolve()
        if source == destination:
            raise ValueError("Input and output folder must be different.")
        if not source.is_dir():
            raise ValueError("Input folder does not exist.")
        destination.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError("No supported image files found in the selected folder.")
        results = {"input_folder": str(source), "output_folder": str(destination),
                   "registered": [], "skipped": []}
        for path in paths:
            try:
                image = load_image(path)
                if image.shape[:2] != self.cleared.shape[:2]:
                    image = downsample(image, downsample_factor)
                if image.shape[:2] != self.cleared.shape[:2]:
                    raise ValueError("image dimensions differ from CLEARED working reference after preprocessing")
                dest = destination / f"{path.stem}_restored.tif"
                if dest.exists():
                    raise ValueError(f"output already exists ({dest.name})")
                save_image(dest, self.warp(image))
                results["registered"].append(path.name)
            except (ValueError, IOError, cv2.error) as exc:
                results["skipped"].append({"filename": path.name, "reason": str(exc)})
        write_csv(destination / "landmarks_used.csv", self.fixed_points, self.cleared_points)
        (destination / "batch_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
