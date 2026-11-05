"""PySide6 desktop workflow: two-image matching, TPS restoration and exports."""
from __future__ import annotations

import json
from pathlib import Path
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressDialog,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .landmarks import read_landmarks, validate_pairs, write_csv
from .registration import Registration, load_image, prepare_images, preview_bgr


def _pixmap(array: np.ndarray) -> QPixmap:
    bgr = preview_bgr(array)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # .copy() detaches Qt image storage from the temporary NumPy buffer.
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy())


class ImageView(QGraphicsView):
    point_clicked = Signal(float, float)

    def __init__(self, selectable: bool = True):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._size = (0, 0)
        self._points: list[tuple[float, float]] = []
        self._pending: tuple[float, float] | None = None
        self._selectable = selectable
        self._pan = False
        self._pan_start = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.setMinimumSize(320, 245)

    def set_image(self, image: np.ndarray | None) -> None:
        self._scene.clear()
        if image is None:
            self._size = (0, 0)
            self._points = []
            return
        pixmap = _pixmap(image)
        self._size = (pixmap.width(), pixmap.height())
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, *self._size)
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mark([], None)

    def mark(self, points: list[tuple[float, float]], pending: tuple[float, float] | None = None) -> None:
        self._points = list(points)
        self._pending = pending
        # Image is the first item in the scene; clear only the overlay graphics.
        for item in list(self._scene.items()):
            if item.data(0) == "landmark-overlay":
                self._scene.removeItem(item)
        if not self._size[0]:
            return
        scale = max(3.0, min(self._size) / 140.0)
        for i, xy in enumerate(points):
            self._draw_point(xy, str(i+1), QColor("#ffcc00"), scale)
        if pending is not None:
            self._draw_point(pending, "?", QColor("#ff5e5e"), scale)

    def _draw_point(self, xy, label, color, radius):
        x,y = xy
        pen = QPen(color, max(1., radius/3.))
        marker = self._scene.addEllipse(x-radius, y-radius, 2*radius, 2*radius, pen, QBrush(Qt.BrushStyle.NoBrush))
        marker.setData(0, "landmark-overlay")
        marker.setZValue(10)
        text = self._scene.addText(label)
        text.setDefaultTextColor(color)
        font = text.font(); font.setPixelSize(max(9,int(radius*3))); font.setBold(True)
        text.setFont(font)
        text.setPos(x+radius, y-radius*2)
        text.setZValue(11)
        text.setData(0, "landmark-overlay")

    def wheelEvent(self, event):
        if not self._size[0]:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else (1/1.2)
        new_scale = self.transform().m11() * factor
        if 0.025 <= new_scale <= 30:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept(); return
        if self._selectable and self._size[0] and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if 0 <= point.x() < self._size[0] and 0 <= point.y() < self._size[1]:
                self.point_clicked.emit(point.x(), point.y())
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan and self._pan_start is not None:
            current = event.position().toPoint()
            delta = current - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = current
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept(); return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep user-selected zoom and pan on resize; use Fit only on load.

    def fit_image(self):
        if self._size[0]:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class TaskWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: str, reg: Registration,
                 output: str, *, grid_step: int = 0, pixel_maps: bool = False,
                 text_maps: bool = False, preprocess: dict | None = None,
                 input_dir: str = "", factor: int = 1):
        super().__init__()
        self.operation = operation
        self.reg = reg
        self.output = output
        self.grid_step = grid_step
        self.pixel_maps = pixel_maps
        self.text_maps = text_maps
        self.preprocess = preprocess
        self.input_dir = input_dir
        self.factor = factor

    @Slot()
    def run(self):
        try:
            if self.operation == "export":
                result = self.reg.export(self.output, self.grid_step or None,
                                         self.pixel_maps, self.text_maps, self.preprocess)
            else:
                result = self.reg.warp_folder(self.input_dir, self.output, self.factor)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=4)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Porcine Tissue TPS Registration")
        self.resize(1450, 900)
        self.fixed_raw: np.ndarray | None = None
        self.cleared_raw: np.ndarray | None = None
        self.fixed: np.ndarray | None = None
        self.cleared: np.ndarray | None = None
        self.fixed_pairs: list[tuple[float,float]] = []
        self.cleared_pairs: list[tuple[float,float]] = []
        self.pending_fixed: tuple[float,float] | None = None
        self._reg: Registration | None = None
        self._worker = None
        self._thread = None
        self._progress = None
        self._last_output = None
        self._image_names = None
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        self.panel = QWidget()
        self.panel.setMaximumWidth(390)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setSpacing(10)
        heading = QLabel("Landmark-based TPS restoration")
        heading.setStyleSheet("font-size: 17px; font-weight: bold")
        panel_layout.addWidget(heading)
        description = QLabel("Match anatomical points before and after tissue clearing, then restore the cleared image onto the original geometry.")
        description.setWordWrap(True)
        panel_layout.addWidget(description)

        images_box = QGroupBox("1 · Load reference images")
        iv = QVBoxLayout(images_box)
        self.load_btn = QPushButton("Load FIXED + CLEARED images…")
        self.load_btn.clicked.connect(self.load_references)
        iv.addWidget(self.load_btn)
        self.paths = QLabel("No images selected")
        self.paths.setWordWrap(True)
        self.paths.setStyleSheet("color: #64748b")
        iv.addWidget(self.paths)
        form = QFormLayout()
        self.factor = QSpinBox(); self.factor.setRange(1, 100); self.factor.setValue(1)
        self.factor.setToolTip("Working coordinate downsampling. Select before loading images.")
        form.addRow("Downsample factor", self.factor)
        self.match_width = QCheckBox("Scale FIXED width to CLEARED")
        self.match_width.setToolTip("Use only when pixel scale/FOV warrants it; does not calibrate physical dimensions.")
        iv.addLayout(form); iv.addWidget(self.match_width)
        self.reset_btn = QPushButton("Clear images / change preprocessing")
        self.reset_btn.clicked.connect(self.reset_images)
        iv.addWidget(self.reset_btn)
        panel_layout.addWidget(images_box)

        points_box = QGroupBox("2 · Select matching landmarks")
        pv = QVBoxLayout(points_box)
        how = QLabel("Click FIXED first, then the corresponding point on CLEARED. Repeat for ≥4 well-spread pairs. Mouse wheel: zoom; right/middle drag: pan.")
        how.setWordWrap(True); pv.addWidget(how)
        self.count_label = QLabel("0 complete pairs")
        pv.addWidget(self.count_label)
        self.table = QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(["Fixed X", "Fixed Y", "Cleared X", "Cleared Y"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        pv.addWidget(self.table)
        h1 = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.remove_btn = QPushButton("Remove selected")
        h1.addWidget(self.undo_btn);h1.addWidget(self.remove_btn)
        self.undo_btn.clicked.connect(self.undo_match)
        self.remove_btn.clicked.connect(self.remove_selected)
        pv.addLayout(h1)
        h2 = QHBoxLayout()
        self.save_btn = QPushButton("Save landmarks…")
        self.open_btn = QPushButton("Load landmarks…")
        h2.addWidget(self.save_btn);h2.addWidget(self.open_btn)
        self.save_btn.clicked.connect(self.save_landmarks)
        self.open_btn.clicked.connect(self.open_landmarks)
        pv.addLayout(h2)
        self.clear_btn = QPushButton("Clear all landmarks")
        self.clear_btn.clicked.connect(self.clear_pairs)
        pv.addWidget(self.clear_btn)
        panel_layout.addWidget(points_box)

        compute_box = QGroupBox("3 · Compute and export")
        cv = QVBoxLayout(compute_box)
        settings = QFormLayout()
        self.beta = QDoubleSpinBox()
        self.beta.setRange(0, 1e8); self.beta.setDecimals(5); self.beta.setValue(0)
        self.beta.setToolTip("Regularization; zero interpolates the landmarks.")
        self.grid = QSpinBox(); self.grid.setRange(0, 10000); self.grid.setValue(0)
        self.grid.setSpecialValueText("Auto (max width / 20)")
        settings.addRow("TPS regularization β", self.beta)
        settings.addRow("Grid step (pixels)", self.grid)
        cv.addLayout(settings)
        self.pixel = QCheckBox("Export pixel-wise float32 TIFF stretch maps")
        self.textmap = QCheckBox("Also export large pixel-wise TXT matrices")
        self.textmap.toggled.connect(lambda yes: self.pixel.setChecked(True) if yes else None)
        cv.addWidget(self.pixel); cv.addWidget(self.textmap)
        self.compute_btn = QPushButton("Compute deformation and export…")
        self.compute_btn.clicked.connect(self.compute)
        cv.addWidget(self.compute_btn)
        self.batch_btn = QPushButton("Apply mapping to image folder…")
        self.batch_btn.clicked.connect(self.batch)
        cv.addWidget(self.batch_btn)
        panel_layout.addWidget(compute_box)
        panel_layout.addStretch(1)
        scroller = QScrollArea(); scroller.setWidgetResizable(True); scroller.setWidget(self.panel)
        scroller.setMinimumWidth(335); scroller.setMaximumWidth(420)
        splitter.addWidget(scroller)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()
        source_page = QWidget(); source_layout = QHBoxLayout(source_page)
        a = QVBoxLayout(); b = QVBoxLayout()
        for name, target in (("FIXED — before clearing", a),("CLEARED — after clearing",b)):
            label=QLabel(name); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight:bold; padding: 5px")
            target.addWidget(label)
        self.fixed_view = ImageView(True); self.cleared_view = ImageView(True)
        a.addWidget(self.fixed_view); b.addWidget(self.cleared_view)
        self.fixed_view.point_clicked.connect(self.pick_fixed)
        self.cleared_view.point_clicked.connect(self.pick_cleared)
        fit_a = QPushButton("Fit FIXED view"); fit_a.clicked.connect(self.fixed_view.fit_image)
        fit_b = QPushButton("Fit CLEARED view"); fit_b.clicked.connect(self.cleared_view.fit_image)
        a.addWidget(fit_a); b.addWidget(fit_b)
        source_layout.addLayout(a,1); source_layout.addLayout(b,1)
        self.tabs.addTab(source_page, "Landmark matching")
        self.preview_views: dict[str, ImageView] = {}
        for title in ("Restored", "Deformed grid", "FIXED / restored overlay", "Stretch classes"):
            view = ImageView(False)
            self.preview_views[title] = view
            container = QWidget(); wrap = QVBoxLayout(container)
            wrap.addWidget(view)
            fit = QPushButton("Fit preview")
            fit.clicked.connect(view.fit_image)
            wrap.addWidget(fit)
            self.tabs.addTab(container, title)
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1,1)
        splitter.setSizes([375, 1100])
        self.statusBar().showMessage("Load a FIXED and a CLEARED image to begin.")

    def _refresh(self):
        self.table.setRowCount(len(self.fixed_pairs))
        for row,(p,q) in enumerate(zip(self.fixed_pairs,self.cleared_pairs)):
            for col, v in enumerate((*p,*q)):
                self.table.setItem(row,col,QTableWidgetItem(f"{v:.1f}"))
        self.fixed_view.mark(self.fixed_pairs, self.pending_fixed)
        self.cleared_view.mark(self.cleared_pairs)
        pending = " — select matching CLEARED point" if self.pending_fixed is not None else " — next: click FIXED"
        self.count_label.setText(f"{len(self.fixed_pairs)} complete pairs{pending}")
        loaded = self.fixed is not None and self.cleared is not None
        busy = self._thread is not None
        for btn in (self.undo_btn, self.remove_btn, self.clear_btn):
            btn.setEnabled(loaded and not busy and bool(self.fixed_pairs or self.pending_fixed))
        self.save_btn.setEnabled(loaded and not busy and len(self.fixed_pairs) >= 4 and self.pending_fixed is None)
        self.open_btn.setEnabled(loaded and not busy)
        self.compute_btn.setEnabled(loaded and not busy and len(self.fixed_pairs) >= 4 and self.pending_fixed is None)
        self.batch_btn.setEnabled(loaded and not busy and self._reg is not None and len(self.fixed_pairs) >= 4 and self.pending_fixed is None)
        self.load_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(loaded and not busy)
        self.factor.setEnabled(not loaded and not busy)
        self.match_width.setEnabled(not loaded and not busy)

    def _error(self, title, message):
        QMessageBox.critical(self, title, str(message))

    def _invalidate(self):
        self._reg = None
        self._last_output = None
        for preview in self.preview_views.values():
            preview.set_image(None)

    def reset_images(self):
        if self._thread is not None:
            return
        if self.fixed is not None and self.fixed_pairs:
            if QMessageBox.question(self, "Clear session", "Clear both images and all landmark pairs?") != QMessageBox.StandardButton.Yes:
                return
        self.fixed_raw = self.cleared_raw = self.fixed = self.cleared = None
        self.fixed_pairs.clear(); self.cleared_pairs.clear(); self.pending_fixed=None
        self.fixed_view.set_image(None); self.cleared_view.set_image(None)
        self.paths.setText("No images selected")
        self._image_names = None
        self._invalidate(); self._refresh()

    def load_references(self):
        if self._thread is not None:
            return
        if self.fixed is not None and self.fixed_pairs:
            if QMessageBox.question(self, "Replace images", "Loading images will discard the current landmark pairs. Continue?") != QMessageBox.StandardButton.Yes:
                return
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        fixed_path,_ = QFileDialog.getOpenFileName(self,"Select FIXED image — before clearing","",filters)
        if not fixed_path: return
        cleared_path,_ = QFileDialog.getOpenFileName(self,"Select CLEARED image — after clearing","",filters)
        if not cleared_path: return
        try:
            f, c = load_image(fixed_path),load_image(cleared_path)
            ff,cc = prepare_images(f,c,self.factor.value(),self.match_width.isChecked())
            self.fixed_raw,self.cleared_raw=f,c
            self.fixed,self.cleared=ff,cc
            self._image_names={"fixed":Path(fixed_path).name,"cleared":Path(cleared_path).name}
            self.fixed_pairs.clear(); self.cleared_pairs.clear(); self.pending_fixed=None
            self.fixed_view.set_image(ff);self.cleared_view.set_image(cc)
            self.paths.setText(f"FIXED: {Path(fixed_path).name} ({ff.shape[1]}×{ff.shape[0]})\nCLEARED: {Path(cleared_path).name} ({cc.shape[1]}×{cc.shape[0]})")
            self._invalidate(); self._refresh()
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage("Click a point on FIXED, then its corresponding point on CLEARED.")
        except Exception as exc:
            self._error("Cannot load images", exc)

    def pick_fixed(self,x,y):
        if self.fixed is None or self._thread is not None:return
        self.pending_fixed=(float(x),float(y)); self._invalidate();self._refresh()
        self.statusBar().showMessage("Now click the corresponding landmark on CLEARED.")

    def pick_cleared(self,x,y):
        if self.cleared is None or self._thread is not None:return
        if self.pending_fixed is None:
            self.statusBar().showMessage("Click FIXED first, then CLEARED.")
            return
        self.fixed_pairs.append(self.pending_fixed)
        self.cleared_pairs.append((float(x),float(y)))
        self.pending_fixed=None
        self._invalidate();self._refresh()
        self.statusBar().showMessage("Landmark pair added. Click FIXED for the next pair.")

    def undo_match(self):
        if self.pending_fixed is not None:
            self.pending_fixed = None
        elif self.fixed_pairs:
            self.fixed_pairs.pop();self.cleared_pairs.pop()
        self._invalidate();self._refresh()

    def remove_selected(self):
        row = self.table.currentRow()
        if row < 0:
            self.statusBar().showMessage("Select a landmark row to remove.")
            return
        self.fixed_pairs.pop(row);self.cleared_pairs.pop(row)
        self.pending_fixed=None
        self._invalidate();self._refresh()

    def clear_pairs(self):
        self.fixed_pairs.clear();self.cleared_pairs.clear();self.pending_fixed=None
        self._invalidate();self._refresh()

    def save_landmarks(self):
        path,_=QFileDialog.getSaveFileName(self,"Save matching landmarks","landmarks.csv","CSV files (*.csv)")
        if not path:return
        try:
            write_csv(path, np.array(self.fixed_pairs),np.array(self.cleared_pairs))
            metadata = {
                "coordinate_system":"working_image_pixels",
                "fixed_shape":list(self.fixed.shape),"cleared_shape":list(self.cleared.shape),
                "downsample_factor":self.factor.value(),"match_fixed_width":self.match_width.isChecked(),
                "note":"Keep this sidecar with the CSV; point coordinates depend on preprocessing.",
            }
            Path(path).with_suffix(".meta.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
            self.statusBar().showMessage(f"Saved {len(self.fixed_pairs)} landmark pairs.")
        except Exception as exc:self._error("Cannot save landmarks",exc)

    def open_landmarks(self):
        path,_=QFileDialog.getOpenFileName(self,"Open landmarks (CSV or original .points)","","Landmarks (*.csv *.points)")
        if not path:return
        try:
            f,c=read_landmarks(path)
            validate_pairs(f,c,(self.fixed.shape[1],self.fixed.shape[0]),(self.cleared.shape[1],self.cleared.shape[0]))
            meta=Path(path).with_suffix(".meta.json")
            if meta.is_file():
                info=json.loads(meta.read_text(encoding="utf-8"))
                if (list(self.fixed.shape)!=info.get("fixed_shape") or list(self.cleared.shape)!=info.get("cleared_shape")
                    or self.factor.value()!=info.get("downsample_factor")
                    or self.match_width.isChecked()!=info.get("match_fixed_width")):
                    raise ValueError("Saved landmark image dimensions do not match this session.")
            self.fixed_pairs=[tuple(p) for p in f.tolist()]
            self.cleared_pairs=[tuple(p) for p in c.tolist()]
            self.pending_fixed=None
            self._invalidate();self._refresh()
            self.statusBar().showMessage(f"Loaded {len(f)} matching pairs.")
        except Exception as exc:self._error("Cannot load landmarks",exc)

    def _model(self) -> Registration:
        if self.pending_fixed is not None:
            raise ValueError("Complete the current matching pair first.")
        if (self._reg is None or self._reg.beta != self.beta.value()):
            self._reg = Registration(self.fixed, self.cleared,
                np.asarray(self.fixed_pairs,dtype=np.float64),np.asarray(self.cleared_pairs,dtype=np.float64),self.beta.value())
        return self._reg

    def compute(self):
        try: reg=self._model()
        except Exception as exc:
            self._error("Invalid landmark configuration",exc);return
        folder=QFileDialog.getExistingDirectory(self,"Select output folder")
        if not folder:return
        existing=Path(folder)/"restored.tif"
        if existing.exists():
            if QMessageBox.question(self,"Overwrite results",f"{existing.name} already exists. Overwrite registration outputs in this folder?") != QMessageBox.StandardButton.Yes:
                return
        self._last_output = folder
        info={"downsample_factor":self.factor.value(),"match_fixed_width":self.match_width.isChecked(),
              "source_image_filenames":self._image_names,
              "fixed_source_shape":list(self.fixed_raw.shape),"cleared_source_shape":list(self.cleared_raw.shape)}
        worker=TaskWorker("export",reg,folder,grid_step=self.grid.value(),pixel_maps=self.pixel.isChecked(),text_maps=self.textmap.isChecked(),preprocess=info)
        self._launch(worker, "Computing TPS deformation and exporting files…")

    def batch(self):
        try:reg=self._model()
        except Exception as exc:
            self._error("Invalid landmark configuration",exc);return
        source=QFileDialog.getExistingDirectory(self,"Select folder of CLEARED images")
        if not source:return
        out=QFileDialog.getExistingDirectory(self,"Select different destination folder")
        if not out:return
        worker=TaskWorker("batch",reg,out,input_dir=source,factor=self.factor.value())
        self._launch(worker,"Warping image folder…")

    def _launch(self,worker:TaskWorker,text:str):
        self._thread=QThread(self)
        self._worker=worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.succeeded.connect(self._task_succeeded)
        worker.failed.connect(self._task_failed)
        worker.succeeded.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_done)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self._progress=QProgressDialog(text,"",0,0,self)
        self._progress.setWindowTitle("Processing")
        self._progress.setCancelButton(None)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.show()
        self._refresh()
        self._thread.start()

    @Slot(object)
    def _task_succeeded(self,result):
        if self._worker.operation == "export":
            base=Path(self._worker.output)
            filenames={
                "Restored":"restored.tif",
                "Deformed grid":"deformed_grid_overlay.png",
                "FIXED / restored overlay":"fixed_restored_overlay.png",
                "Stretch classes":"stretch_direction_classes.png",
            }
            for name,file in filenames.items():
                try:self.preview_views[name].set_image(load_image(base/file))
                except Exception as exc:self.statusBar().showMessage(f"Preview unavailable: {exc}")
            self.tabs.setCurrentIndex(1)
            QMessageBox.information(self,"Registration complete",f"Registration outputs saved in:\n{base}\n\nForward RMS landmark error: {result['forward_landmark_error']['rms_px']:.4g} px")
        else:
            QMessageBox.information(self,"Batch complete",f"Warped {len(result['registered'])} images.\nSkipped {len(result['skipped'])} images.\n\nReports saved to:\n{self._worker.output}")
        self.statusBar().showMessage("Processing complete. Inspect exported outputs and registration report.")

    @Slot(str)
    def _task_failed(self,error):
        self._error("Processing failed",error)
        self.statusBar().showMessage("Processing failed; check the error details.")

    @Slot()
    def _thread_done(self):
        if self._progress is not None:self._progress.close();self._progress=None
        if self._thread is not None:
            self._thread.deleteLater();self._thread=None
        self._worker=None
        self._refresh()

    def closeEvent(self,event):
        if self._thread is not None:
            QMessageBox.information(self,"Processing active","Finish the current operation before closing the application.")
            event.ignore();return
        event.accept()


def main():
    import sys
    app=QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Porcine Tissue TPS Registration")
    window=MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
