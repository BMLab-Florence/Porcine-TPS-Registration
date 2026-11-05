# Porcine Tissue TPS Registration

**Python desktop application for landmark-based 2D registration of cardiac tissue images before and after tissue clearing.**

This application helps restore a **CLEARED** (post-treatment) tissue image to the geometry of a **FIXED** (pre-clearing) reference by fitting a two-dimensional thin-plate spline (TPS) to manually identified corresponding landmarks. It visualizes the restored image and deformation grid, exports cell-wise X/Y stretch ratios, and can apply the same mapping to other images acquired on the same spatial grid.

The software supports the shape-restoration step described in *The structural landscape of the porcine infarct border zone: cardiomyocyte disarray and collagen organization at micron-scale resolution* (Giardini et al.; publication details to be added after acceptance). It does **not** reproduce the paper's subsequent 3D segmentation, fibre orientation, disarray, or statistical analyses.

## Requirements

- Windows 10/11 (64-bit), macOS, or a Linux desktop.
- **Python 3.10–3.13, 64-bit** (Python 3.12 recommended).
- Internet connection during dependency installation.
- No C++ compiler, CMake, separate Qt installation or manually compiled OpenCV required.

The installation obtains **PySide6** (graphical interface), **NumPy** (TPS solver) and **opencv-python-headless** (image processing) from PyPI. OpenCV's *headless* build is intentional: the GUI is provided by PySide6 rather than `cv2.imshow`. Install into a fresh virtual environment to avoid conflicts with other OpenCV packages.

## Install and open the application

### Windows — step by step

1. Install **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/), if necessary. Open **PowerShell** and check:

   ```powershell
   py -3.12 --version
   ```

2. On GitHub, choose **Code → Download ZIP** and extract the repository into a normal folder (for example, under Documents).
3. **Easiest option:** double-click **`install_windows.bat`** in the extracted folder. It creates a virtual environment, installs the dependencies and runs the internal tests. When it reports success, double-click **`start_windows.bat`** to open the application. You do not need to enter terminal commands.

**Alternative — manual install** (useful if you need to diagnose an error or use a different compatible Python version): open the extracted folder in File Explorer, click the address bar, type `powershell` and press Enter. This opens PowerShell **inside the project folder**. Create an environment and install the application:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e .
   ```

Launch the graphical interface:

   ```powershell
   .\.venv\Scripts\python.exe launch.py
   ```

   For later sessions, double-click **`start_windows.bat`** inside the same folder. It uses the environment installed in step 4.

**If `py -3.12` is not found:** install Python 3.12 or use the installed compatible version (for example `py -3.11`) in the manual environment-creation command. The double-click installer specifically expects Python 3.12. **If an installation fails:** save the complete error text and your `python --version` output; do not install more OpenCV variants into the same environment.

### Linux / macOS

Open a terminal in the extracted repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python launch.py
```

On some Linux distributions you may need standard desktop OpenGL/Qt system libraries; the application requires a graphical desktop. All functionality is local and does not upload images.

### Run the internal checks (optional)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS: use `python -m unittest discover -s tests -v` inside the active virtual environment. These tests operate on small arrays created in memory; **no example images or patient/animal datasets are distributed**.

## Registration workflow

1. Click **Load FIXED + CLEARED images…**. Select the original **FIXED** tissue image (before clearing) followed by the **CLEARED** image (after clearing). Images appear side by side.
2. If needed, set a **downsample factor** *before* loading the images. Leave it at `1` to retain native pixel coordinates. The optional **Scale FIXED width to CLEARED** is for cases where the two images have different pixel sampling over equivalent fields of view; it is not a physical calibration. To change preprocessing later, select **Clear images / change preprocessing** and reload the images.
3. Click a landmark on **FIXED**, then click the matching landmark on **CLEARED**. Repeat for at least **four spatially distributed pairs**; more good correspondences generally provide more reliable local deformation. Use the mouse wheel to zoom and right/middle mouse dragging to pan.
4. Use **Undo**, **Remove selected** or **Clear all landmarks** to correct matches. You can **Save landmarks…** as CSV or **Load landmarks…** from a saved CSV or the historic `.points` format. CSV coordinates refer to the **working, preprocessed images**; save the accompanying `.meta.json` file with the CSV.
5. Set **TPS regularization β** (default `0`) and **grid step** (default: approximately one twentieth of the larger image width). Leave default regularization unless a different value is required by your protocol.
6. Click **Compute deformation and export…**, choose an output folder, and inspect the **Restored**, **Deformed grid**, **FIXED / restored overlay**, and **Stretch classes** tabs. Output files are saved in the chosen folder.
7. Optionally click **Apply mapping to image folder…** to process multiple images. These inputs **must have the same geometry, registration, origin and coordinate system** as the CLEARED reference (after the selected downsampling). Choose a different destination folder. Per-image failures are listed in the batch report.

The application performs **manual landmark selection**, not automatic keypoint detection. The tissue boundaries are not automatically segmented. Use landmarks on relevant anatomical contours or scar irregularities as appropriate to your experiment.

## Exported files

| File | Contents |
| --- | --- |
| `restored.tif` | CLEARED image resampled in FIXED coordinates (8-/16-bit or supported float image depth preserved). |
| `landmarks_used.csv` | Matched `fixed_x,fixed_y,cleared_x,cleared_y` coordinates actually used. |
| `deformed_grid_overlay.png` | TPS-deformed CLEARED grid drawn on the restored image. |
| `stretch_x_cells.csv`, `stretch_y_cells.csv` | Per-cell horizontal/vertical stretch ratios (without header). |
| `stretch_cells_detailed.csv` | Per-cell row, column and both stretch ratios. |
| `fixed_restored_overlay.png` | 50% reference/restored visual comparison. |
| `fixed_vs_restored.png` | Side-by-side FIXED and restored images. |
| `stretch_direction_classes.png` | Qualitative expansion/contraction classes, **not** a quantitative colour scale. |
| `registration_report.json` | Software versions, dimensions, settings, mapping conventions and landmark-fit errors. |
| `stretch_x_pixels.tif`, `stretch_y_pixels.tif` | Optional float32 per-pixel *rasterization of cell ratios*, not an analytical TPS Jacobian. |
| `stretch_x_pixels.txt`, `stretch_y_pixels.txt` | Optional text export of those maps (may be very large). |

For batch processing, each accepted image is saved as `<original_stem>_restored.tif` in the selected destination, together with `landmarks_used.csv` and `batch_report.json`.

### Meaning of stretch and transform direction

- **Forward geometric map:** CLEARED → FIXED. This is applied to grid vertices to visualize the deformation that restores shape.
- **Backward image-sampling map:** FIXED → CLEARED. The output image is filled by sampling the cleared source, thereby avoiding gaps from forward pixel scattering.
- **Cell X stretch:** average length of the top and bottom *transformed* edges divided by the original cell width. **Cell Y stretch:** equivalent average length of left and right edges divided by the original cell height. `1` means no length change, `>1` expansion, `<1` contraction.
- Because deformed cell edges can rotate or shear, these ratios **are not Cartesian components of the exact deformation Jacobian** and do not measure fibre orientation. When rasterized into pixel maps, pixels outside all deformed cells are `0` (undefined), and the last cell wins at overlaps.

Coordinates and stretch values are in **pixels and dimensionless ratios**. Any millimetre interpretation requires independent image calibration.

## Input constraints and scientific limitations

- The application supports **single-frame 2D** PNG, JPEG, BMP and TIFF images handled by OpenCV. It does **not** directly register a multi-page TIFF stack, a volumetric microscopy dataset or a 3D deformation field. Apply the same 2D mapping to individual registered planes only when scientifically justified.
- Shape restoration is only as reliable as the landmark correspondence, spatial coverage and pixel-size consistency. Results outside the landmark convex hull may be unstable; visually verify restored contours against the fixed reference.
- The **NumPy TPS solver** uses the standard 2D radial basis `U(r) = r² log(r²)` with affine terms and optional diagonal regularization. The historic research implementation used OpenCV C++ `ThinPlateSplineShapeTransformer`. Both are TPS approaches, **but bitwise/numerical identity with the historic C++ pipeline is not claimed** until paired images, point sets and outputs have been compared. The original program also included preprocessing conventions that must be matched before any one-to-one comparison.
- The paper's source image datasets, original matching-point files and downstream image analysis are **not contained in this repository**. This package enables running the registration procedure on your own data; it does not yet establish exact replication of published biological measurements.
- Use of the same mapping for batch images assumes fixed spatial correspondence; images of a different shape or sampling are rejected or skipped. Different images are never automatically registered to each other.

## Project structure

```text
Porcine-TPS-Registration-Python/
├── README.md
├── pyproject.toml
├── requirements.txt
├── launch.py
├── install_windows.bat
├── start_windows.bat
├── porcine_tps/
│   ├── __init__.py
│   ├── __main__.py
│   ├── gui.py
│   ├── landmarks.py
│   └── registration.py
├── tests/
│   └── test_registration.py
└── .github/workflows/
    └── tests.yml
```

No experimental data or sample images are included.

## Citation, permissions and support

When using this application in academic work, please cite the associated article once its bibliographic information is available (reference above). For reproducibility, also report the software version, preprocessing parameters, landmark coordinates, TPS β and grid step; these are included in the exported report and landmark file.

**Release checklist before making the repository public:** confirm code ownership and collaborators' permission; choose and add a software license; run the automated checks and GUI on the target operating system; compare outputs against the historic C++ implementation using approved experimental images and their landmark sets; add final article citation/DOI when known.

**Licensing status:** no open-source license is granted by the files currently provided. A repository being visible on GitHub does not by itself grant rights to reuse or modify its contents. Add a suitable LICENSE only after agreement among the rights holders.
