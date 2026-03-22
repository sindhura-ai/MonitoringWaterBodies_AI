# Monitoring Water from Space — Satellite Imagery + AI for Civil Engineers

**Author:** Dr. Sindhu | **Organization:** Smart Bhujal | **Contact:** sindhura@smartbhujal.com

A hands-on project that introduces civil engineers and water resources professionals to satellite image processing and AI/ML through a practical application: monitoring Bangalore's water bodies from space.

**Theme:** Progressively build from raw satellite data → spectral indices → thresholding → Random Forest → U-Net → Foundation Models.

**Audience:** Civil engineering, water resources, urban planning professionals. No prior AI/ML or image processing experience needed.

## Quick Start

```bash
git clone https://github.com/sindhura-ai/MonitoringWaterBodies_AI.git
cd MonitoringWaterBodies_AI

python3 -m venv myenv
source myenv/bin/activate
pip install -r setup/requirements.txt

jupyter lab notebooks/
```

Run notebooks in order: **01 → 02 → 03 → 04**. Notebook 01 downloads satellite data automatically.

## Notebooks

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | `01_get_and_visualize.ipynb` | Free satellite data via STAC, band composites, NIR/SWIR for water |
| 02 | `02_image_processing.ipynb` | MNDWI/NDVI/NDBI indices, thresholding, histograms, morphological cleanup |
| 03 | `03_ml_classification.ipynb` | Random Forest, feature importance, U-Net segmentation, transfer learning |
| 04 | `04_foundation_models.ipynb` | DINOv2, feature extraction, 5-way method comparison |

## The Progression

```
Thresholds → Random Forest → U-Net → Foundation Models
  (Notebook 02)  (Notebook 03)   (Notebook 03)  (Notebook 04)

Simple, fast                              Powerful, flexible
No training                               Needs labeled data
No spatial context                        Understands neighborhoods
```

## What You'll Learn

- Get **free 10m satellite imagery** every 5 days (Sentinel-2 via STAC API)
- **MNDWI** for water body detection, **NDVI** for vegetation, **NDBI** for impervious surfaces
- **Thresholding** and **morphological operations** — core image processing techniques
- **Random Forest** — fast, interpretable ML with feature importance
- **U-Net** — deep learning for pixel-level segmentation with spatial context
- **Foundation models** (DINOv2) — pre-trained visual features + spectral indices
- Export results as **GeoTIFF** for QGIS/ArcGIS workflows

## Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- GPU optional (all demos work on CPU)
- Internet for initial data download (Notebook 01)

## License

Smart Bhujal Pvt. Ltd
