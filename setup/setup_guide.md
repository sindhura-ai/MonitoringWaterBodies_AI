# Setup Guide

## Option 1: pip + venv (Recommended)

```bash
cd IMGPROC

# Create a virtual environment
python3 -m venv myenv
source myenv/bin/activate  # macOS/Linux
# myenv\Scripts\activate   # Windows

# Install dependencies
pip install -r setup/requirements.txt

# Verify
python -c "import rasterio, torch, sklearn, pystac_client; print('All imports OK')"
```

Note: You may need GDAL system libraries first. On macOS: `brew install gdal`. On Ubuntu: `sudo apt install gdal-bin libgdal-dev`.

## Option 2: Conda

```bash
conda env create -f setup/environment.yml
conda activate satellite-ai
```

## Option 3: Google Colab (Zero Setup)

Each notebook can run in Colab — uncomment the `!pip install` cell at the top.

## Running the Notebooks

```bash
source myenv/bin/activate
jupyter lab notebooks/
```

Run in order: **01 → 02 → 03 → 04**. Each notebook saves outputs that the next one uses.

Notebook 01 downloads satellite data automatically — no separate data download step needed.

## GPU Setup (Optional)

All demos run on CPU. U-Net training takes ~2 min on CPU, ~20s on GPU.

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `rasterio` import error | Install GDAL first (`brew install gdal` on macOS) |
| STAC API timeout | Check internet; Notebook 01 must run online first |
| Memory error | Close other apps; the image is ~70MB in memory |
| `torch` not finding GPU | Reinstall PyTorch for your platform from pytorch.org |
| DINOv2 download slow | First run downloads ~85MB model; cached after that |
