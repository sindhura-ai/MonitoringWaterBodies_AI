"""Shared utility functions for satellite imagery notebooks.

Focused on water resources and land use monitoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio


# Land cover class definitions (water-resources focused)
LAND_COVER_CLASSES = {
    0: ("Water", "#1565C0"),
    1: ("Vegetation", "#2E7D32"),
    2: ("Bare Soil", "#8D6E63"),
    3: ("Built-up", "#78909C"),
}

LAND_COVER_CMAP = ListedColormap([c for _, c in LAND_COVER_CLASSES.values()])
LAND_COVER_LABELS = [name for name, _ in LAND_COVER_CLASSES.values()]


def load_band(path):
    """Load a single band GeoTIFF as a numpy array."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile


def normalize_band(band, percentile_clip=(2, 98)):
    """Normalize a band to 0-1 range using percentile clipping."""
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band)
    vmin, vmax = np.percentile(valid, percentile_clip)
    clipped = np.clip(band, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin + 1e-10)


def make_rgb(red, green, blue, percentile_clip=(2, 98)):
    """Create an RGB composite from three bands with histogram stretching."""
    rgb = np.stack([
        normalize_band(red, percentile_clip),
        normalize_band(green, percentile_clip),
        normalize_band(blue, percentile_clip),
    ], axis=-1)
    return np.clip(rgb, 0, 1)


def compute_ndvi(nir, red):
    """Compute NDVI = (NIR - Red) / (NIR + Red). Measures vegetation health."""
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0)
    return ndvi.astype(np.float32)


def compute_ndwi(green, nir):
    """Compute NDWI = (Green - NIR) / (Green + NIR). Highlights open water."""
    denom = green + nir
    ndwi = np.where(denom > 0, (green - nir) / denom, 0)
    return ndwi.astype(np.float32)


def compute_mndwi(green, swir):
    """Compute MNDWI = (Green - SWIR) / (Green + SWIR). Better water detection in built-up areas."""
    denom = green + swir
    mndwi = np.where(denom > 0, (green - swir) / denom, 0)
    return mndwi.astype(np.float32)


def compute_ndbi(swir, nir):
    """Compute NDBI = (SWIR - NIR) / (SWIR + NIR). Highlights built-up/impervious surfaces."""
    denom = swir + nir
    ndbi = np.where(denom > 0, (swir - nir) / denom, 0)
    return ndbi.astype(np.float32)


def extract_water_mask(ndwi, threshold=0.0):
    """Extract a binary water mask from NDWI. NDWI > threshold = water."""
    return (ndwi > threshold).astype(np.uint8)


def compute_water_area_km2(water_mask, pixel_size_m=10):
    """Compute total water area in sq km from a binary mask."""
    pixel_area_km2 = (pixel_size_m ** 2) / 1e6
    return water_mask.sum() * pixel_area_km2


def plot_image(data, title="", cmap=None, figsize=(10, 8), colorbar=False, vmin=None, vmax=None):
    """Display an image with optional colorbar."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def plot_comparison(images, titles, cmaps=None, figsize=(18, 5)):
    """Display multiple images side by side for comparison."""
    n = len(images)
    if cmaps is None:
        cmaps = [None] * n
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    return fig, axes


def plot_land_cover(classification, title="Land Cover Classification", figsize=(10, 8)):
    """Display a land cover classification map with legend."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(classification, cmap=LAND_COVER_CMAP, vmin=0, vmax=len(LAND_COVER_CLASSES) - 1)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=color)
        for name, color in LAND_COVER_CLASSES.values()
    ]
    ax.legend(patches, LAND_COVER_LABELS, loc="lower right", fontsize=10)
    plt.tight_layout()
    return fig, ax


def plot_water_overlay(rgb, water_mask, title="Water Bodies Detected", figsize=(10, 8)):
    """Overlay detected water bodies on RGB image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    water_overlay = np.ma.masked_where(water_mask == 0, water_mask)
    ax.imshow(water_overlay, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


def confusion_matrix_report(y_true, y_pred, class_names=None):
    """Print a simple confusion matrix and per-class metrics."""
    from sklearn.metrics import classification_report, confusion_matrix

    if class_names is None:
        class_names = LAND_COVER_LABELS

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    return cm
