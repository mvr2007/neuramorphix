"""
NeuraMorphix - Monitoring Vegetation Changes using Satellite Imagery
Problem Statement: CS12 (Geospatial Environment)
Team: Tech Squad (Krishna Saketh, Ayush Pramanik, Adeera Rajput, MVR Tarun)
Institution: SRM Institute of Science and Technology
HackForge 2026
"""

import io
import base64
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
def init_page_styling():
    """
    Initializes Streamlit page configuration and NeuraMorphix presentation theme CSS.
    """
    st.set_page_config(
        page_title="NeuraMorphix | Satellite Vegetation Monitor",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for NeuraMorphix deck branding (Purple & Cyan accents, modern cards)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #180b38 0%, #2b1055 50%, #0d1b2a 100%);
            padding: 24px 30px;
            border-radius: 16px;
            color: white;
            margin-bottom: 24px;
            border-left: 6px solid #00d2ff;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        }
        
        .brand-pill {
            display: inline-block;
            background: linear-gradient(90deg, #6f42c1, #00d2ff);
            color: white;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 4px 12px;
            border-radius: 20px;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        
        .header-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #ffffff 0%, #00d2ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-sub {
            font-size: 0.95rem;
            color: #ccd6f6;
            margin-top: 6px;
            margin-bottom: 0;
        }

        .kpi-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        }
        
        .kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            margin-bottom: 4px;
        }
        
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        
        .kpi-sub {
            font-size: 0.82rem;
            margin-top: 6px;
            font-weight: 500;
        }
        
        .alert-banner-critical {
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border: 1px solid #feb2b2;
            border-left: 6px solid #e53e3e;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            color: #9b2c2c;
        }

        .alert-banner-normal {
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
            border: 1px solid #9ae6b4;
            border-left: 6px solid #38a169;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            color: #22543d;
        }

        .meta-tag {
            display: inline-block;
            font-size: 0.78rem;
            padding: 3px 8px;
            border-radius: 6px;
            background: #f1f5f9;
            color: #475569;
            margin-right: 6px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .legend-box {
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-size: 0.85rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-top: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 4px;
        }
        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. PRE-CONFIGURED AOI DATABASE & GEOSPATIAL DEFINITIONS
# -----------------------------------------------------------------------------
PRESET_AOIS = {
    "Amazon Basin (Rondônia Deforestation Arc, Brazil)": {
        "bounds": (-10.15, -63.15, -9.85, -62.85),  # (min_lat, min_lon, max_lat, max_lon)
        "center": [-10.0, -63.0],
        "zoom": 11,
        "description": "Active agricultural frontier & fishbone logging corridors in Western Amazonia.",
        "default_scenario": "deforestation"
    },
    "Black Forest (Schwarzwald, Germany - Drought & Bark Beetle)": {
        "bounds": (48.35, 8.15, 48.65, 8.45),
        "center": [48.5, 8.3],
        "zoom": 11,
        "description": "Temperate coniferous canopy experiencing drought stress and bark beetle clearing.",
        "default_scenario": "canopy_dieback"
    },
    "Chennai / SRMIST Peri-Urban Corridor (Urban Sprawl)": {
        "bounds": (12.78, 80.00, 12.88, 80.10),
        "center": [12.83, 80.05],
        "zoom": 12,
        "description": "Rapid suburban infrastructure expansion displacing peri-urban scrubland & wetlands.",
        "default_scenario": "urban_expansion"
    },
    "Sundarbans Biosphere Reserve (Mangrove Dynamics)": {
        "bounds": (21.80, 88.80, 22.10, 89.10),
        "center": [21.95, 88.95],
        "zoom": 11,
        "description": "Tidal mangrove estuary with cyclonic disturbance and dynamic mudflat regeneration.",
        "default_scenario": "regrowth"
    }
}


# -----------------------------------------------------------------------------
# 3. CORE GEOSPATIAL & SYNTHETIC DATA GENERATOR ENGINE
# -----------------------------------------------------------------------------
def generate_synthetic_sentinel_bands(
    dims=(280, 280),
    scenario="deforestation",
    seed=42
):
    """
    Generates physically consistent Sentinel-2 Band 4 (Red: ~665nm) and
    Band 8 (NIR: ~842nm) surface reflectance arrays for bi-temporal monitoring.
    Simulates realistic landscapes (water, dense forests, soil, urban patches)
    and specific change signatures between Date 1 and Date 2.
    """
    rng = np.random.RandomState(seed)
    rows, cols = dims
    
    # 1. Base terrain simulation via multi-frequency continuous fields
    y, x = np.mgrid[0:rows, 0:cols]
    
    # Low frequency landscape elevation & soil moisture gradients
    field_low = np.sin(x / max(10.0, cols / 6.0)) * np.cos(y / max(10.0, rows / 6.0))
    sigma_val = max(2.0, min(rows, cols) / 24.0)
    field_mid = gaussian_filter(rng.randn(rows, cols), sigma=sigma_val)
    base_landscape = (field_low * 0.4 + field_mid * 0.6)
    base_landscape = (base_landscape - base_landscape.min()) / (base_landscape.max() - base_landscape.min() + 1e-6)

    # 2. Meandering river/water body mask (low NIR, low Red, negative NDVI)
    river_path = (rows * 0.5) + np.sin(x / max(1.0, cols / 10.0)) * (rows * 0.12)
    river_dist = np.abs(y - river_path)
    river_width = max(2.0, rows * 0.02)
    water_mask = river_dist < river_width
    water_mask = gaussian_filter(water_mask.astype(float), sigma=1.0) > 0.35

    # 3. Date 1 Initial Surface Reflectance (Healthy baseline)
    veg_density = np.clip(base_landscape * 1.2 - 0.1, 0.0, 1.0)
    
    # Red Band Date 1 (absorbed by chlorophyll in vegetation)
    red_d1 = 0.22 * (1.0 - veg_density) + 0.04 * veg_density + rng.normal(0, 0.008, dims)
    # NIR Band Date 1 (scattered by mesophyll cell structure)
    nir_d1 = 0.20 * (1.0 - veg_density) + 0.62 * veg_density + rng.normal(0, 0.012, dims)
    
    # Water body signature
    red_d1[water_mask] = 0.04
    nir_d1[water_mask] = 0.015

    # 4. Date 2 Disturbance / Evolution Simulation
    red_d2 = red_d1.copy()
    nir_d2 = nir_d1.copy()
    
    if scenario == "deforestation":
        change_mask = np.zeros(dims, dtype=bool)
        
        # Fishbone corridors
        for frac_y in [0.28, 0.68]:
            cy = int(rows * frac_y)
            half_w = max(1, int(rows * 0.015))
            y_start, y_end = max(0, cy - half_w), min(rows, cy + half_w)
            change_mask[y_start:y_end, int(cols * 0.08):int(cols * 0.92)] = True
            
            # Perpendicular cut parcels
            step_x = max(12, int(cols * 0.12))
            parcel_w = max(4, int(step_x * 0.5))
            cut_len = max(8, int(rows * 0.14))
            for off_x in range(int(cols * 0.12), int(cols * 0.88), step_x):
                p_x_end = min(cols, off_x + parcel_w)
                change_mask[max(0, cy - cut_len):min(rows, cy + cut_len), off_x:p_x_end] = True
        
        # Add random illegal deforestation clusters
        num_clusters = max(3, int(min(rows, cols) / 35))
        min_dim = min(rows, cols)
        low_bound = max(2, int(min_dim * 0.1))
        high_bound = max(low_bound + 2, int(min_dim * 0.9))
        
        for _ in range(num_clusters):
            cx = rng.randint(low_bound, max(low_bound + 1, cols - low_bound))
            cy = rng.randint(low_bound, max(low_bound + 1, rows - low_bound))
            radius = max(3, int(min_dim * 0.06))
            dist_sq = (x - cx)**2 + (y - cy)**2
            cluster = dist_sq < (radius ** 2)
            change_mask = change_mask | cluster
            
        change_mask = change_mask & (~water_mask)
        num_changed = int(np.sum(change_mask))
        if num_changed > 0:
            red_d2[change_mask] = 0.26 + rng.normal(0, 0.015, num_changed)
            nir_d2[change_mask] = 0.18 + rng.normal(0, 0.015, num_changed)

    elif scenario == "urban_expansion":
        center_x, center_y = cols * 0.85, rows * 0.85
        dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        expansion_radius = max(15.0, min(rows, cols) * 0.40)
        expansion_zone = (dist_center < expansion_radius) & (~water_mask)
        noise_cut = rng.rand(*dims) > 0.32
        urban_mask = expansion_zone & noise_cut
        num_urban = int(np.sum(urban_mask))
        if num_urban > 0:
            red_d2[urban_mask] = 0.28 + rng.normal(0, 0.01, num_urban)
            nir_d2[urban_mask] = 0.22 + rng.normal(0, 0.01, num_urban)

    elif scenario == "canopy_dieback":
        dieback_mask = (base_landscape > 0.55) & (rng.rand(*dims) > 0.4) & (~water_mask)
        red_d2[dieback_mask] = red_d1[dieback_mask] * 1.5 + 0.05
        nir_d2[dieback_mask] = nir_d1[dieback_mask] * 0.60

    elif scenario == "regrowth":
        regrowth_zone = (base_landscape < 0.45) & (~water_mask) & (rng.rand(*dims) > 0.3)
        red_d2[regrowth_zone] = np.maximum(0.04, red_d1[regrowth_zone] * 0.5)
        nir_d2[regrowth_zone] = np.minimum(0.70, nir_d1[regrowth_zone] * 1.8 + 0.15)
        
    else:  # "seasonal_stable"
        noise_seasonal = rng.normal(0, 0.01, dims)
        red_d2 = np.clip(red_d1 + noise_seasonal * 0.5, 0.01, 1.0)
        nir_d2 = np.clip(nir_d1 + noise_seasonal, 0.01, 1.0)

    # Ensure all reflectance values stay within physical bounds [0.001, 1.0]
    red_d1 = np.clip(red_d1, 0.001, 1.0)
    nir_d1 = np.clip(nir_d1, 0.001, 1.0)
    red_d2 = np.clip(red_d2, 0.001, 1.0)
    nir_d2 = np.clip(nir_d2, 0.001, 1.0)

    return red_d1, nir_d1, red_d2, nir_d2


# -----------------------------------------------------------------------------
# 4. NDVI CALCULATION & SPATIAL CHANGE DETECTION CORE
# -----------------------------------------------------------------------------
def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Calculates Normalized Difference Vegetation Index (NDVI) with
    ZeroDivisionError and floating point anomaly protection.
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    """
    assert nir_band.shape == red_band.shape, "Band dimensions must align perfectly."
    
    denominator = nir_band + red_band
    numerator = nir_band - red_band

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(
            np.abs(denominator) < 1e-7,
            0.0,
            numerator / denominator
        )
    
    # Clip NDVI strictly to theoretical physical limits [-1.0, 1.0]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    # Replace any leftover NaNs or Infs
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
    return ndvi


def classify_vegetation_changes(
    ndvi_d1: np.ndarray,
    ndvi_d2: np.ndarray,
    threshold: float = 0.20,
    pixel_res_meters: float = 10.0
):
    """
    Performs pixel-level bi-temporal change detection and computes
    surface area metrics (Hectares) and categorical masks.
    """
    delta_ndvi = ndvi_d2 - ndvi_d1
    
    # Categorical Masks
    loss_mask = delta_ndvi < -threshold
    gain_mask = delta_ndvi > threshold
    stable_mask = (~loss_mask) & (~gain_mask)

    total_pixels = delta_ndvi.size
    loss_pixels = int(np.sum(loss_mask))
    gain_pixels = int(np.sum(gain_mask))
    stable_pixels = int(np.sum(stable_mask))

    # Pixel area calculation (Sentinel-2 standard is 10m x 10m = 100 sq meters = 0.01 Hectares)
    sq_m_per_pixel = pixel_res_meters * pixel_res_meters
    ha_per_pixel = sq_m_per_pixel / 10000.0

    total_area_ha = total_pixels * ha_per_pixel
    loss_area_ha = loss_pixels * ha_per_pixel
    gain_area_ha = gain_pixels * ha_per_pixel
    stable_area_ha = stable_pixels * ha_per_pixel

    loss_pct = (loss_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    gain_pct = (gain_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    stable_pct = (stable_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    mean_ndvi_d1 = float(np.mean(ndvi_d1))
    mean_ndvi_d2 = float(np.mean(ndvi_d2))
    mean_delta = float(np.mean(delta_ndvi))

    stats = {
        "total_pixels": total_pixels,
        "total_area_ha": round(total_area_ha, 2),
        "loss_pixels": loss_pixels,
        "loss_area_ha": round(loss_area_ha, 2),
        "loss_pct": round(loss_pct, 2),
        "gain_pixels": gain_pixels,
        "gain_area_ha": round(gain_area_ha, 2),
        "gain_pct": round(gain_pct, 2),
        "stable_pixels": stable_pixels,
        "stable_area_ha": round(stable_area_ha, 2),
        "stable_pct": round(stable_pct, 2),
        "mean_ndvi_d1": round(mean_ndvi_d1, 3),
        "mean_ndvi_d2": round(mean_ndvi_d2, 3),
        "mean_delta": round(mean_delta, 3),
        "net_change_ha": round(gain_area_ha - loss_area_ha, 2)
    }

    return delta_ndvi, loss_mask, gain_mask, stable_mask, stats


# -----------------------------------------------------------------------------
# 5. FOLIUM RASTER OVERLAY & COLORMAP CONVERTERS
# -----------------------------------------------------------------------------
def array_to_folium_overlay_url(array: np.ndarray, colormap_name="RdYlGn", vmin=-1.0, vmax=1.0) -> str:
    """
    Transforms a continuous 2D numpy array into a base64 encoded PNG data URI
    using a Matplotlib colormap for clean Folium ImageOverlay injection.
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap_name)
    rgba = cmap(norm(array))
    
    # Scale to 0-255 uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def classified_mask_to_folium_url(
    loss_mask: np.ndarray,
    gain_mask: np.ndarray,
    stable_mask: np.ndarray,
    loss_color=(230, 57, 70, 210),    # Red
    gain_color=(42, 157, 143, 210),   # Green
    stable_color=(180, 180, 180, 40)  # Subtle translucent gray
) -> str:
    """
    Creates an RGB-A image for the 3-class change detection map.
    """
    rows, cols = loss_mask.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    
    rgba[loss_mask] = loss_color
    rgba[gain_mask] = gain_color
    rgba[stable_mask] = stable_color

    img = Image.fromarray(rgba, mode="RGBA")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def render_folium_geospatial_map(
    bounds,
    center,
    zoom,
    ndvi_d1_url,
    ndvi_d2_url,
    change_url,
    aoi_name="Target AOI"
):
    """
    Constructs an interactive Folium Map with side-by-side / toggleable
    layers: Basemap, Baseline NDVI, Monitoring NDVI, and Classified ΔNDVI.
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    image_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True
    )
    
    # Basemap options
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Basemap (Esri)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="CartoDB Clean Light",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap Streets",
        overlay=False,
        control=True
    ).add_to(m)

    # Overlays
    folium.raster_layers.ImageOverlay(
        image=change_url,
        bounds=image_bounds,
        name="ΔNDVI Change Map (Red: Loss | Green: Gain)",
        opacity=0.88,
        interactive=True,
        cross_origin=False,
        zindex=3
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=ndvi_d2_url,
        bounds=image_bounds,
        name="Date 2 NDVI (Monitoring)",
        opacity=0.80,
        interactive=True,
        cross_origin=False,
        show=False,
        zindex=2
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=ndvi_d1_url,
        bounds=image_bounds,
        name="Date 1 NDVI (Baseline)",
        opacity=0.80,
        interactive=True,
        cross_origin=False,
        show=False,
        zindex=1
    ).add_to(m)

    # AOI Bounding Box outline
    folium.Rectangle(
        bounds=image_bounds,
        color="#00d2ff",
        weight=2.5,
        fill=False,
        dash_array="5, 5",
        tooltip=f"AOI: {aoi_name}"
    ).add_to(m)

    # Leaflet Plugins
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="meters").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m


# -----------------------------------------------------------------------------
# 6. HISTOGRAM & VISUALIZATION PLOT GENERATORS
# -----------------------------------------------------------------------------
def plot_ndvi_distribution_comparison(ndvi_d1, ndvi_d2, stats):
    """
    Renders Matplotlib histogram & density curve comparing Date 1 vs Date 2 NDVI.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    d1_flat = ndvi_d1.flatten()
    d2_flat = ndvi_d2.flatten()

    bins = np.linspace(-0.5, 1.0, 60)

    ax.hist(d1_flat, bins=bins, alpha=0.55, color="#2b6cb0", label=f"Date 1 (Baseline: μ = {stats['mean_ndvi_d1']})", density=True)
    ax.hist(d2_flat, bins=bins, alpha=0.55, color="#e53e3e", label=f"Date 2 (Monitoring: μ = {stats['mean_ndvi_d2']})", density=True)

    ax.axvline(stats['mean_ndvi_d1'], color="#2b6cb0", linestyle="--", linewidth=1.5)
    ax.axvline(stats['mean_ndvi_d2'], color="#e53e3e", linestyle="--", linewidth=1.5)

    ax.set_title("Spectral NDVI Distribution Shift (Bi-Temporal)", fontsize=11, fontweight="bold", pad=10, color="#1e293b")
    ax.set_xlabel("NDVI (Normalized Difference Vegetation Index)", fontsize=9, color="#475569")
    ax.set_ylabel("Frequency Density", fontsize=9, color="#475569")
    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8.5)
    
    plt.tight_layout()
    return fig


def plot_classification_pie_chart(stats):
    """
    Renders clean donut chart for the land cover transition proportions.
    """
    fig, ax = plt.subplots(figsize=(5, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    labels = ["Vegetation Loss", "Stable Cover", "Vegetation Gain"]
    sizes = [stats["loss_pct"], stats["stable_pct"], stats["gain_pct"]]
    colors = ["#e63946", "#cbd5e1", "#2a9d8f"]
    explode = (0.08, 0, 0)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )

    for text in texts:
        text.set_fontsize(8.5)
        text.set_color("#1e293b")
        text.set_fontweight("medium")
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title("Land Cover Area Classification (%)", fontsize=11, fontweight="bold", pad=10, color="#1e293b")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 7. MAIN APPLICATION CONTROLLER
# -----------------------------------------------------------------------------
def main():
    init_page_styling()

    # --- HEADER SECTION ---
    st.markdown("""
    <div class="main-header">
        <div class="brand-pill">NeuraMorphix • Problem Statement CS12</div>
        <h1 class="header-title">Monitoring Vegetation Changes using Satellite Imagery</h1>
        <p class="header-sub">
            Autonomous Bi-Temporal Sentinel-2 NDVI Analytics, Deforestation Risk Mapping & Geospatial Compliance Engine.
            <br>
            <span class="meta-tag">HackForge 2026</span>
            <span class="meta-tag">Team Tech Squad</span>
            <span class="meta-tag">SRM Institute of Science and Technology</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR INTERFACE ---
    st.sidebar.header("🛰️ Geospatial Parameters")
    st.sidebar.caption("Configure target area of interest and spectral change sensitivities.")

    # Location Selector
    aoi_mode = st.sidebar.radio("AOI Selection Mode", ["Pre-configured Bounding Boxes", "Custom Coordinates Input"])

    if aoi_mode == "Pre-configured Bounding Boxes":
        selected_aoi_name = st.sidebar.selectbox("Select Target Region", list(PRESET_AOIS.keys()))
        aoi_data = PRESET_AOIS[selected_aoi_name]
        bounds = aoi_data["bounds"]
        center = aoi_data["center"]
        zoom = aoi_data["zoom"]
        default_scenario = aoi_data["default_scenario"]
        st.sidebar.info(f"📌 **Description:** {aoi_data['description']}")
    else:
        selected_aoi_name = "Custom Area of Interest"
        c1, c2 = st.sidebar.columns(2)
        min_lat = c1.number_input("Min Latitude", value=12.80, format="%.4f")
        max_lat = c2.number_input("Max Latitude", value=12.90, format="%.4f")
        min_lon = c1.number_input("Min Longitude", value=80.00, format="%.4f")
        max_lon = c2.number_input("Max Longitude", value=80.10, format="%.4f")
        bounds = (min_lat, min_lon, max_lat, max_lon)
        center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
        zoom = 12
        default_scenario = "deforestation"

    # Date Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Observation Epochs")
    col_d1, col_d2 = st.sidebar.columns(2)
    date_1 = col_d1.date_input("Date 1 (Baseline)", value=datetime.date(2024, 1, 15))
    date_2 = col_d2.date_input("Date 2 (Monitoring)", value=datetime.date(2026, 1, 15))

    if date_1 >= date_2:
        st.sidebar.warning("⚠️ Baseline Date 1 should precede Monitoring Date 2.")

    # NDVI Threshold Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Detection Thresholds")
    ndvi_threshold = st.sidebar.slider(
        "NDVI Sensitivity Threshold (ΔNDVI)",
        min_value=0.05,
        max_value=0.40,
        value=0.20,
        step=0.01,
        help="Pixels with (NDVI_Date2 - NDVI_Date1) < -Threshold are classified as Vegetation Loss. Above +Threshold are classified as Gain."
    )

    alert_loss_threshold = st.sidebar.slider(
        "Automated Risk Alert Trigger (% Loss)",
        min_value=5.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="Triggers an immediate compliance warning if the percentage of deforestation exceeds this value."
    )

    # Simulation Scenario / Data Source Fallback
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Simulation & Sensor Options")
    scenario_list = ["deforestation", "urban_expansion", "canopy_dieback", "regrowth", "seasonal_stable"]
    scenario_index = scenario_list.index(default_scenario) if default_scenario in scenario_list else 0
    selected_scenario = st.sidebar.selectbox(
        "Landscape Dynamic Signature",
        scenario_list,
        index=scenario_index,
        help="Selects the simulated bi-temporal disturbance profile for the synthetic Sentinel-2 array generator."
    )

    sensor_res = st.sidebar.selectbox("Spatial Ground Resolution", ["10m (Sentinel-2 Native)", "20m (Aggregated)"], index=0)
    pixel_res = 10.0 if "10m" in sensor_res else 20.0

    st.sidebar.markdown("---")
    st.sidebar.caption("NeuraMorphix v2.6 | Geospatial AI Prototype")

    # --- PROCESS PIPELINE ---
    with st.spinner("Fetching Sentinel-2 bands and calculating bi-temporal NDVI analytics..."):
        # 1. Generate / Fetch bands
        red_d1, nir_d1, red_d2, nir_d2 = generate_synthetic_sentinel_bands(
            dims=(280, 280),
            scenario=selected_scenario,
            seed=int(abs(bounds[0] * 100) + abs(bounds[1] * 100)) % 1000
        )

        # 2. Compute NDVI for both observation dates
        ndvi_d1 = calculate_ndvi(nir_d1, red_d1)
        ndvi_d2 = calculate_ndvi(nir_d2, red_d2)

        # 3. Classify changes and compute area metrics
        delta_ndvi, loss_mask, gain_mask, stable_mask, stats = classify_vegetation_changes(
            ndvi_d1=ndvi_d1,
            ndvi_d2=ndvi_d2,
            threshold=ndvi_threshold,
            pixel_res_meters=pixel_res
        )

        # 4. Generate visual overlay data URLs for Folium
        ndvi_d1_url = array_to_folium_overlay_url(ndvi_d1, colormap_name="RdYlGn", vmin=-0.2, vmax=0.85)
        ndvi_d2_url = array_to_folium_overlay_url(ndvi_d2, colormap_name="RdYlGn", vmin=-0.2, vmax=0.85)
        change_url = classified_mask_to_folium_url(loss_mask, gain_mask, stable_mask)

    # --- RISK & ALERT SYSTEM BANNER ---
    is_critical_alert = stats["loss_pct"] >= alert_loss_threshold

    if is_critical_alert:
        st.markdown(f"""
        <div class="alert-banner-critical">
            <h4 style="margin:0 0 6px 0; font-weight:700; display:flex; align-items:center;">
                🚨 CRITICAL COMPLIANCE ALERT: Severe Vegetation Loss Detected!
            </h4>
            <p style="margin:0; font-size:0.92rem;">
                <strong>Warning:</strong> Deforestation / loss rate within the monitored bounding box has reached 
                <strong>{stats['loss_pct']}%</strong> ({stats['loss_area_ha']} ha), breaching the configured environmental safety threshold of 
                <strong>{alert_loss_threshold}%</strong>. Immediate ground verification and regulatory reporting recommended under CS12 environmental enforcement protocols.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner-normal">
            <h4 style="margin:0 0 6px 0; font-weight:700; display:flex; align-items:center;">
                ✅ Environmental Status Monitored: Within Permitted Variance
            </h4>
            <p style="margin:0; font-size:0.92rem;">
                Vegetation loss rate is currently <strong>{stats['loss_pct']}%</strong> ({stats['loss_area_ha']} ha), which is below the critical alert threshold of 
                <strong>{alert_loss_threshold}%</strong>. Canopy dynamics reflect stable or recovering conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- KEY METRICS DASHBOARD (4 KPI CARDS) ---
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #e63946;">
            <div class="kpi-label">Vegetation Loss</div>
            <div class="kpi-value" style="color: #e63946;">{stats['loss_area_ha']} <span style="font-size:1rem; color:#64748b;">ha</span></div>
            <div class="kpi-sub" style="color: #e63946;">🔻 {stats['loss_pct']}% of total area</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #2a9d8f;">
            <div class="kpi-label">Vegetation Gain</div>
            <div class="kpi-value" style="color: #2a9d8f;">{stats['gain_area_ha']} <span style="font-size:1rem; color:#64748b;">ha</span></div>
            <div class="kpi-sub" style="color: #2a9d8f;">🌱 +{stats['gain_pct']}% regrowth</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        net_color = "#2a9d8f" if stats['net_change_ha'] >= 0 else "#e63946"
        net_prefix = "+" if stats['net_change_ha'] >= 0 else ""
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {net_color};">
            <div class="kpi-label">Net Surface Balance</div>
            <div class="kpi-value" style="color: {net_color};">{net_prefix}{stats['net_change_ha']} <span style="font-size:1rem; color:#64748b;">ha</span></div>
            <div class="kpi-sub" style="color: #64748b;">Mean ΔNDVI: {stats['mean_delta']:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        alert_badge_color = "#e53e3e" if is_critical_alert else "#38a169"
        alert_text = "TRIGGERED" if is_critical_alert else "NORMAL"
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {alert_badge_color};">
            <div class="kpi-label">Risk Status</div>
            <div class="kpi-value" style="color: {alert_badge_color};">{alert_text}</div>
            <div class="kpi-sub" style="color: #64748b;">Threshold: {alert_loss_threshold}% Loss</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE MAP & SPATIAL VISUALIZATION ---
    col_map, col_charts = st.columns([1.65, 1.0])

    with col_map:
        st.subheader("🗺️ High-Resolution Bi-Temporal Spatial Change Map")
        st.caption("Layered interactive Folium map georeferenced to the bounding box. Use the top-right layer control to toggle between baseline, monitoring, and classified masks.")

        folium_map = render_folium_geospatial_map(
            bounds=bounds,
            center=center,
            zoom=zoom,
            ndvi_d1_url=ndvi_d1_url,
            ndvi_d2_url=ndvi_d2_url,
            change_url=change_url,
            aoi_name=selected_aoi_name
        )

        st_folium(folium_map, width="100%", height=520)

        # Legend Box
        st.markdown("""
        <div class="legend-box">
            <div style="font-weight:700; margin-bottom:6px; color:#1e293b;">Classification Map Legend:</div>
            <div style="display:flex; flex-wrap:wrap; gap:16px;">
                <div class="legend-item"><span class="legend-color" style="background:#e63946;"></span><strong>Vegetation Loss</strong> (ΔNDVI &lt; -Threshold)</div>
                <div class="legend-item"><span class="legend-color" style="background:#2a9d8f;"></span><strong>Vegetation Gain / Regrowth</strong> (ΔNDVI &gt; +Threshold)</div>
                <div class="legend-item"><span class="legend-color" style="background:#cbd5e1;"></span><strong>Stable Cover</strong> (Within Threshold)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_charts:
        st.subheader("📊 Spectral Analytics & Distributions")
        st.caption(f"Comparing NDVI response curves: {date_1.strftime('%b %Y')} vs {date_2.strftime('%b %Y')}")

        # Histogram Distribution
        hist_fig = plot_ndvi_distribution_comparison(ndvi_d1, ndvi_d2, stats)
        st.pyplot(hist_fig, use_container_width=True)

        # Land Cover Transition Pie Chart
        pie_fig = plot_classification_pie_chart(stats)
        st.pyplot(pie_fig, use_container_width=True)

    # --- COMPLIANCE REPORT & EXPORT SECTION ---
    st.markdown("---")
    st.subheader("📋 Geospatial Compliance & Audit Trail Report")
    st.caption("Generate verifiable environmental reports for land administration, climate tracking, and carbon footprint analysis (Deck Slide 6).")

    summary_df = pd.DataFrame([
        {"Metric": "Target Area of Interest", "Value": selected_aoi_name},
        {"Metric": "Geographic Bounding Box", "Value": f"[{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]"},
        {"Metric": "Baseline Epoch (Date 1)", "Value": str(date_1)},
        {"Metric": "Monitoring Epoch (Date 2)", "Value": str(date_2)},
        {"Metric": "Total AOI Monitored Area", "Value": f"{stats['total_area_ha']:,} ha ({stats['total_pixels']:,} pixels)"},
        {"Metric": "Spatial Ground Resolution", "Value": f"{pixel_res}m per pixel"},
        {"Metric": "NDVI Loss / Gain Threshold", "Value": f"±{ndvi_threshold:.2f}"},
        {"Metric": "Vegetation Loss / Deforestation", "Value": f"{stats['loss_area_ha']} ha ({stats['loss_pct']}%)"},
        {"Metric": "Vegetation Gain / Reforestation", "Value": f"{stats['gain_area_ha']} ha ({stats['gain_pct']}%)"},
        {"Metric": "Stable / Unchanged Canopy", "Value": f"{stats['stable_area_ha']} ha ({stats['stable_pct']}%)"},
        {"Metric": "Net Surface Change", "Value": f"{stats['net_change_ha']} ha"},
        {"Metric": "Baseline Mean NDVI", "Value": f"{stats['mean_ndvi_d1']}"},
        {"Metric": "Monitoring Mean NDVI", "Value": f"{stats['mean_ndvi_d2']}"},
        {"Metric": "Mean Spectral Shift (ΔNDVI)", "Value": f"{stats['mean_delta']:+.3f}"},
        {"Metric": "Environmental Risk Level", "Value": "CRITICAL RISK" if is_critical_alert else "STABLE / LOW RISK"}
    ])

    col_tbl, col_dl = st.columns([2, 1])

    with col_tbl:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col_dl:
        st.markdown("#### 📥 Export Metrics")
        st.write("Download formatted compliance data for reporting and GIS archiving:")

        # CSV Download
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download Metrics (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"NeuraMorphix_Report_{date_1}_{date_2}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Plain-Text Audit Report
        audit_report = f"""================================================================================
NEURAMORPHIX: SATELLITE VEGETATION MONITORING COMPLIANCE AUDIT
Problem Statement: CS12 (Geospatial Environment) | HackForge 2026
Team Tech Squad: SRM Institute of Science and Technology
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
================================================================================

1. EXECUTIVE SUMMARY:
   Area of Interest: {selected_aoi_name}
   Bounding Box: Lat [{bounds[0]:.4f} to {bounds[2]:.4f}], Lon [{bounds[1]:.4f} to {bounds[3]:.4f}]
   Baseline Date (T1): {date_1}
   Monitoring Date (T2): {date_2}
   Sensors: Sentinel-2 Optical (B4 Red ~665nm, B8 NIR ~842nm at {pixel_res}m)

2. CORE CHANGE DETECTION METRICS:
   - Total Monitored Area: {stats['total_area_ha']} ha ({stats['total_pixels']} pixels)
   - Vegetation Loss: {stats['loss_area_ha']} ha ({stats['loss_pct']}%)
   - Vegetation Gain: {stats['gain_area_ha']} ha ({stats['gain_pct']}%)
   - Stable Canopy: {stats['stable_area_ha']} ha ({stats['stable_pct']}%)
   - Net Surface Delta: {stats['net_change_ha']} ha

3. SPECTRAL INDICES:
   - Mean NDVI T1: {stats['mean_ndvi_d1']}
   - Mean NDVI T2: {stats['mean_ndvi_d2']}
   - Mean Delta (T2 - T1): {stats['mean_delta']:+.3f}
   - Sensitivity Threshold: ±{ndvi_threshold:.2f}

4. REGULATORY COMPLIANCE STATUS:
   Alert Trigger Setpoint: {alert_loss_threshold}% loss
   Status: {"CRITICAL WARNING - ACTION REQUIRED" if is_critical_alert else "COMPLIANT / WITHIN STABLE THRESHOLD"}

5. STAKEHOLDER BENEFIT CHANNELS (Deck Slide 6):
   - Real Estate & Land Administration: Low-cost illegal clearing verification.
   - Climate Action: Carbon sink tracking and deforestation footprint audit.
   - Wildlife & Ecosystems: Habitat integrity monitoring.

================================================================================
Generated by NeuraMorphix Automated Pipeline. All rights reserved.
================================================================================
"""
        st.download_button(
            label="📝 Download Audit Report (TXT)",
            data=audit_report,
            file_name=f"NeuraMorphix_Audit_Report_{date_1}_{date_2}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.info("💡 **Tip:** Pre-calculated spectral metrics and automated alerts overcome big data transmission latency (Deck Slide 5).")


if __name__ == "__main__":
    main()
