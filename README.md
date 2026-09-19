# 🌱 NeuraMorphix: Monitoring Vegetation Changes using Satellite Imagery
**Problem Statement ID:** CS12 — Geospatial Environment  
**Event:** HackForge 2026  
**Team Name:** Tech Squad  
**Members:** Krishna Saketh, Ayush Pramanik, Adeera Rajput, MVR Tarun  
**Institution:** SRM Institute of Science and Technology  

---

## 📌 Executive Summary
**NeuraMorphix** is an automated geospatial analytics prototype that leverages multi-spectral satellite imagery (Sentinel-2) to detect and quantify bi-temporal canopy dynamics, deforestation, urban sprawl, and ecosystem regeneration. By calculating bi-temporal Normalized Difference Vegetation Index (NDVI) scores and isolating spectral shifts, the application delivers actionable spatial change maps, automated regulatory alerts, and compliance audit exports without requiring manual, cost-prohibitive physical land surveys.

---

## 🚀 Key Features

### 1. Geospatial Area of Interest (AOI) Selector
- **Pre-Configured Global Hotspots:**
  - *Amazon Basin (Rondônia Deforestation Arc, Brazil)* — active agricultural frontier and fishbone logging corridors.
  - *Black Forest (Schwarzwald, Germany)* — temperate coniferous drought stress and bark beetle clearing.
  - *Chennai / SRMIST Peri-Urban Corridor* — suburban infrastructure expansion displacing peri-urban scrubland.
  - *Sundarbans Biosphere Reserve* — mangrove delta dynamics and tidal regeneration.
- **Custom Coordinate Input:** Real-time user input for custom bounding boxes (`Min/Max Latitude` and `Min/Max Longitude`).

### 2. Multi-Spectral Processing Core
- **Physical Band Selection:** Emulates Sentinel-2 Band 4 (Red: ~665 nm) and Band 8 (Near-Infrared: ~842 nm) at 10m ground resolution.
- **Robust NDVI Formula:**
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
  Includes division-by-zero protection (`np.errstate`, epsilon damping, and `np.where`), strictly bounding outputs to $[-1.0, +1.0]$.
- **Synthetic Sentinel-2 Generator:** Physically consistent spatial array generator incorporating terrain gradient models, river channel water absorption, and realistic disturbance signatures (logging corridors, illegal clusters, urban fringe growth, and canopy dieback) so the prototype runs offline and instantly without API downtime.

### 3. Spatial Change Detection & Analytics
- **Change Calculation:** $\Delta\text{NDVI} = \text{NDVI}_{T2} - \text{NDVI}_{T1}$
- **Tri-Category Classification:**
  - 🟥 **Vegetation Loss / Deforestation:** $\Delta\text{NDVI} < -\text{Threshold}$
  - ⬜ **Stable / Unchanged Canopy:** $-\text{Threshold} \le \Delta\text{NDVI} \le +\text{Threshold}$
  - 🟩 **Vegetation Gain / Regrowth:** $\Delta\text{NDVI} > +\text{Threshold}$
- **Geodetic Area Quantities:** Hectares ($ha$) and percentage metrics derived dynamically based on sensor pixel resolution ($10\text{m} \times 10\text{m} = 100\,\text{m}^2 = 0.01\,\text{ha}$).

### 4. Interactive Visualization UI
- **KPI Metrics Dashboard:** 4 prominent cards reporting Area Lost ($ha$ and %), Area Gained ($ha$ and %), Net Surface Balance ($ha$), and Environmental Risk Status.
- **Interactive Multi-Layer Folium Map:**
  - Georeferenced raster overlays for Baseline NDVI ($T_1$), Monitoring NDVI ($T_2$), and Classified Change Mask.
  - Dynamic basemaps: *Esri World Imagery*, *CartoDB Clean Light*, and *OpenStreetMap*.
  - Fullscreen view, metric measurement tool, layer toggle control, and bounding box outline.
- **Spectral Distribution Charts:** Matplotlib histograms comparing bi-temporal NDVI distributions and a donut chart breaking down land cover transitions.

### 5. Risk & Compliance Reporting System
- **Automated Threshold Alert:** Prominent visual warning banner triggered whenever vegetation loss exceeds the regulatory threshold (e.g., $>10\%$).
- **One-Click Export:**
  - `NeuraMorphix_Report.csv` — structured metrics table.
  - `NeuraMorphix_Audit_Report.txt` — compliance audit document complete with UTC timestamps, bounding boxes, sensor metadata, and stakeholder benefit analysis.

---

## 🛠️ Architecture & Tech Stack

```
   [User Input UI] (AOI Bounding Box, Dates T1/T2, NDVI & Alert Thresholds)
          │
          ▼
   [Data Ingestion Core] ───► Synthetic Sentinel-2 Generator (B4 Red & B8 NIR)
          │
          ▼
   [NDVI Analytics Engine] ──► NDVI_T1 & NDVI_T2 Calculation (ZeroDivision Protected)
          │
          ▼
   [Change Classification] ──► ΔNDVI = NDVI_T2 - NDVI_T1 (Loss, Stable, Gain)
          │
          ├──────────────────────────────┬──────────────────────────────┐
          ▼                              ▼                              ▼
 [Interactive Folium Map]      [Matplotlib Analytics]        [Risk & Compliance]
 (Georeferenced ImageOverlay)  (Spectral Histograms & Donut) (Audit Export & Alert)
```

| Layer | Technologies Used |
|---|---|
| **Frontend Framework** | Streamlit |
| **Data Processing & Geospatial** | NumPy, Rasterio, GeoPandas, SciPy, Pandas |
| **Mapping & Visualization** | Folium, streamlit-folium, Matplotlib, Pillow |

---

## 💻 Installation & Quickstart

```bash
# 1. Clone or navigate to the workspace
cd d:\Hackforge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

---

## 👥 Authors
**Team Tech Squad** — SRM Institute of Science and Technology  
- Krishna Saketh
- Ayush Pramanik
- Adeera Rajput
- MVR Tarun
