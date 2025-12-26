# Urban Heat Island (UHI) Analysis

## Project Overview
This project focuses on analyzing the Urban Heat Island (UHI) effect using satellite imagery and building footprint data. The goal is to understand how urban structures and vegetation cover influence local temperature variations in specific regions.

## Study Areas
The analysis focuses on the following regions:
- **Brazil**
- **Chile**
- **Sierra Leone**

## Data Sources
1.  **Satellite Imagery**: Used to derive Land Surface Temperature (LST) and Spectral Indices.
2.  **Building Footprints**: High-resolution data used to calculate building density and urban geometry.

## Methodology
The project workflow involves:
1.  **Data Preprocessing**: Cleaning and aligning satellite images with building footprint shapefiles.
2.  **Spectral Analysis**: Calculating indices to characterize land cover.
3.  **LST Retrieval**: Converting thermal band data to Land Surface Temperature.
4.  **Statistical Analysis**: Correlating urban density (from footprints) and vegetation cover with temperature.

## Key Outputs & Results

### 1. Spectral Indices (NDVI, NDBI)
*   **Output**: Raster maps representing Normalized Difference Vegetation Index (NDVI) and Normalized Difference Built-up Index (NDBI).
*   **Explanation**:
    *   **NDVI**: Indicates the density of green vegetation. High values suggest healthy vegetation, which typically provides a cooling effect.
    *   **NDBI**: Highlights built-up areas. High values correlate with impervious surfaces that retain heat.

### 2. Land Surface Temperature (LST) Maps
*   **Output**: Thermal maps showing the spatial distribution of surface temperatures across the study areas.
*   **Explanation**: These maps identify "hotspots" where the UHI effect is most intense, typically coinciding with dense urban structures and low vegetation.

### 3. Building Density Analysis
*   **Output**: Aggregated metrics showing building coverage per unit area.
*   **Explanation**: This output quantifies urbanization intensity, serving as a key variable to explain temperature variations.

### 4. Correlation Analysis
*   **Output**: Scatter plots and correlation coefficients (e.g., Pearson's r) between LST, NDVI, and Building Density.
*   **Explanation**:
    *   **LST vs. NDVI**: Typically shows a negative correlation (more vegetation = lower temperature).
    *   **LST vs. Building Density**: Typically shows a positive correlation (more buildings = higher temperature).

## Usage
To replicate this analysis, ensure the datasets listed in `.gitignore` (Spectral Indices, Building Footprints) are placed in the `data/` directory.
