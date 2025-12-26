# %%
import sys
import subprocess
import os

# FIX: Handle PROJ_LIB conflict with PostGIS
for env_var in ['PROJ_LIB', 'PROJ_DATA']:
    if env_var in os.environ and 'postgresql' in os.environ[env_var].lower():
        print(f"Removing conflicting {env_var} pointing to PostgreSQL")
        del os.environ[env_var]

try:
    import stackstac
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pystac-client", "planetary-computer", "stackstac", "rioxarray", "xarray", "dask"])

# %%
"""
Urban Heat Island Classification - Pipeline
Full Pipeline with Visualization and Model Evaluation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Core STAC Libraries
import pystac_client
import stackstac
import rioxarray
from dask.diagnostics import ProgressBar

# ML Libraries
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# %%
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_cloud_mask(stack, method='scl', strategy='temporal_median'):
    """
    Modular cloud masking function for Sentinel-2 data.
    
    Args:
        stack (xarray.DataArray): Input stack with bands and time dimension.
        method (str): 'scl' uses the Scene Classification Layer (standard for L2A).
        strategy (str): 
            - 'nan': Replace cloud pixels with NaN.
            - 'temporal_median': Replace with NaN (median calculation handles the rest).
            - 'interpolate': Interpolate NaN values over time.
            
    Returns:
        xarray.DataArray: Cloud-masked stack.
    """
    print(f"  ☁️ Applying cloud mask (Method: {method}, Strategy: {strategy})...")
    
    if method == 'scl':
        if 'scl' not in stack.band.values:
            print("    ⚠️ SCL band not found. Skipping masking.")
            return stack
            
        scl = stack.sel(band='scl')
        # SCL Classes to mask: 3 (Shadow), 8 (Medium), 9 (High), 10 (Cirrus)
        # We keep: 4 (Vegetation), 5 (Bare Soil), 6 (Water), 7 (Unclassified)
        cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
        
        # Apply mask (set clouds to NaN)
        stack = stack.where(~cloud_mask)
        
    elif method == 's2cloudless':
        print("    ⚠️ s2cloudless requires local processing. Using SCL (provided by AWS) is recommended for this pipeline.")

    if strategy == 'interpolate':
        stack = stack.interpolate_na(dim='time', method='linear')
        
    return stack

def download_sentinel_data(city_name, gdf, folder, force=False):
    """
    Downloads Sentinel-2 data from AWS (Element84) to avoid token requirements.
    Calculates indices (NDVI, NDBI, NDWI) and saves them.
    """
    # Check if data already exists
    indices_check = ["NDVI", "NDBI", "NDWI"]
    if not force and all(os.path.exists(os.path.join(folder, f"{city_name}_{idx}.tif")) for idx in indices_check):
        print(f"✓ Data for {city_name} already exists. Skipping download.")
        return True

    print(f"\n{'='*70}")
    print(f"DOWNLOADING SENTINEL-2 DATA: {city_name}")
    print(f"{'='*70}")

    # 1. Search Data
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    
    band_mapping = {
        "red": "B04",
        "green": "B03",
        "blue": "B02",
        "nir": "B08",
        "swir16": "B11"
    }
    
    # Add SCL for cloud masking
    assets = list(band_mapping.keys()) + ["scl"]
    
    bbox = tuple(gdf.total_bounds)
    print(f"Bounding Box: {bbox}")

    # Search for items
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2024-01-01/2024-05-30",
        query={"eo:cloud_cover": {"lt": 20}}
    )

    items = search.item_collection()
    
    if len(items) == 0:
        print(f"⚠️  No items found for {city_name} in this date range.")
        return None

    print(f"✓ Found {len(items)} Sentinel-2 scenes")
    print(f"  Date range: {items[0].datetime.date()} to {items[-1].datetime.date()}")

    # 2. Create Stack
    print(f"  Creating data stack...")
    data = stackstac.stack(
        items,
        assets=assets, 
        chunksize=4096,
        resolution=10,
        bounds_latlon=bbox,
        epsg=3857,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(dict(AWS_NO_SIGN_REQUEST="YES"))
    )

    # 3. Rename Bands
    new_band_names = []
    for b in data.band.values.tolist():
        if b in band_mapping:
            new_band_names.append(band_mapping[b])
        else:
            new_band_names.append(b) # Keep 'scl' as is
    data = data.assign_coords(band=new_band_names)

    # 4. Apply Cloud Masking
    # NOTE: Masking must happen BEFORE the composite step so clouds are treated as missing data (NaN).
    # The median function in step 5 will then ignore these NaNs and pick clear pixels.
    data = apply_cloud_mask(data, method='scl', strategy='temporal_median')

    # 5. Create Median Composite & Indices
    print(f"  Computing median composite and spectral indices...")
    
    with ProgressBar():
        median = data.median(dim="time").compute()

    # Calculate Indices with epsilon to prevent division by zero
    epsilon = 1e-10
    b03 = median.sel(band="B03")
    b04 = median.sel(band="B04")
    b08 = median.sel(band="B08")
    b11 = median.sel(band="B11")

    ndvi = (b08 - b04) / (b08 + b04 + epsilon)
    ndbi = (b11 - b08) / (b11 + b08 + epsilon)
    ndwi = (b03 - b08) / (b03 + b08 + epsilon)

    # Clip values to valid range [-1, 1]
    ndvi = ndvi.clip(-1, 1)
    ndbi = ndbi.clip(-1, 1)
    ndwi = ndwi.clip(-1, 1)

    # 5. Save to Disk
    os.makedirs(folder, exist_ok=True)
    indices = {"NDVI": ndvi, "NDBI": ndbi, "NDWI": ndwi}
    
    for name, array in indices.items():
        out_path = os.path.join(folder, f"{city_name}_{name}.tif")
        
        if array.rio.crs is None:
            array.rio.write_crs(data.rio.crs, inplace=True)
            
        array.rio.to_raster(out_path)
        print(f"  ✅ Saved: {city_name}_{name}.tif")

    print(f"{'='*70}\n")
    return True


def visualize_tiff_indices(city_name, folder, base_path):
    """
    Visualize the downloaded TIFF files with proper symbology
    """
    print(f"\n📸 Visualizing spectral indices for {city_name}...")
    
    indices = ['NDVI', 'NDBI', 'NDWI']
    tiff_files = [os.path.join(folder, f"{city_name}_{idx}.tif") for idx in indices]
    
    # Check if all files exist
    if not all(os.path.exists(f) for f in tiff_files):
        print(f"  ⚠️ Some TIFF files not found for {city_name}")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define color schemes and ranges for each index
    index_configs = {
        'NDVI': {'cmap': 'RdYlGn', 'vmin': -1, 'vmax': 1, 'title': 'NDVI (Vegetation Index)'},
        'NDBI': {'cmap': 'RdYlBu_r', 'vmin': -1, 'vmax': 1, 'title': 'NDBI (Built-up Index)'},
        'NDWI': {'cmap': 'Blues', 'vmin': -1, 'vmax': 1, 'title': 'NDWI (Water Index)'}
    }
    
    for idx, (index_name, tiff_path) in enumerate(zip(indices, tiff_files)):
        ax = axes[idx]
        config = index_configs[index_name]
        
        # Read the TIFF file
        with rasterio.open(tiff_path) as src:
            data = src.read(1)  # Read first band
            
            # Mask nodata values
            data = np.ma.masked_equal(data, src.nodata)
            
            # Plot
            im = ax.imshow(data, cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'])
            ax.set_title(f"{config['title']}\n{city_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Column (pixels)', fontsize=10)
            ax.set_ylabel('Row (pixels)', fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(index_name, fontsize=10, fontweight='bold')
            
            # Add statistics text
            valid_data = data.compressed()  # Get non-masked values
            if len(valid_data) > 0:
                stats_text = f"Min: {valid_data.min():.3f}\nMean: {valid_data.mean():.3f}\nMax: {valid_data.max():.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, family='monospace')
    
    plt.tight_layout()
    output_path = os.path.join(base_path, f'TIFF_Visualization_{city_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  ✅ Saved: TIFF_Visualization_{city_name}.png")


def extract_features(gdf, folder, force_processing=False):
    """Loops through points and extracts mean values from TIFs."""
    city_name = gdf['city'].iloc[0]
    
    # Ensure data is available (download/process if missing or forced)
    download_sentinel_data(city_name, gdf, folder, force=force_processing)
    
    for idx_name in ['NDVI', 'NDBI', 'NDWI']:
        tif_path = os.path.join(folder, f"{city_name}_{idx_name}.tif")
        
        if os.path.exists(tif_path):
            print(f"  📊 Extracting {idx_name} for {city_name}...")
            with rasterio.open(tif_path) as src:
                # Transform entire GDF to raster CRS once for performance
                # We assume raster is in a metric CRS (like 3857) for buffering
                gdf_raster = gdf.to_crs(src.crs)
                points_features = []
                
                for _, row in tqdm(gdf_raster.iterrows(), total=len(gdf_raster), leave=False, desc=f"  {idx_name}"):
                    # Buffer directly in raster CRS
                    geom_in_raster_crs = row.geometry.buffer(BUFFER_DISTANCE)
                    
                    try:
                        out_image, _ = mask(src, [mapping(geom_in_raster_crs)], crop=True, nodata=src.nodata)
                        val = out_image[out_image != src.nodata].mean()
                        points_features.append(val)
                    except:
                        points_features.append(np.nan)
                
                gdf[f'{idx_name.lower()}_mean'] = points_features
    return gdf


def prepare_features(gdf, feature_cols):
    """Selects feature columns from GeoDataFrame, handling missing columns."""
    return gdf.reindex(columns=feature_cols, fill_value=np.nan)


def clean_data(df):
    """Replace inf/-inf with NaN and clip extreme values"""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            mean = df[col].mean()
            std = df[col].std()
            
            if pd.notna(mean) and pd.notna(std) and std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def to_gdf(df, name):
    """Convert DataFrame to GeoDataFrame"""
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    gdf['city'] = name
    if 'ID' in df.columns:
        gdf['ID'] = df['ID']
    return gdf


# %%
# ============================================================================
# SECTION 1: SETTINGS & PATHS
# ============================================================================
print("\n" + "="*70)
print("SECTION 1: URBAN HEAT ISLAND CLASSIFICATION")
print("="*70)

BASE_PATH = r"F:\Jupyter Notebooks\EY Urban Heat Island Challenge\data"
SENTINEL_PATH = BASE_PATH 
CRS_STORAGE = "EPSG:4326"
CRS_ANALYSIS = "EPSG:3857"
BUFFER_DISTANCE = 100 

BRAZIL_UHI = os.path.join(BASE_PATH, "Sample_Brazil_uhi_data.csv")
CHILE_UHI = os.path.join(BASE_PATH, "sample_chile_uhi_data.csv")
Freetown_VAL = os.path.join(BASE_PATH, "Test.csv")

print(f"\n📁 Data Directory: {BASE_PATH}")
print(f"🌍 Analysis CRS: {CRS_ANALYSIS}")
print(f"📏 Buffer Distance: {BUFFER_DISTANCE}m")

# %%
# ============================================================================
# SECTION 2: DATA LOADING & INITIAL EXPLORATION
# ============================================================================

print("\n" + "="*70)
print("SECTION 2: LOADING & EXPLORING DATA")
print("="*70)

gdfs = []
data_sources = [(BRAZIL_UHI, "Rio_de_Janeiro"), (CHILE_UHI, "Santiago"), (Freetown_VAL, "Freetown")]

for path, name in tqdm(data_sources, desc="Loading Datasets"):
    df = pd.read_csv(path)
    gdfs.append(to_gdf(df, name))

brazil_gdf, chile_gdf, freetown_gdf = gdfs

# FIX: Ensure UHI_Class is integer encoded (0=Low, 1=Medium, 2=High)
for gdf in [brazil_gdf, chile_gdf]:
    if 'UHI_Class' in gdf.columns and gdf['UHI_Class'].dtype == 'object':
        gdf['UHI_Class'] = gdf['UHI_Class'].replace({'Low': 0, 'Medium': 1, 'High': 2})

# Print dataset statistics
print("\n📊 DATASET STATISTICS:")
print(f"{'City':<20} {'Total Points':<15} {'Columns':<10}")
print("-" * 50)
print(f"{'Rio de Janeiro':<20} {len(brazil_gdf):<15} {len(brazil_gdf.columns):<10}")
print(f"{'Santiago':<20} {len(chile_gdf):<15} {len(chile_gdf.columns):<10}")
print(f"{'Freetown (Test)':<20} {len(freetown_gdf):<15} {len(freetown_gdf.columns):<10}")

# Visualize UHI class distribution for training data
if 'UHI_Class' in brazil_gdf.columns and 'UHI_Class' in chile_gdf.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Brazil
    brazil_counts = brazil_gdf['UHI_Class'].value_counts().sort_index()
    axes[0].bar(brazil_counts.index, brazil_counts.values, color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    axes[0].set_title('UHI Class Distribution - Rio de Janeiro', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UHI Class (0=Low, 1=Medium, 2=High)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(brazil_counts.values):
        axes[0].text(i, v + 200, str(v), ha='center', fontweight='bold')
    
    # Chile
    chile_counts = chile_gdf['UHI_Class'].value_counts().sort_index()
    axes[1].bar(chile_counts.index, chile_counts.values, color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    axes[1].set_title('UHI Class Distribution - Santiago', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UHI Class (0=Low, 1=Medium, 2=High)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(chile_counts.values):
        axes[1].text(i, v + 200, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, '01_UHI_Class_Distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ UHI class distribution visualization saved as '01_UHI_Class_Distribution.png'")

# %%
# ============================================================================
# SECTION 3: SATELLITE DATA PROCESSING & FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("SECTION 3: PROCESSING SATELLITE DATA & EXTRACTING FEATURES")
print("="*70)

processed_gdfs = []
for gdf in [brazil_gdf, chile_gdf, freetown_gdf]:
    processed_gdfs.append(extract_features(gdf, SENTINEL_PATH, force_processing=False))

brazil_gdf, chile_gdf, freetown_gdf = processed_gdfs

print("\n✓ Feature Extraction Complete")

# %%
# ============================================================================
# VISUALIZE DOWNLOADED TIFF FILES
# ============================================================================
print("\n" + "="*70)
print("SECTION 3.1: VISUALIZING DOWNLOADED SATELLITE IMAGERY")
print("="*70)

for city_name in ["Rio_de_Janeiro", "Santiago", "Freetown"]:
    visualize_tiff_indices(city_name, SENTINEL_PATH, BASE_PATH)

print("\n✓ All TIFF visualizations complete")

# Display extracted features summary
print("\n📊 EXTRACTED FEATURES SUMMARY:")
feature_cols = ['ndvi_mean', 'ndbi_mean', 'ndwi_mean']
for city, gdf in [('Rio de Janeiro', brazil_gdf), ('Santiago', chile_gdf), ('Freetown', freetown_gdf)]:
    print(f"\n{city}:")
    if all(col in gdf.columns for col in feature_cols):
        print(gdf[feature_cols].describe().round(4))
    else:
        print("  Features not yet extracted")

# Visualize spectral indices
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
datasets = [('Rio de Janeiro', brazil_gdf), ('Santiago', chile_gdf)]
indices = ['ndvi_mean', 'ndbi_mean', 'ndwi_mean']
index_titles = ['NDVI (Vegetation)', 'NDBI (Built-up)', 'NDWI (Water)']

for col_idx, (idx_name, idx_title) in enumerate(zip(indices, index_titles)):
    for row_idx, (city, gdf) in enumerate(datasets):
        ax = axes[row_idx, col_idx]
        if idx_name in gdf.columns:
            gdf.plot(column=idx_name, ax=ax, legend=True, cmap='RdYlGn', markersize=1, 
                    legend_kwds={'label': idx_title, 'orientation': 'horizontal'})
            ax.set_title(f'{idx_title} - {city}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=10)
            ax.set_title(f'{idx_title} - {city}', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, '02_Spectral_Indices_Spatial.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Spectral indices spatial visualization saved as '02_Spectral_Indices_Spatial.png'")

# %%
# ============================================================================
# SECTION 4: MODEL TRAINING & CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("SECTION 4: MODEL TRAINING & EVALUATION")
print("="*70)

FEATURE_COLS = [
    'ndvi_mean', 'ndbi_mean', 'ndwi_mean',
    'building_count', 'building_area_total', 'building_height_mean', 'building_volume_proxy'
]

# Combine training data
train_gdf = pd.concat([brazil_gdf, chile_gdf], ignore_index=True)

X_train = prepare_features(train_gdf, FEATURE_COLS)
y_train = train_gdf['UHI_Class']
X_val = prepare_features(freetown_gdf, FEATURE_COLS)

# Clean data
print("\n🧹 Cleaning data...")
X_train_clean = clean_data(X_train)
X_val_clean = clean_data(X_val)

print(f"\nTraining set:")
print(f"  - Total samples: {len(X_train_clean)}")
print(f"  - Features: {X_train_clean.shape[1]}")
print(f"  - NaN count: {X_train_clean.isna().sum().sum()}")
print(f"  - Inf count: {np.isinf(X_train_clean).sum().sum()}")

print(f"\nValidation set:")
print(f"  - Total samples: {len(X_val_clean)}")
print(f"  - Features: {X_val_clean.shape[1]}")
print(f"  - NaN count: {X_val_clean.isna().sum().sum()}")
print(f"  - Inf count: {np.isinf(X_val_clean).sum().sum()}")

# Build and train model
print("\n🤖 Training Gradient Boosting Classifier...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=0))
])

pipeline.fit(X_train_clean, y_train)
print("✓ Model training complete")

# Cross-validation on training data
print("\n📊 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(pipeline, X_train_clean, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Scores: {cv_scores.round(4)}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Training set predictions for evaluation
y_train_pred = pipeline.predict(X_train_clean)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\n📈 TRAINING SET PERFORMANCE:")
print(f"Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
cm = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'], 
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix - Training Set', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, '03_Confusion_Matrix_Training.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Confusion matrix saved as '03_Confusion_Matrix_Training.png'")

# Feature importance
feature_importance = pipeline.named_steps['clf'].feature_importances_
feature_names = [col for col in FEATURE_COLS if col in X_train_clean.columns]
importance_df = pd.DataFrame({
    'Feature': feature_names[:len(feature_importance)],
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n📊 FEATURE IMPORTANCE RANKING:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance in UHI Classification', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, '04_Feature_Importance.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Feature importance saved as '04_Feature_Importance.png'")

#%%
# ============================================================================
# SECTION 5: PREDICTIONS ON TEST SET & EXPORT
# ============================================================================

print("\n" + "="*70)
print("SECTION 5: MAKING PREDICTIONS ON Freetown TEST SET")
print("="*70)

print("\n🔮 Making predictions...")
freetown_gdf['UHI_Class'] = pipeline.predict(X_val_clean)

# Rename ID column
if 'ID' in freetown_gdf.columns:
    freetown_gdf.rename(columns={'ID': 'id'}, inplace=True)
    print("✓ Renamed 'ID' column to 'id'")
elif 'id' not in freetown_gdf.columns:
    freetown_gdf['id'] = freetown_gdf.index
    print("✓ Created 'id' column from index")

# Map UHI_Class to Target labels
label_mapping = {
    0: 'Low',
    1: 'Medium',
    2: 'High'
}
freetown_gdf['Target'] = freetown_gdf['UHI_Class'].map(label_mapping)

# Validation
if freetown_gdf['Target'].isnull().any():
    missing_count = freetown_gdf['Target'].isnull().sum()
    print(f"WARNING: {missing_count} rows could not be mapped! Check 'UHI_Class' for unexpected values.")
else:
    print("Success: All classes mapped correctly.")

# Export predictions
output_path = os.path.join(BASE_PATH, "Final_Submission.csv")
try:
    freetown_gdf[['id', 'Target']].to_csv(output_path, index=False)
    print(f"\n✅ Submission file saved to: {output_path}")
except PermissionError:
    print(f"\n⚠️  Permission denied: '{output_path}' is likely open in another program.")
    # Save with a timestamp to avoid the lock
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(BASE_PATH, f"Final_Submission_{timestamp}.csv")
    freetown_gdf[['id', 'Target']].to_csv(output_path, index=False)
    print(f"✅ Submission file saved to: {output_path}")

print(freetown_gdf[['id', 'Target']].head())

# Prediction summary
print("\n📊 PREDICTION SUMMARY (Freetown Test Set):")
pred_counts = freetown_gdf['UHI_Class'].value_counts().sort_index()
print("\nClass Distribution:")
for cls, count in pred_counts.items():
    class_name = ['Low', 'Medium', 'High'][int(cls)]
    percentage = (count / len(freetown_gdf)) * 100
    print(f"  {class_name:8} (Class {cls}): {count:5} points ({percentage:5.2f}%)")
print(f"\nTotal predictions: {len(freetown_gdf)}")

# Visualize predictions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart
colors = ['green', 'orange', 'red']
axes[0].bar(pred_counts.index, pred_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].set_title('Predicted UHI Class Distribution - Freetown', fontsize=14, fontweight='bold')
axes[0].set_xlabel('UHI Class (0=Low, 1=Medium, 2=High)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['Low', 'Medium', 'High'])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(pred_counts.values):
    percentage = (v / len(freetown_gdf)) * 100
    axes[0].text(i, v + 100, f'{v}\n({percentage:.1f}%)', ha='center', fontweight='bold', fontsize=10)

# Spatial distribution
freetown_gdf.plot(column='UHI_Class', ax=axes[1], legend=True, cmap='RdYlGn_r', 
                   markersize=3, categorical=True,
                   legend_kwds={'title': 'UHI Class'})
axes[1].set_title('Spatial Distribution of Predicted UHI Classes - Freetown', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Longitude', fontsize=11)
axes[1].set_ylabel('Latitude', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, '05_Freetown_Predictions.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Prediction visualizations saved as '05_Freetown_Predictions.png'")

# %%
# ============================================================================
# SECTION 6: FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("PIPELINE EXECUTION SUMMARY")
print("="*70)

print("\n✅ All sections completed successfully!")
print("\n📊 Model Performance Summary:")
print(f"  - Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  - Training Set Accuracy: {train_accuracy:.4f}")
print(f"  - Test Set Predictions: {len(freetown_gdf)} points")

print("\n📁 Generated Output Files:")
files = [
    "01_UHI_Class_Distribution.png - Training data class distribution",
    "02_Spectral_Indices_Spatial.png - Spatial maps of NDVI, NDBI, NDWI",
    "03_Confusion_Matrix_Training.png - Model performance confusion matrix",
    "04_Feature_Importance.png - Feature importance ranking",
    "05_Freetown_Predictions.png - Test set prediction results",
    "TIFF_Visualization_Rio_de_Janeiro.png - Raw satellite imagery for Rio",
    "TIFF_Visualization_Santiago.png - Raw satellite imagery for Santiago",
    "TIFF_Visualization_Freetown.png - Raw satellite imagery for Freetown",
    "Final_Submission.csv - Submission file with predictions"
]

for i, file in enumerate(files, 1):
    print(f"  {i}. {file}")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE! Ready for submission.")
print("="*70 + "\n")
# %%
