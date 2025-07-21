# Comparison of Spontaneous Maps with Ground Truth in Monkey Recordings (Monkey L)

## Overview

This analysis evaluates how closely **spontaneous orientation preference maps**, inferred from nLFP activity, match the **true orientation preference map** in **monkey L**. The methodology largely mirrors the pipeline used in the spontaneous map stability analysis, with the key difference being the use of a reference “ground truth” map for all comparisons.

---

## Data

- **Source:** LFP recordings from monkey L.
- **Sessions:** 3 recordings, each approximately **1300 seconds** long.
- **Signal type used:** nLFP (negative Local Field Potentials), extracted from raw LFP signals.
- **Ground truth map:** Orientation preference map obtained from evoked activity or external validation.

---

## Preprocessing

1. **Segmentation**
   - Each recording was divided into **200-second segments**.

2. **nLFP Detection**
   - Spontaneous nLFP events were detected using multiple **threshold levels**.

3. **Binning**
   - Detected events were binned using different **bin sizes**

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - For each segment, a spontaneous orientation map was inferred using:
     - Specific **detection thresholds**,
     - A range of **bin sizes**,
     - **PCA dimension** values from 1 to 6.

2. **Comparison to Ground Truth**
   - Each inferred map was:
     - **Globally aligned** to the true map to minimize RMSE,
     - Compared to the true map using **Root Mean Squared Error (RMSE)**.

3. **Aggregation**
   - RMSE values were aggregated across:
     - All segments,
     - All recordings,
     - Each combination of threshold, bin size, and PCA dimension pair.
   - The **mean RMSE** and **standard error** were computed for each configuration.

---

## Visualization

- Results were displayed using **heatmaps**:
  - **X-axis:** PCA dimension pairs (e.g., 1,2; 2,3; …)
  - **Y-axis:** Bin sizes
  - **Facets:** Different thresholds for nLFP detection

---

## Workflow

- Generate spontaneous map data using `main_monkey_params_analysis.py`
- Compute RMSE against the true map using `main_generate_rmse_dist_real_monkey_param.py`
- Concatenate results across recordings using `concat_dataframes.py`
- Visualize using `main_plot_average_heatmap_param.py`

