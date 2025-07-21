# Analysis of Spontaneous Map Stability in Monkey Recordings (Monkey L)

## Overview

This analysis investigates the **stability of orientation preference maps** inferred from spontaneous activity (nLFP) in **monkey L array 10**. The pipeline replicates the analysis previously applied to human recordings, aiming to assess the consistency of inferred maps across time segments, while varying detection thresholds, bin sizes, and PCA dimensionality.

---

## Data

- **Source:** LFP recordings from monkey L.
- **Sessions:** 3 recordings, each approximately **1300 seconds** long.
- **Signal type used:** nLFP (negative Local Field Potentials), extracted from raw LFP signals.

---

## Preprocessing

1. **Segmentation**
   - Each recording was split into **200-second segments**.

2. **nLFP Detection**
   - Negative LFP events were detected using multiple **threshold values**.

3. **Binning**
   - Events were binned using a range of **bin sizes**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - For each segment, an **orientation preference map** was inferred using:
     - Varying **thresholds**,
     - **Bin sizes**,
     - **PCA dimensions** from 1 to 6.

2. **Intra-session Map Comparison**
   - Within each recording, all unique segment pairs were compared using:
     - **Global alignment** of maps
     - **Root Mean Squared Error (RMSE)** to quantify dissimilarity.

3. **Aggregation**
   - For each combination of threshold, bin size, and PCA dimension pair:
     - The **mean RMSE** and **standard error** were computed across all three recordings.

---

## Visualization

- RMSE values were visualized using **heatmaps**:
  - **X-axis:** PCA dimension pairs (e.g., 1,2; 2,3; …)
  - **Y-axis:** Bin sizes
  - **Facets:** Different thresholds for nLFP detection

---

## Workflow

- Generate array objects using `main_monkey_params_analysis.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_average_heatmap_param.py`


