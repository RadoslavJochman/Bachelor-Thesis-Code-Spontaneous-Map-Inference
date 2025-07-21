# Map Stability in Monkey L (Eyes Opened)

## Overview

This analysis investigates the stability of **orientation preference maps** inferred from spontaneous activity in Monkey L, considering only time periods when the **monkey's eyes were opened**. The goal is to assess how segment length and parameter settings influence the similarity of inferred maps.

---

## Data

* **Source:** LFP recordings from Monkey L
* **Sessions:** 3 recordings, each approximately **1300 seconds** long
* **Condition:** Eye state = **opened**

---

## Preprocessing

1. **Eye State Filtering**

   * Segments were extracted from the parts of the recordings when the monkey's **eyes were opened**.

2. **Segmentation**

   * The filtered data was split into non-overlapping segments of the following lengths:

     * **50, 100, 150, 200, and 250 seconds**

3. **nLFP Detection and Binning**

   * Events were detected using multiple **thresholds**.
   * Events were binned using different **bin sizes**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**

   * For each segment, a **spontaneous orientation preference map** was inferred using PCA on the binned nLFP data.
   * Maps were computed for various combinations of:

     * Detection **thresholds**
     * **Bin sizes**
     * **PCA dimensions** (1 through 6)

2. **Intra-segment Map Comparison**

   * For each segment length, all possible pairs of maps were compared within each recording.
   * **Root Mean Squared Error (RMSE)** was calculated between the maps.
   * Maps were globally **aligned** before RMSE computation to minimize differences due to spatial shift or rotation.

3. **Aggregation**

   * RMSE values were aggregated across sessions.
   * For each segment length and parameter combination, the **mean RMSE** and **standard error** were calculated.

---

## Visualization

* Results visualized as **heatmaps**:

  * **X-axis:** PCA dimension pairs
  * **Y-axis:** Bin sizes
  * **Facets:** Detection thresholds
  * Separate panels for each **segment length**
---
## Workflow

- Generate array objects using `main_monkey_params_analysis.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_average_heatmap_param.py`
