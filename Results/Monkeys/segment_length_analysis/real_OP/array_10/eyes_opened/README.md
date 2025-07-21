# # Comparison of Spontaneous Maps with Ground Truth in Monkey Recordings Monkey L (Eyes Open)

## Overview

This analysis evaluates how closely spontaneous orientation preference maps inferred from short segments match the **true orientation preference map**. The data used consists only of time periods when **Monkey L's eyes were open**.

---

## Data

* **Source:** LFP recordings from Monkey L
* **Sessions:** 3 recordings, each approximately **1300 seconds** long
* **Condition:** Eye state = **open**
* **Reference:** True orientation preference map available for comparison

---

## Preprocessing

1. **Eye State Filtering**

   * Extracted only the periods during which the monkey's **eyes were open**.

2. **Segmentation**

   * Filtered data was divided into non-overlapping segments of:

     * **50, 100, 150, 200, and 250 seconds**

3. **nLFP Detection and Binning**

   * Events detected using multiple **thresholds**
   * Events binned using a range of **bin sizes**

---

## Analysis Pipeline

1. **Spontaneous Map Inference**

   * Each segment was used to infer a spontaneous orientation preference map using PCA.
   * Maps were computed using different combinations of:

     * Detection **thresholds**
     * **Bin sizes**
     * **PCA dimensions** (1 through 6)

2. **RMSE to True Map Comparison**

   * Each inferred map was compared to the known **true orientation preference map**.
   * RMSE was computed after aligning maps to minimize shifts or rotations.

3. **Aggregation**

   * RMSE values were averaged across recordings.
   * For each parameter set and segment length, **mean RMSE** and **standard error** were calculated.

---

## Visualization

* **Heatmaps** used to display RMSE between inferred and true maps:

  * **X-axis:** PCA dimension
  * **Y-axis:** Bin size
  * **Facets:** Detection thresholds
  * Panels grouped by **segment length**

---
## Workflow

- Generate spontaneous map data using `main_monkey_params_analysis.py`
- Compute RMSE against the true map using `main_generate_rmse_dist_real_monkey_param.py`
- Concatenate results across recordings using `concat_dataframes.py`
- Visualize using `main_plot_average_heatmap_param.py`
