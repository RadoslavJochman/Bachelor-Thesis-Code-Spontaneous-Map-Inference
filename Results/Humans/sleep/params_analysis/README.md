# Analysis of Spontaneous Map Stability in Human Sleep Recordings

## Overview

This analysis investigates the stability of **orientation preference maps** inferred from spontaneous activity (nLFP) during sleep in human subjects. Data from **six separate recording sessions**, each lasting **10 minutes**, were used. The focus was on evaluating how consistent the inferred maps are across different segments of the same recording, using combinations of detection thresholds, bin sizes, and PCA dimensions.

---

## Data

- **Source:** Human LFP recordings during sleep.
- **Sessions:** 6 recordings, each 10 minutes long.
- **Signal type used:** nLFP (negative Local Field Potentials), extracted from LFP data.

---

## Preprocessing

1. **Segmentation**
   - Each 20-minute recording was split into **200-second segments**.

2. **nLFP Detection**
   - nLFP events were extracted using **multiple thresholds**.

3. **Binning**
   - Events were binned using various **bin sizes**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - For each segment, a **spontaneous orientation map** was computed based on the distribution of nLFP events.
   - Maps were inferred for various combinations of:
     - Detection **thresholds**.
     - **Bin sizes**.
     - **PCA dimensions**, ranging from 1 to 6.

2. **Intra-session Map Comparison**
   - For each recording, **pairwise comparisons** of maps from different segments were performed.
   - **Root Mean Squared Error (RMSE)** was computed between maps.
   - Before computing RMSE, maps were **aligned** to minimize RMSE (to account for spatial shifts or rotations).

3. **Aggregation**
   - For each combination of:
     - nLFP threshold,
     - bin size,
     - and PCA dimensions pair,
   - The **mean RMSE** and **standard error** were computed across recordings.

---

## Visualization

- **Heatmap visualization** was used to present the RMSE results.
- **X-axis:** PCA dimension pairs (e.g., 1,2, 2,3, …).
- **Y-axis:** Bin sizes.
- **Facets:** Different thresholds for nLFP detection.

---
## Workflow
- Generate array objects using the `main_bin_size_stability.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_average_heatmap_param.py`

---
## Notes

- This analysis helps assess the **robustness and reproducibility** of inferred spontaneous maps across different preprocessing and embedding configurations.