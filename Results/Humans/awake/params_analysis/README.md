# Analysis of Spontaneous Map Stability in Human Wakefulness Recordings

## Overview

This analysis investigates the stability of **orientation preference maps** inferred from spontaneous activity (nLFP) during **wakefulness** in human subjects. Data from **four separate recording sessions**, each lasting **10 minutes**, were used. The analysis follows the same pipeline as the sleep data, focusing on the consistency of inferred maps across different segments of the same recording, using combinations of detection thresholds, bin sizes, and PCA dimensions.

---

## Data

- **Source:** Human LFP recordings during wakefulness.
- **Sessions:** 4 recordings, each 10 minutes long.
- **Signal type used:** nLFP (negative Local Field Potentials), extracted from LFP data.

---

## Preprocessing

1. **Segmentation**
   - Each 10-minute recording was split into **200-second segments**.

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
   - Before computing RMSE, maps were **aligned** to minimize RMSE.

3. **Aggregation**
   - For each combination of:
     - nLFP threshold,
     - bin size,
     - and PCA dimensions pair,
   - The **mean RMSE** and **standard error** were computed across recordings.

---

## Visualization

- **Heatmap visualization** was used to present the RMSE results.
- **X-axis:** PCA dimension pairs (e.g., 1,2; 2,3; …).
- **Y-axis:** Bin sizes.
- **Facets:** Different thresholds for nLFP detection.

---

## Workflow

- Generate array objects using `main_bin_size_stability.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_average_heatmap_param.py`

---

## Notes

- This analysis complements the sleep dataset, helping to assess whether the stability of spontaneous maps is preserved under wakefulness conditions using the same computational pipeline.
