# Temporal Decay of Spontaneous Map Similarity – Sleep (Unfiltered)

## Overview

This analysis investigates how the **similarity of spontaneous orientation preference maps** decays over time during **sleep**, using the **full (unfiltered)** dataset. All electrodes were included in the analysis. The goal is to assess the **temporal stability** of maps without any quality-based restrictions on electrode selection.

---

## Data

- **Condition:** Sleep
- **Sessions:** 6 recordings, each 10 minutes long
- **Dataset variant:** Unfiltered (all electrodes included)
- **Signal:** nLFP extracted from LFP data

---

## Preprocessing

1. **Segmentation**
   - Each 10-minute recording was split into **five non-overlapping 2-minute segments**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - Maps were inferred from each segment using **optimal parameters**

2. **Alignment and RMSE**
   - All **unique segment pairs** within a recording were compared.
   - Maps were **aligned** to each other
   - **RMSE** was computed for each pair.

3. **Temporal Distance Association**
   - Each RMSE value was tagged with the **temporal distance** between segments (2–8 minutes).

4. **Aggregation**
   - RMSE values were aggregated across all recordings.
   - Summary statistics were computed **as a function of temporal distance**.

---
## Visualization

- RMSE vs. **temporal distance** using bar graph with lines depicting standard error
---
## Workflow
- Generate array objects using the `main_temporal_stability.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_mean_rmse_bar.py`