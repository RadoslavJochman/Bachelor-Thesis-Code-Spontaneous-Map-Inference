# Temporal Decay of Spontaneous Map Similarity – Wakefulness (Filtered)

## Overview

This analysis measures the **temporal decay** of spontaneous map similarity during **wakefulness**, using a **quality-filtered electrode subset**. Only units meeting specific criteria were included, ensuring high signal fidelity.

---

## Data

- **Condition:** Wakefulness
- **Sessions:** 4 recordings, each 10 minutes long
- **Dataset variant:** Filtered (high-quality electrodes)
- **Signal:** nLFP extracted from LFP data

---

## Preprocessing

1. **Electrode Filtering**
   - Selection based on:
     - Signal-to-Noise Ratio (SNR)
     - Presence Ratio
     - Firing Rate
     - ISI Violation Ratio

2. **Segmentation**
   - Each recording split into **five non-overlapping 2-minute segments**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - Maps inferred using optimal parameters from grid search.

2. **Alignment and RMSE**
   - Maps aligned to a reference before RMSE computation.

3. **Temporal Distance Association**
   - RMSE values assigned to segment separation (2–8 minutes).

4. **Aggregation**
   - RMSEs aggregated across recordings and grouped by time lag.

---
## Visualization

- RMSE vs. **temporal distance** using bar graph with lines depicting standard error
---
## Workflow
- Select only high-quality electrodes using `main_generate_good_channels.py`
- Generate array objects using the `main_temporal_stability.py` with the `--good_channels`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_mean_rmse_bar.py`