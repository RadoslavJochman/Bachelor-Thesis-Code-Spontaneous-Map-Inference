# Temporal Decay of Spontaneous Map Similarity – Sleep (Filtered)

## Overview

This analysis investigates the **temporal stability** of spontaneous orientation preference maps during **sleep**, using a **quality-filtered dataset**. Only high-quality electrodes were included, based on defined signal metrics. The analysis reveals how restricting to well-isolated units impacts map consistency over time.

---

## Data

- **Condition:** Sleep
- **Sessions:** 6 recordings, each 10 minutes long
- **Dataset variant:** Filtered (high-quality electrodes only)
- **Signal:** nLFP extracted from LFP data

---

## Preprocessing

1. **Electrode Filtering**
   - Electrodes were selected based on:
     - Signal-to-Noise Ratio (SNR)
     - Presence Ratio
     - Firing Rate
     - ISI Violation Ratio

2. **Segmentation**
   - Each recording was divided into **five 2-minute segments**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - Maps inferred from each segment using **optimal parameters**.

2. **Alignment and RMSE**
   - Each map pair was **aligned**
   - **RMSE** calculated for all unique pairs.

3. **Temporal Distance Association**
   - RMSEs linked to temporal distance (2–8 min).

4. **Aggregation**
   - All RMSEs aggregated and summarized per time lag.

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