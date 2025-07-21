# Temporal Decay of Spontaneous Map Similarity – Wakefulness (Unfiltered)

## Overview

This analysis quantifies how **map similarity changes over time** during **wakefulness**, using the **unfiltered dataset**. All electrodes were included in the pipeline, and spontaneous maps were inferred using optimal parameters.

---

## Data

- **Condition:** Wakefulness
- **Sessions:** 4 recordings, each 10 minutes long
- **Dataset variant:** Unfiltered (all electrodes)
- **Signal:** nLFP extracted from LFP data

---

## Preprocessing

1. **Segmentation**
   - Each 10-minute recording was split into **five 2-minute segments**.

---

## Analysis Pipeline

1. **Spontaneous Map Inference**
   - Maps decoded from segments using optimal grid search parameters.

2. **Alignment and RMSE**
   - Alignment of each map pair
   - RMSE calculated for all unique segment pairs.

3. **Temporal Distance Association**
   - RMSEs labeled with segment separation (2–8 minutes).

4. **Aggregation**
   - RMSEs pooled across recordings.
   - All RMSEs aggregated and summarized per time lag.

---

## Visualization

- RMSE vs. **temporal distance** using bar graph with lines depicting standard error
---
## Workflow
- Generate array objects using the `main_temporal_stability.py`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_mean_rmse_bar.py`