# Analysis of Spontaneous Map Stability in Human Sleep Recordings

## Overview

This analysis investigates the stability of **orientation preference maps** inferred from spontaneous activity (nLFP) during **sleep** in human subjects, using only **high-quality electrodes**, selected by specific signal metrics.

---

## Data

- **Source:** Human LFP recordings during sleep.
- **Sessions:** 6 recordings, each 10 minutes long.
- **Signal type used:** nLFP (negative Local Field Potentials), extracted from LFP data.

---

## Preprocessing

### 1. Electrode Quality Filtering
The following filters were applied:
- **Signal-to-Noise Ratio (SNR)**
- **Presence Ratio**
- **Firing Rate**
- **ISI Violation Ratio**

### 2. Standard Pipeline

- **Segmentation:** 200-second segments.
- **nLFP Detection:** Using multiple thresholds.
- **Binning:** Using various bin sizes.

---

## Analysis Pipeline

- **Spontaneous Map Inference:** For each segment, maps were inferred using different:
  - Thresholds
  - Bin sizes
  - PCA dimensions (1 to 6)

- **Intra-session Comparison:**
  - Pairwise **RMSE** between maps from different segments.
  - Maps were **aligned** before RMSE to minimize spatial mismatch.

- **Aggregation:**
  - Mean RMSE and standard error were calculated for each parameter combination.

---

## Visualization

- **Heatmaps** were used for results.
  - **X-axis:** PCA dimension pairs (e.g., 1,2; 2,3; …)
  - **Y-axis:** Bin sizes
  - **Facets:** Detection thresholds

---

## Workflow
- Select only high-quality electrodes using `main_generate_good_channels.py`
- Generate array objects using the `main_bin_size_stability.py` with the `--good_channels`
- Calculate the RMSE for each sample using `main_generate_rmse_dist_2samples_param.py`
- Concatenate `.csv` files from all samples together using `concat_dataframes.py`
- Plot the heatmaps using `main_plot_average_heatmap_param.py`

---

## Notes
- Comparison between full-electrode and filtered analyses reveals the impact of signal quality.