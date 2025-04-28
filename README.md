# Spontaneous Activity Analysis Toolkit

**Authors**: Radoslav Jochman

A comprehensive Python toolkit for analyzing spontaneous neural activity from multichannel recordings, including human visual cortex data.  
This project enables preprocessing, visualization, stability assessment, and comparison of spontaneous patterns across experimental conditions.

---

## 📚 Features

- Load and preprocess raw LFP or spike recordings (`.nix`, `.ns5`, `.ns6`)
- Create **ArrayAnalysis** objects representing electrode array data
- Compute **spontaneous maps** via PCA projection
- Evaluate **stability** of maps across:
  - Bin size
  - Time
  - Data segments
  - Different thresholds
- Compute **RMSE** (Root Mean Squared Error) distances between maps
- Visualize:
  - Spontaneous activity maps
  - RMSE heatmaps and boxplots
  - Success rates related to functional vs spatial distances
- Filter **good channels** based on SNR, presence ratio, firing rates

---

## 🗂 Project Structure

| File | Purpose                                                                                             |
|:-----|:----------------------------------------------------------------------------------------------------|
| `array_analysis.py` | Main `ArrayAnalysis` class (core data structure).                                                   |
| `analysis.py` | Functions for frame extraction, map calculation, PCA analysis.                                      |
| `helper.py` | Utility functions for map alignment, RMSE computation, preprocessing, etc.                          |
| `preprocessing.py` | Raw LFP and spike preprocessing.                                                                    |
| `plotting.py` | Visualizations with `plotnine`.                                                                     |
| Various `main_*.py` scripts | Command-line interfaces for major analysis tasks (bin size stability, RMSE calculations, plotting). |

---

## 📜 Key Pipelines (CLI Scripts)

| Script | What it does |
|:---|:---|
| `main_generate_good_channels.py` | Filters and saves good quality electrodes. |
| `main_generate_arr_obj.py` | Creates a pickled `ArrayAnalysis` object from input data. |
| `main_bin_size_stability.py` | Analyzes spontaneous map stability across different bin sizes. |
| `main_temporal_stability.py` | Splits sessions into time segments and checks map stability. |
| `main_params_analysis.py` | Full parameter sweep (bin size, segment splitting) across recordings. |
| `main_generate_rmse_dist_param.py` | Computes RMSE between samples and a fixed reference across PC pairs. |
| `main_generate_rmse_dist_2samples_param.py` | Computes RMSE between corresponding sample pairs. |
| `main_generate_rmse_dist_in_time.py` | RMSE evolution across time-separated recordings. |
| `main_plot_average_heatmap_param.py` | Plots averaged RMSE heatmaps. |
| `main_plot_heatmap_param.py` | Plots RMSE heatmaps. |
| `main_plot_mean_rmse_by_time.py` | Visualizes RMSE evolution across time. |
| `main_plot_spontaneous_map.py` | Plots spontaneous maps aligned to a reference. |
| `main_plot_success_rate.py` | Visualizes success rate in a discrimination task vs spatial/functional distances. |
| `concat_dataframes.py` | Combines multiple CSV RMSE results into a single file. |

---

## 🛠 Requirements

- Python 3.8+
- Main libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `neo`
  - `plotnine`
  - `scikit-learn`
  - `pyyaml`
  - `quantities`
  - `elephant`
- `blackrock-utilities`

Install all with:

```bash
pip install numpy pandas scipy neo plotnine scikit-learn pyyaml quantities elephant
```
---

## 🚀 Example Usage

**Generate Good Channels:**

```bash
python main_generate_good_channels.py --spikes_dir data/ --snr 5 --presence_ratio 0.95 --result_path results/good_channels.csv
```

**Bin Size Stability:**

```bash
python main_bin_size_stability.py --analysis_params_dir configs/analysis.yaml --params_dir configs/data.yaml --data_dir data/ --result_dir results/bin_stability
```

**Compute RMSE between samples and reference:**

```bash
python main_generate_rmse_dist_param.py --data_location results/bin_stability/ --ref_obj_path results/reference.pkl --PCs 1,2,3,4 --sample_name Subject1 --result_dir results/rmse
```

**Plot RMSE heatmap:**

```bash
python main_plot_heatmap_param.py --data_location results/rmse/Subject1.csv --result_dir figures/ --result_name heatmap_subject1.png
```

---

## 🧠 Scientific Context

This toolkit is designed primarily for **investigating spontaneous activity structures** in neural populations (e.g., human visual cortex).  
Key scientific concepts it handles:

- **Spontaneous cortical maps** via PCA
- **Decoding and stability** of functional maps
- **Temporal dynamics** of spontaneous representations


