"""
Plotting
This script contains functions for
Plotting graph of how RMSE depends on bin size with specific threshold factor and PCs
    -ggplot_average_heatmap_param()
    -ggplot_std_heatmap_param()
Plotting RMSE for different samples
    -ggplot_rmse()
Plotting spontaneous maps
    -ggplot_spontaneous_map()
Author: Radoslav Jochman
"""
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from plotnine.themes.themeable import axis_line_x, legend_text

from array_analysis import *
from scipy.stats import linregress
import helper
import numpy as np
from plotnine import (ggplot, aes, geom_point, geom_smooth, labs, theme_minimal, scale_color_manual,
                      annotate, scale_shape_manual, geom_hline, scale_y_continuous, scale_x_continuous,
                      geom_text,coord_equal, scale_fill_gradientn, theme, element_rect,
                      element_blank, element_text, guides, guide_colorbar, geom_boxplot, facet_grid, geom_tile,
                      facet_wrap,scale_fill_gradient, as_labeller, geom_line, geom_ribbon, scale_color_gradientn,geom_col,
                      geom_errorbar,geom_raster,element_line)

def ggplot_spontaneous_map_human(analysis_array, ref):
    """
    Generates a spontaneous map plot using plotnine, mimicking the style of ggplot2 in Python.

    This function rotates the spontaneous map from the analysis_array to align with the reference
    map provided by ref, then constructs a plot that visualizes electrode orientations:
      - It creates a DataFrame from the rotated map, converting valid orientation values from radians
        to degrees, and another DataFrame for electrode coordinates and labels obtained from a helper.
      - A custom HSV color gradient is generated to map orientation values (0° to 180°) to colors.
      - Axis limits are computed with added margins for a cleaner layout.
      - The plot is assembled with square markers (colored by orientation), electrode labels on top,
        an equal coordinate ratio, and a minimal theme with a vertical color bar legend.

    Parameters:
        analysis_array: An object containing a 'spontaneous_map' attribute (a numpy array of orientation data).
        ref: A reference object that includes a 'spontaneous_map' attribute (a numpy array of orientation data).

    Returns:
        A plotnine ggplot object that can be rendered using .draw() or saved using .save(...).
    """

    # Align spontaneous maps
    ref_map = ref.spontaneous_map
    spont_map = analysis_array.spontaneous_map
    spont_map = helper.find_ideal_rotation(ref_map, spont_map, 5000)

    # Build a DataFrame for valid points:
    # Use indices (iy, ix) as x and y, and convert orientation (in rad) to degrees.
    data_list = []
    for ix in range(spont_map.shape[0]):
        for iy in range(spont_map.shape[1]):
            val = spont_map[ix, iy]
            if val > -0.5:
                orientation_deg = 180.0 * val / np.pi
                data_list.append((iy, ix, orientation_deg))
    df_map = pd.DataFrame(data_list, columns=['x', 'y', 'orientation'])

    # Build a DataFrame for electrode labels
    elec_list = []
    for i in range(96):
        ex, ey = helper.get_coords_from_electrode_human(i + 1)
        if spont_map[int(ey), int(ex)]>-0.5:
            elec_list.append((ex, ey, i + 1))
    df_elec = pd.DataFrame(elec_list, columns=['x', 'y', 'label'])

    # Create a custom HSV gradient for orientation (0 to 180°)
    cmap = plt.cm.get_cmap('hsv')
    n_colors = 256
    hsv_hex = [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]

    # Compute x-axis limits with margin.
    # Use df_map x values, then add 6% extra space on each side.
    x_min = df_map['x'].min()
    x_max = df_map['x'].max()
    x_range = x_max - x_min
    x_margin = 0.06 * x_range if x_range > 0 else 1
    x_limits = (x_min - x_margin, x_max + x_margin)

    # Compute y-axis limits from df_map's y values with a margin.
    y_min = df_map['y'].min()
    y_max = df_map['y'].max()
    y_range = y_max - y_min
    y_margin = 0.06 * y_range if y_range > 0 else 1
    y_limits = (y_min - y_margin, y_max + y_margin)

    # Construct the plot:
    p = (
            ggplot(df_map, aes('x', 'y'))
            + geom_point(shape='s', size=22, color='black')
            + geom_point(aes(fill='orientation'), shape='s', size=21)
            + geom_text(df_elec, aes('x', 'y', label='label'), color='black', size=15)
            + scale_fill_gradientn(
        colors=hsv_hex,
        limits=[0, 180],
        breaks=[0, 45, 90, 135, 180],
        name="Estimated orientation (°)"
    )

        + coord_equal()
        + labs(x="", y="")
        + scale_x_continuous(limits=x_limits, expand=(0, 0))
        + scale_y_continuous(limits=y_limits, expand=(0, 0))
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            panel_grid=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            legend_position='right',
            legend_key_size=40,
            legend_text=element_text(size=12),
            legend_title=element_text(angle=0, size=12, ha='center', va='bottom'),
            figure_size=(8, 8),
                )
        + guides(
            fill=guide_colorbar(direction='vertical'))
    )

    return p

def ggplot_heatmap_param(df_results):
    """
    Expects df_results to have columns:
      - bin_size (float)
      - TH (e.g. -1, -2, -3)
      - PC_pair (string like "0-1", "0-2", etc.)
      - RMSE (float)

    Creates a faceted heatmap:
      - x-axis: bin_size (continuous)
      - y-axis: threshold (discrete)
      - fill: RMSE
      - one facet per PC_pair
    """


    df_plot = df_results.copy()
    df_plot['TH'] = df_plot['TH'].astype(str)

    #Define a limited set of breaks for bin_size ticks
    df_plot["bin_size"] = df_plot["bin_size"].astype(float)
    y_breaks = np.arange(np.min(df_plot["bin_size"]),np.max(df_plot["bin_size"]),0.2)

    #Define labels for facets using TH values
    th_labels = {}
    for TH in df_plot['TH']:
        th_labels[TH] = f"Threshold {TH}"

    cmap = plt.cm.get_cmap('inferno_r')
    n_colors = 256
    hsv_hex = [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]

    p = (
        ggplot(df_plot, aes(x='PC_pair', y='bin_size', fill='RMSE'))
        + geom_tile()
        + facet_wrap('~TH', ncol=len(th_labels.keys()), labeller=as_labeller(th_labels))
        + scale_fill_gradientn(colors=hsv_hex)
        #+ scale_fill_gradient(high="red",low="yellow")
        + scale_y_continuous(breaks=y_breaks)  # fewer x ticks
        + labs(
            title="RMSE over Parameter Grid",
            x="PC pair",
            y="Bin Size",
            fill="RMSE"
        )
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            figure_size=(30, 6),
            axis_text_x=element_text(angle=90, vjust=0.5, hjust=1)
        )
    )
    return p

def ggplot_average_heatmap_param(df_results, title: str):
    """
    Expects df_results to have columns:
      - bin_size (float)
      - TH (e.g. -1, -2, -3)
      - PC_pair (string like "0-1", "0-2", etc.)
      - RMSE (float)

    Creates a faceted heatmap:
      - x-axis: bin_size (continuous)
      - y-axis: threshold (discrete)
      - fill: Mean RMSE
      - one facet per PC_pair
    """


    df_plot = df_results.copy()
    df_plot['TH'] = df_plot['TH'].astype(str)

    #Define a limited set of breaks for bin_size ticks
    df_results=df_results[np.isclose(df_results["bin_size"]*20 % 2, 0,atol=1e-8)]

    y_breaks = np.arange(np.min(df_results["bin_size"])+0.1,np.max(df_results["bin_size"]+0.2),0.2)
    df_summary = (
        df_results.groupby(['bin_size', 'TH', 'PC_pair'], as_index=False)
        .agg(mean_RMSE=('RMSE', 'mean'))
    )
    print(f"Average max {df_summary['mean_RMSE'].max()}\n min {df_summary['mean_RMSE'].min()}")
    #Define labels for facets using TH values
    th_labels = {}
    for TH in df_plot['TH']:
        th_labels[TH] = f"Threshold {TH}"

    cmap = plt.cm.get_cmap('inferno_r')
    n_colors = 256
    hsv_hex = [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]
    p = (
        ggplot(df_summary, aes(x='PC_pair', y='bin_size', fill='mean_RMSE'))
        + geom_raster()
        + facet_wrap('~TH', ncol=len(th_labels.keys()), labeller=as_labeller(th_labels))
        + scale_fill_gradientn(colors=hsv_hex,
                               limits=(0.1, 0.9),
                               breaks=[0.3, 0.5, 0.7])
        #+ scale_fill_gradient(high="red", low="yellow")
        + scale_y_continuous(breaks=y_breaks)  # fewer x ticks
        + labs(
            title=title,
            x="PC pair",
            y="Bin Size [s]",
            fill="Mean RMSE [rad]"
        )
        + theme_minimal()
        + theme(
            panel_border=element_blank(),
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            figure_size=(30, 6),
            legend_title=element_text(rotation=90, va="center_baseline", ha="center", x=-10, size=15),  #x=200,y=-400
            legend_text=element_text(size=11),
            axis_text_x=element_text(angle=90, vjust=0.5, hjust=1, size=11),
            axis_text_y=element_text(size=11),
            axis_title_x = element_text(size=20),
            axis_title_y = element_text(size=20),
            strip_text_x=element_text(size=20)

        )
    )
    return p

def ggplot_std_heatmap_param(df_results, title: str):
    """
    Expects df_results to have columns:
      - bin_size (float)
      - TH (e.g. -1, -2, -3)
      - PC_pair (string like "0-1", "0-2", etc.)
      - RMSE (float)

    Creates a faceted heatmap:
      - x-axis: bin_size (continuous)
      - y-axis: threshold (discrete)
      - fill: Mean RMSE
      - one facet per PC_pair
    """


    df_plot = df_results.copy()
    df_plot['TH'] = df_plot['TH'].astype(str)

    #Define a limited set of breaks for bin_size ticks
    df_results=df_results[np.isclose(df_results["bin_size"]*20 % 2, 0,atol=1e-8)]

    y_breaks = np.arange(np.min(df_results["bin_size"])+0.1,np.max(df_results["bin_size"]+0.2),0.2)

    df_summary = df_results.groupby(['bin_size', 'TH', 'PC_pair'], as_index=False).agg(
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        count=("RMSE", "count")
    )

    # Calculate standard error
    df_summary['se'] = df_summary['std_RMSE'] / (df_summary['count'] ** 0.5)
    print(f"Standard error max {df_summary['se'].max()}\n min {df_summary['se'].min()}")
    #Define labels for facets using TH values
    th_labels = {}
    for TH in df_plot['TH']:
        th_labels[TH] = f"Threshold {TH}"

    cmap = plt.cm.get_cmap('inferno_r')
    n_colors = 256
    hsv_hex = [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]
    p = (
        ggplot(df_summary, aes(x='PC_pair', y='bin_size', fill='se'))
        + geom_raster()
        + facet_wrap('~TH', ncol=len(th_labels.keys()), labeller=as_labeller(th_labels))
        + scale_fill_gradientn(colors=hsv_hex,
                               limits=(0, 0.1),
                               breaks=[0.02, 0.04, 0.06, 0.08])
        #+ scale_fill_gradient(high="red", low="yellow")
        + scale_y_continuous(breaks=y_breaks)  # fewer x ticks
        + labs(
            title=title,
            x="PC pair",
            y="Bin Size [s]",
            fill="Standard error of RMSE [rad]",

        )
        + theme_minimal()
        + theme(
            panel_border=element_blank(),
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            figure_size=(30, 6),
            legend_title=element_text(rotation=90, va="center_baseline", ha="center", x=-13, size=15),  #,y=-400
            legend_text=element_text(size=11),
            axis_text_x=element_text(angle=90, vjust=0.5, hjust=1,size=11),
            axis_text_y=element_text(size=11),
            axis_title_x = element_text(size=20),
            axis_title_y = element_text(size=20),
            strip_text_x=element_text(size=20)

        )
    )
    return p

def ggplot_lineband_param(df_results):
    """
    Expects a DataFrame with columns:
      - bin_size: numeric (e.g., 0.15, 0.20, ..., 5.0)
      - TH: numeric or categorical (e.g., -1, -2, -3)
      - PC_pair: string (e.g., "0-1", "0-2", etc.)
      - RMSE: numeric error value
      - sample: identifier for each sample (so there can be multiple RMSE values per combination)

    This function groups the data by bin_size, TH, and PC_pair, computes the mean RMSE
    and the standard error (SE) across samples, then plots the mean RMSE with error bars.

    Returns a plotnine object.
    """

    # Group by parameters and compute summary statistics
    df_summary = (
        df_results.groupby(['bin_size', 'TH', 'PC_pair'], as_index=False)
        .agg(mean_RMSE=('RMSE', 'mean'),
             sd_RMSE=('RMSE', 'std'),
             n_samples=('RMSE', 'count'))
    )
    # Compute standard error
    df_summary['se_RMSE'] = df_summary['sd_RMSE'] / df_summary['n_samples'] ** 0.5

    # Define x-axis breaks (you can adjust as needed)
    x_breaks = np.arange(np.min(df_summary["bin_size"]), np.max(df_summary["bin_size"]), 0.5)
    #df_summary = df_summary[df_summary["PC_pair"].isin(["0,1", "0,2", "0,3","0,4","1,2","1,3","1,4","2,3",""])]

    p = (ggplot(df_summary, aes(x='bin_size'))
         # Ribbon for the uncertainty band:
         + geom_ribbon(aes(ymin='mean_RMSE - sd_RMSE', ymax='mean_RMSE + sd_RMSE'),
                       fill="blue", alpha=0.3)
         # Line for the mean RMSE:
         + geom_line(aes(y='mean_RMSE'), color='blue', size=0.8)
         + facet_grid('TH ~ PC_pair',)
         + scale_x_continuous(breaks=x_breaks)
         + labs(title="Mean RMSE with Uncertainty Band",
                x="Bin Size",
                y="Mean RMSE")
         + theme_minimal()
         + theme(axis_text_x=element_text(angle=90, hjust=1, vjust=0.5),
                 panel_background=element_text(fill='white'),
                 plot_background=element_text(fill='white'),
                 legend_position='none',
                 figure_size=(24, 8)
                 )
    )
    return p

def ggplot_success_rate(df):
    """
    Expects a DataFrame with columns:
      - spatial_distance: Numeric values for spatial distance (x-axis)
      - functional_distance: Numeric values for functional distance (y-axis)
      - success_rate: Numeric values (0 to 1) mapped to color

    Returns a plotnine object with a scatter plot:
      - x-axis: Spatial distance
      - y-axis: Functional distance
      - Color: Success rate using a viridis-like continuous color scale
    """
    # Create a viridis colormap using matplotlib and convert it to a list of hex colors.
    n_colors = 256
    cmap = plt.cm.get_cmap('viridis', n_colors)
    viridis_hex = [mcolors.to_hex(cmap(i / n_colors)) for i in range(n_colors)]

    p = (ggplot(df, aes(x='spatial_distance', y='functional_distance', color='success_rate'))
         + geom_point(size=3)
         + scale_color_gradientn(colors=viridis_hex, limits=(0, 1))
         + labs(x="Spatial distance", y="Functional distance", color="Success rate")
         + theme_minimal()
         + theme(
                panel_background=element_text(fill='white'),
                plot_background=element_text(fill='white'),
            )
         )
    return p

def ggplot_mean_rmse_bar(df: pd.DataFrame):
    """
    Plots the mean RMSE for each time_diff as a bar chart with error bars.

    Parameters:
        df (pd.DataFrame): DataFrame with columns "sample", "time_diff", and "RMSE".

    Returns:
        plotnine.ggplot: A ggplot object representing the bar chart.
    """
    # Group by time_diff and calculate mean, standard deviation, and count
    df_summary = df.groupby("time_diff", as_index=False).agg(
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        count=("RMSE", "count")
    )
    # Calculate standard error
    df_summary['se'] = df_summary['std_RMSE'] / (df_summary['count'] ** 0.5)

    # Create the bar chart with error bars
    p = (ggplot(df_summary, aes(x="time_diff", y="mean_RMSE"))
         + geom_col(fill="skyblue")  # Creates the bars
         + geom_errorbar(aes(ymin="mean_RMSE - se", ymax="mean_RMSE + se"), width=0.2)
         + labs(
                title="",
                x="Temporal distance [min]",
                y="Mean RMSE [rad]"
            )
         + scale_y_continuous(limits=(0,0.07),breaks=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
         + theme(
                panel_background=element_rect(fill='white'),
                plot_background=element_rect(fill='white'),
                panel_border=element_rect(color='black', fill=None, size=1),
                figure_size=(30, 6),
                axis_text_x=element_text(size=25),
                axis_text_y=element_text(size=25),
                axis_title_x=element_text(size=25),
                axis_title_y=element_text(size=25),
                strip_text_x=element_text(size=25)
            )
         )

    return p