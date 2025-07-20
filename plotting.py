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

def ggplot_rmse_vs_bin_size(df: pd.DataFrame):
    """
    Generates a ggplot graph showing RMSE vs. bin size with a linear regression line.

    Parameters:
        df (pd.DataFrame):
            DataFrame containing at least the following columns:
                - 'rmse' (float): RMSE values plotted on the y-axis.
                - 'bin_size' (float): Bin sizes plotted on the x-axis.
            Optionally, df can include:
                - 'TH' (int or str): Threshold factor values (1, 2, 3) represented as shapes (-1: circle, -2: square, -3: triangle).
                - 'PC' (tuple or str): Principal component pairs represented as colors.

    Returns:
        plotnine.ggplot: ggplot object visualizing RMSE against bin sizes, including:
            - Scatter points colored by 'PC' if provided.
            - Scatter points shaped by 'TH' if provided.
            - Linear regression line (red) with R² annotation displayed.
    """
    # Ensure 'bin_size' and 'rmse' are numeric
    df['bin_size'] = df['bin_size'].astype(float)
    df['rmse'] = df['rmse'].astype(float)

    # Compute regression parameters for annotation
    x = df['bin_size'].values
    y = df['rmse'].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value ** 2

    # Determine annotation coordinates (95% of max to place in top right)
    annot_x = df['bin_size'].max() * 0.95
    annot_y = df['rmse'].max() * 0.95
    annotation = annotate(
        "text",
        x=annot_x,
        y=annot_y,
        label=f"(R² = {r2:.3f})",
        ha="right",
        va="top",
        size=10
    )

    # Build the aes mapping based on available columns
    if 'PC' in df.columns and 'TH' in df.columns:
        mapping = aes(x='bin_size', y='rmse', color='PC', shape='TH')
    elif 'PC' in df.columns:
        mapping = aes(x='bin_size', y='rmse', color='PC')
    elif 'TH' in df.columns:
        mapping = aes(x='bin_size', y='rmse', shape='TH')
    else:
        mapping = aes(x='bin_size', y='rmse')

    # Process PC column if present
    if 'PC' in df.columns:
        # Convert non-string values into strings if needed
        df['PC'] = df['PC'].apply(lambda x: f"PCs:{x[0]} {x[1]}" if not isinstance(x, str) else x)
        # Generate a discrete Viridis palette for unique PC groups
        unique_groups = sorted(df['PC'].unique())
        n_groups = len(unique_groups)
        viridis_cmap = plt.get_cmap('viridis', n_groups)
        #convert rgb colors to hex
        colors = [mcolors.rgb2hex(viridis_cmap(i)) for i in range(n_groups)]
        color_mapping = dict(zip(unique_groups, colors))

    # Process TH column if present.
    if 'TH' in df.columns:
        # Assume that the values in TH are valid shape codes.
        df['TH'] = df['TH'].astype(str)
        df['TH'] = df['TH'].apply(lambda x: f"-{x}")
        shape_mapping = {'-1': 'o',  # circle
                         '-2': 's',  # square
                         '-3': '^'}

    # Build the base plot
    p = (ggplot(df, mapping)
         + geom_point(size=2)
         + geom_smooth(aes(group=1), method='lm', se=False, color='red', size=1)
         + labs(
                title='RMSE vs bin size',
                x='Bin size[s]',
                y='RMSE',
                shape='Threshold Factor',
                color="Principal Component"
           )
         + theme_minimal()
         + annotation
    )

    # Add manual scales if the corresponding columns exist
    if 'PC' in df.columns:
        p = p + scale_color_manual(values=color_mapping)
    if 'TH' in df.columns:
        p = p + scale_shape_manual(values=shape_mapping)

    return p

def ggplot_rmse(rmse_dic: dict, ref_obj: ArrayAnalysis):
    """
    Generates a scatter plot of RMSE values for each sample with horizontal lines indicating control thresholds.

    The function first computes specific control percentiles (5th and 1st) for a reference map by generating a
    distribution of RMSE values from randomized permutations (using `generate_control_rmse_distr`). These lines serve
    as baselines to compare individual sample RMSE values.

    Parameters:
        ref_obj: object
            ArrayAnalysis object that should contain a spontaneous map used to compute
            control percentile thresholds.
        rmse_dic (dict):
            A dictionary mapping sample identifiers to their corresponding RMSE values.

    Returns:
        plotnine.ggplot:
            A ggplot object visualizing the sample-wise RMSE values with control threshold lines.
    """
    # Calculate control percentiles
    percentile_5 = helper.generate_control_rmse_distr(ref_obj.spontaneous_map, percentile=5)
    percentile_1 = helper.generate_control_rmse_distr(ref_obj.spontaneous_map, percentile=1)
    num_samples = len(rmse_dic)

    # Build a DataFrame for sample points
    df_points = pd.DataFrame({
        'sample': list(rmse_dic.keys()),
        'rmse': list(rmse_dic.values())
    })

    # Convert sample to int if it’s numeric
    df_points['sample'] = df_points['sample'].astype(int, errors='ignore')
    # Sort by sample so the ordering is consistent
    df_points.sort_values(by='sample', inplace=True)
    df_points['x'] = range(1, num_samples + 1)

    # DataFrame for the horizontal control lines
    df_lines = pd.DataFrame({
        'Percentiles of the RMSE of the control maps': ["5th", "1st"],
        'yintercept': [percentile_5, percentile_1]
    })

    # Compute y-axis limits based on all RMSE values
    all_values = list(rmse_dic.values()) + [percentile_5, percentile_1]
    y_min, y_max = min(all_values), max(all_values)
    margin = 0.1 * (y_max - y_min)
    y_limits = (y_min - margin, y_max + margin)

    # Build the plot
    p = (
        ggplot()
        + geom_hline(
            data=df_lines,
            mapping=aes(
                yintercept='yintercept',
                color='Percentiles of the RMSE of the control maps'
            ),
            size=1
        )
        + geom_point(
            data=df_points,
            mapping=aes(x='x', y='rmse'),
            color='blue',
            size=3
        )
        + labs(
            title=f"",
            x="Sample",
            y="RMSE [rad]"
        )
        + scale_y_continuous(limits=y_limits)
        + scale_x_continuous(
        breaks=df_points['x'],
        labels=df_points['sample'],
        limits=(0.5, num_samples + 0.5),
        expand=(0.05, 0.05),
        minor_breaks=None
    )
        + theme_minimal()
    )

    return p

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

def ggplot_mean_rmse_by_time(df: pd.DataFrame):
    """
    Plots the mean RMSE for each time_diff value.

    Parameters:
        df (pd.DataFrame): DataFrame with columns "sample", "time_diff", and "RMSE".

    Returns:
        plotnine.ggplot: A plot showing mean RMSE versus time_diff.
    """
    # Group by time_diff and compute the mean RMSE
    df_mean = df.groupby("time_diff", as_index=False)["RMSE"].mean()

    # Create the plot with plotnine
    p = (
        ggplot(df_mean, aes(x='time_diff', y='RMSE')) +
        geom_line(group=1) +  # Connects the points with a line
        geom_point(size=3) +  # Adds points for each mean RMSE
        labs(
            title="Mean RMSE vs. Time Difference",
            x="Time Difference",
            y="Mean RMSE"
        )
        + theme(
        panel_background=element_rect(fill='white'),
        plot_background=element_rect(fill='white'),
        figure_size=(30, 6),
        legend_title=element_text(rotation=90, va="baseline"),  # x=200,y=-400
        axis_text_x=element_text(angle=90, vjust=0.5, hjust=1, )
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

def ggplot_rmse_box(df: pd.DataFrame):
    """
    Creates a box plot of RMSE for each time_diff.

    Parameters:
        df (pd.DataFrame): DataFrame with columns "sample", "time_diff", and "RMSE".

    Returns:
        plotnine.ggplot: A ggplot object representing the box plot.
    """
    p = (ggplot(df, aes(x="factor(time_diff)", y="RMSE"))
         + geom_boxplot()
         + labs(
                title="",
                x="Temporal distance [min]",
                y="RMSE [rad]"
            )
         + scale_y_continuous(limits=(0,1.5),breaks=[0.3, 0.6, 0.9, 1.2, 1.5])
         + theme(
                panel_background=element_rect(fill='white'),
                plot_background=element_rect(fill='white'),
                panel_border=element_rect(color='black', fill=None, size=1),
                figure_size=(30, 6),
                axis_text_x=element_text(size=8),
                axis_text_y=element_text(size=8),
                axis_title_x=element_text(size=15),
                axis_title_y=element_text(size=15),
                strip_text_x=element_text(size=15)

            )
         )

    return p