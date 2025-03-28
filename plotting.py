"""
Plotting
This script contains functions for
Aligning spontaneous map to a reference map
    -find_ideal_rotation()
Calculating circular difference between two spontaneous maps
    -circ_diff()
Calculating RMSE of two spontaneous maps
    -rmse_angles()
Generating control distribution of RMSE and calculating percentile
    -generate_rmse_distr()
Plotting graph of how RMSE depends on bin size with specific threshold factor and PCs
    -ggplot_rmse_vs_bin_size()
Plotting RMSE for different samples
    -ggplot_rmse()

Author: Radoslav Jochman
"""
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from array_analysis import *
from scipy.stats import linregress
import numpy as np
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme_minimal, scale_color_manual, annotate, scale_shape_manual, geom_hline, scale_y_continuous, scale_x_continuous

def find_ideal_rotation(ref: np.ndarray, rot: np.ndarray, n_steps: int=1000):
    """
    Finds the rotation offset (from 0 to π) that best aligns a given spontaneous map to a reference map.

    This function evaluates equally spaced rotation offsets (steps) within [0, π] and applies each offset
    to the map `rot`. It then computes the circular difference between the offset map and the reference map
    `ref` (using mean squared error as a criterion), selecting the offset that yields the smallest MSE.

    Parameters:
        ref (numpy.ndarray):
            The reference spontaneous map (2D array).
        rot (numpy.ndarray):
            The map to be aligned (2D array with same shape as 'ref').
        n_steps (int, optional):
            Number of rotation increments in the search space, evenly distributed from 0 to π.
            A larger value produces finer precision but increases computational cost.

    Returns:
        numpy.ndarray:
            The best-aligned map.
    """

    steps = np.linspace(0, np.pi, n_steps)
    best_err = np.inf
    best_rotated = None

    for step in steps:
        candidate = np.mod(rot + step, np.pi)
        diff = circ_diff(ref, candidate)
        err = np.mean(diff ** 2)
        if err < best_err:
            best_err = err
            best_rotated = candidate
    #mark the corners as no valid
    best_rotated[0, 0] = -2
    best_rotated[0, 9] = -2
    best_rotated[9, 0] = -2
    best_rotated[9, 9] = -2
    return best_rotated

def circ_diff(a, b):
    """
    Computes the circular difference (with a period of π) between two spontaneous maps.

    This function calculates the absolute difference between two angles (or angle arrays),
    then adjusts it to ensure the difference is within the range [0, π/2], effectively
    treating π as a full period.

    Parameters:
        a: float or array-like
            The first angle or collection of angles in radians. Typically in [0, π).
        b: float or array-like
            The second angle or collection of angles in radians. Typically in [0, π).

    Returns:
        float or numpy.ndarray:
            The circular difference between the inputs, within the range [0, π/2].
            If inputs are arrays, the output is an array of the same shape.
    """
    diff = np.abs(a - b)
    return np.minimum(diff, np.pi - diff)

def rmse_angles(a, b):
    """
    Calculates the root mean square error (RMSE) for two angle arrays, accounting for circular difference.

    This function uses `circ_diff` to compute the circular difference (period of π) between corresponding
    angles in `a` and `b`, then calculates the square root of the mean of the squared differences.

    Parameters:
        a (array-like):
            The first set of angles in radians.
        b (array-like):
            The second set of angles in radians, typically matching `a`.

    Returns:
        float:
            The RMSE value representing the circular difference between `a` and `b`.
    """
    differences = circ_diff(a, b)
    mse = np.mean(differences**2)
    return np.sqrt(mse)

def generate_rmse_distr(ref_map: np.ndarray, n_iter: int=2000, percentile: int=1):
    """
    Creates a distribution of RMSE values by randomly permuting a reference map and aligning each permutation
    back to the original.

    The function performs the following steps for each iteration:
      1. Flattens `ref_map` and randomly permutes its elements, then reshapes it to the original shape.
      2. Aligns the permuted map to the reference map using `find_ideal_rotation`.
      3. Computes the RMSE between the reference map and the aligned permutation using `rmse_angles`.
      4. Collects these RMSE values into a list.

    After all iterations, the function returns the specified percentile value of the resulting RMSE distribution.
    This process can be used to estimate a baseline or threshold for statistical analysis when comparing map
    alignments.

    Parameters:
        ref_map (numpy.ndarray):
            The reference spontaneous map, typically a 2D array.
        n_iter (int, optional):
            Number of random permutations to generate. Default is 2000.
        percentile (float, optional):
            Which percentile of the RMSE distribution to return. Default is 1 (1st percentile).

    Returns:
        float:
            The selected percentile from the distribution of RMSE values obtained from random permutations.
    """
    rmse_dist = []
    for i in range(n_iter):
        permut_map = np.random.permutation(ref_map.flatten()).reshape(ref_map.shape)
        permut_map = find_ideal_rotation(ref_map,permut_map)
        rmse = rmse_angles(ref_map,permut_map)
        rmse_dist.append(rmse)
    return np.percentile(np.array(rmse_dist), percentile)

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
    distribution of RMSE values from randomized permutations (using `generate_rmse_distr`). These lines serve
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
    percentile_5 = generate_rmse_distr(ref_obj.spontaneous_map, percentile=5)
    percentile_1 = generate_rmse_distr(ref_obj.spontaneous_map, percentile=1)
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