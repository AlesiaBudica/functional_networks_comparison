#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.iolib.table import SimpleTable

from nigsp import io as ng


def density(brain_mask_path, vessel_mask_path, wholebrain_mask=None):
    _, brain_mask, _ = ng.load_nifti_get_mask(brain_mask_path, is_mask=True, ndim=3)

    if wholebrain_mask is not None:
        _, wholebrain_mask, _ = ng.load_nifti_get_mask(wholebrain_mask, is_mask=True, ndim=3)
        brain_mask = brain_mask * wholebrain_mask

    brain_mask_voxels = np.sum(brain_mask)

    _, vessel_mask, _ = ng.load_nifti_get_mask(vessel_mask_path, is_mask=True, ndim=3)
    vessel_inside_brain = vessel_mask * brain_mask
    vessel_mask_voxels = np.sum(vessel_inside_brain)

    vascular_density = vessel_mask_voxels / brain_mask_voxels

    return brain_mask_voxels, vessel_mask_voxels, vascular_density


def perform_linear_regression(x, y):
    """
    Perform simple linear regression using ordinary least squares (OLS).

    Parameters
    ----------
    x : numpy.ndarray
        1D array of predictor (independent) variables.
    y : numpy.ndarray
        1D array of target (dependent) variables.

    Returns
    -------
    dict
        A dictionary containing regression coefficients and error metrics.
    """
    # Ensure inputs are flat 1D numpy arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    if len(x) != len(y):
        raise ValueError("The x and y arrays must have the same length.")

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate terms for slope (beta_1) using covariance and variance formulas
    covariance_xy = np.sum((x - x_mean) * (y - y_mean))
    variance_x = np.sum((x - x_mean) ** 2)

    if variance_x == 0:
        raise ValueError("Variance of x is zero. Cannot compute regression line.")

    # Calculate slope (m) and intercept (c) -> y = mx + c
    slope = covariance_xy / variance_x
    intercept = y_mean - (slope * x_mean)

    # Calculate predictions and residuals
    y_pred = (slope * x) + intercept
    residuals = y - y_pred

    # Calculate metrics (RMSE and R-squared)
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    ss_residual = np.sum(residuals ** 2)
    ss_total = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

    # Calculate p-value using scipy.stats
    _, _, _, p_value, _ = stats.linregress(x, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "rmse": rmse,
        "predictions": y_pred
    }


def print_regression_metrics(metrics_dict, label, output_file="regression_results.txt"):
    """
    Formats linear regression results into a professional ASCII table using statsmodels,
    prints it to the terminal, and saves/appends it to a text file.
    """
    # 1. Define rows and data columns for the table
    headers = ["Metric Parameter", f"Value ({label})"]
    table_data = [
        ["Line Equation", f"y = {metrics_dict['slope']:.4f}*x + {metrics_dict['intercept']:.4f}"],
        ["Slope (Beta 1)", f"{metrics_dict['slope']:.4f}"],
        ["Intercept (b)", f"{metrics_dict['intercept']:.4f}"],
        ["R² Accuracy", f"{metrics_dict['r_squared']:.4f} ({metrics_dict['r_squared']*100:.1f}%)"],
        ["p-value", f"{metrics_dict['p_value']:.4f}"],
        ["RMSE Error", f"{metrics_dict['rmse']:.4f}"]
    ]
    
    # 2. Generate the statsmodels SimpleTable
    # txt_fmt controls the visual borders of the table
    from statsmodels.iolib.table import default_txt_fmt
    sm_table = SimpleTable(table_data, headers=headers, title=f"REGRESSION SUMMARY: {label}", txt_fmt=default_txt_fmt)
    
    # 3. Convert table to a clean string format
    table_string = sm_table.as_text() + "\n\n"
    
    # 4. Print table to your terminal screen
    print(table_string)
    
    # 5. Append table to your output file
    with open(output_file, "a") as f:
        f.write(table_string)

    return metrics_dict


def plot_linear_regression(x, y, regression_results, label, x_axis_label, y_axis_label, fmri_output_directory):
    """
    Plots the original data points as a scatter plot and overlays 
    the calculated linear regression line.

    Parameters
    ----------
    x : array-like
        Original independent variables (e.g., sessions or time points).
    y : array-like
        Original dependent variables (e.g., parcel/network activity).
    regression_results : dict
        The output dictionary from the perform_linear_regression function.
    x_axis_label : str, optional
        The label for the x-axis.
    y_axis_label : str, optional
        The label for the y-axis.
    """
    # 1. Convert inputs to arrays to ensure they plot cleanly
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    y_pred = regression_results["predictions"]
    slope = regression_results["slope"]
    intercept = regression_results["intercept"]
    r_2 = regression_results["r_squared"]

    # 2. Initialize the plot figure
    plt.figure(figsize=(8, 5), dpi=300)

    # 3. Scatter plot for the actual raw data points
    plt.scatter(x_arr, y_arr, color="darkblue", alpha=0.7, edgecolors="k", s=80, label="Actual Data")

    # 4. Line plot for the predicted linear regression line
    # Sorting x makes sure the line draws smoothly from left to right
    sort_idx = np.argsort(x_arr)
    plt.plot(x_arr[sort_idx], y_pred[sort_idx], color="crimson", linewidth=2.5, 
             label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    # 6. Labels, grid, and legend styling
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
   
    if label == "DNB_ECN-DNA" or label == "ECN_ECN-DNA" or label == "Whole-Brain_ECN-DNA":
        
        # Add legend in the lower left corner
        plt.legend(loc="lower left", fontsize=11)
        
        # Add text box in the upper right containing the R² score
        stats_text = f"$R^2$ = {r_2:.3f}\nRMSE = {regression_results['rmse']:.3f}\np = {regression_results['p_value']:.4f}"
        plt.gca().text(0.975, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    elif label == "ECN_ECN-DNB":
        
        # Add legend in the upper left corner
        plt.legend(loc="upper left", fontsize=11)
        
        # Add text box in the upper right containing the R² score
        stats_text = f"$R^2$ = {r_2:.3f}\nRMSE = {regression_results['rmse']:.3f}\np = {regression_results['p_value']:.4f}"
        plt.gca().text(0.975, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    elif label == "DNA_ECN-DNA":
        
        # Add legend in the lower right corner
        plt.legend(loc="lower right", fontsize=11)
        
        # Add text box in the lower left containing the R² score
        stats_text = f"$R^2$ = {r_2:.3f}\nRMSE = {regression_results['rmse']:.3f}\np = {regression_results['p_value']:.4f}"
        plt.gca().text(0.025, 0.05, stats_text, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    else:
        
        # Add legend in the lower right corner
        plt.legend(loc="lower right", fontsize=11)
        
        # Add text box in the upper left containing the R² score
        stats_text = f"$R^2$ = {r_2:.3f}\nRMSE = {regression_results['rmse']:.3f}\np = {regression_results['p_value']:.4f}"
        plt.gca().text(0.025, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    # 7. Display the plot window
    plt.tight_layout()
    plot_name = os.path.join(fmri_output_directory, f"Linear_Regression_{label}.png")
    plt.savefig(plot_name, dpi=300)
    plt.show()


def plot_vascular_density_swarm(density_array, subjects, fmri_output_directory):
    """
    Creates a swarm-style scatter plot showing the vascular density of each brain region
    across all subjects using only Matplotlib.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Define distinct colors for each subject to track them across regions
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    regions = list(density_array.keys())  # ["Whole-Brain", "ECN", "DNA", "DNB"]

    # Loop through each brain region (X-axis positions: 0, 1, 2, 3)
    for x_idx, region in enumerate(regions):
        densities = density_array[region]
        
        # Add a tiny horizontal jitter to prevent overlapping points (Swarm effect)
        # np.linspace or a small deterministic spread keeps it clean
        jitter = np.linspace(-0.15, 0.15, len(subjects))
        
        for sub_idx, sub in enumerate(subjects):
            x_pos = x_idx + jitter[sub_idx]
            y_pos = densities[sub_idx]
            
            # Only add the label for the first region so the legend doesn't duplicate
            label = f"Sub-{sub}" if x_idx == 0 else ""
            
            plt.scatter(x_pos, y_pos, color=colors[sub_idx], s=120, edgecolors="k", 
                        alpha=0.85, zorder=3, label=label)

    # Style the plot
    plt.xticks(range(len(regions)), regions, fontsize=12, fontweight="normal")
    plt.ylabel("Vascular Density", fontsize=12, fontweight="normal")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    
    # Place legend outside or neatly inside
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    
    # Save the figure
    plot_name = os.path.join(fmri_output_directory, "Vascular_Density_Swarm_Plot.png")
    plt.savefig(plot_name, dpi=300)
    plt.close()
    print(f"Swarm plot successfully saved to: {plot_name}\n")


def main():

    fmri_output_directory = "/data/func_net_comp"

    subjects = ['01', '02', '03', '04', '05', '06']

    rate_ecn_dna_array = np.genfromtxt(os.path.join(fmri_output_directory, "rate_ecn_dna"))
    rate_ecn_dnb_array = np.genfromtxt(os.path.join(fmri_output_directory, "rate_ecn_dnb"))

    # MRI data processing for vascular density
    vessel_segmentation_directory = "/data/vessels/manualsegready/"
   
    density_array = {"Whole-Brain": [], "ECN": [], "DNA": [], "DNB": []}

    print("====================================")

    for sub in subjects:
        full_path_brain = os.path.abspath(f"/data/vessels/sub-{sub}/ses-7T/anat/00.sub-{sub}_ses-7T_part-mag_T2starw_brain_mask.nii.gz")

        file_name_vessel = f"00.sub-{sub}_ses-7T_part-mag_T2starw_imgavg_preprocessed_vessels.nii.gz"
        full_path_vessel = os.path.abspath(os.path.join(vessel_segmentation_directory, file_name_vessel))
        
        brain_mask_voxels, vessel_mask_voxels, vascular_density = density(full_path_brain, full_path_vessel)
        density_array["Whole-Brain"].append(vascular_density)

        print(f"Subject {sub}")
        print(f"Whole-Brain Total Voxels : {brain_mask_voxels}")
        print(f"Whole-Brain Vessel Voxels : {vessel_mask_voxels}")
        print(f"Calculated Whole-Brain Vascular Density : {vascular_density:.6f}")
        print("====================================")

        for k, x in [("ECN", "ecn"), ("DNA", "dna"), ("DNB", "dnb")]:
            file_name_network = f'map_{x}_mask_sub-{sub}_vesselres.nii.gz'
            full_path_network = os.path.abspath(os.path.join(fmri_output_directory, file_name_network))

            brain_mask_voxels, vessel_mask_voxels, vascular_density = density(full_path_network, full_path_vessel, full_path_brain)
            density_array[k].append(vascular_density)

            print(f"Subject {sub}")
            print(f"{k} Total Voxels : {brain_mask_voxels}")
            print(f"{k} Vessel Voxels : {vessel_mask_voxels}")
            print(f"Calculated {k} Vascular Density : {vascular_density:.6f}")
            print("====================================")

    # Generate the swarm plot comparing all regions across subjects
    plot_vascular_density_swarm(density_array, subjects, fmri_output_directory)

    # Regression analysis between switching rates and vascular density
    for k in density_array.keys():
        metrics_ecn_dna = perform_linear_regression(density_array[k], rate_ecn_dna_array)
        metrics_ecn_dnb = perform_linear_regression(density_array[k], rate_ecn_dnb_array)
        print_regression_metrics(
            metrics_ecn_dna,
            label=f"{k} Vessel Density vs ECN-DNA",
            output_file=os.path.abspath(os.path.join(fmri_output_directory, f"{k}_ecn-dna_regression.txt")),
        )
        print_regression_metrics(
            metrics_ecn_dnb,
            label=f"{k} Vessel Density vs ECN-DNB",
            output_file=os.path.abspath(os.path.join(fmri_output_directory, f"{k}_ecn-dnb_regression.txt")),
        )

        plot_linear_regression(
            density_array[k],
            rate_ecn_dna_array,
            metrics_ecn_dna,
            label=f"{k}_ECN-DNA",
            x_axis_label=f"{k} Vessel Density",
            y_axis_label="ECN-DNA Switches",
            fmri_output_directory=fmri_output_directory
        )
        plot_linear_regression(
            density_array[k],
            rate_ecn_dnb_array,
            metrics_ecn_dnb,
            label=f"{k}_ECN-DNB",
            x_axis_label=f"{k} Vessel Density",
            y_axis_label="ECN-DNB Switches",
            fmri_output_directory=fmri_output_directory
        )
    
    return 


if __name__ == "__main__":
    main()

"""
Copyright [2026] [Alesia Maria Budica]
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""