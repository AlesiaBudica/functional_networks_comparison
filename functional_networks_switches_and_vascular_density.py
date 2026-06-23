import numpy as np
import os
import nibabel as nb
import statsmodels.api as sm
import nigsp as ng
from nibabel.funcs import concat_images
import networkx as nx
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.table import default_txt_fmt

def load_subject_seeds(seed_file_path):
    """
    Reads a comma-separated file containing subject-specific seed coordinates.
    Returns a dictionary mapping subject ID to their network coordinates.
    """
    subject_seeds = {}
    with open(seed_file_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            parts = [p.strip() for p in line.split(',')]
            sub = parts[0]
            
            # Extract coordinates as integer integers
            ecn = [int(parts[1]), int(parts[2]), int(parts[3])]
            dna = [int(parts[4]), int(parts[5]), int(parts[6])]
            dnb = [int(parts[7]), int(parts[8]), int(parts[9])]
            
            subject_seeds[sub] = {
                'ecn': ecn,
                'dna': dna,
                'dnb': dnb
            }
    return subject_seeds

def read_and_concatenate_subject_sessions(data_dir, output_dir, subject_ids, session_ids):
    """
    Loops through subjects, reads their session NIfTI files, 
    concatenates them, saves the combined image, and returns the output paths.
    
    Parameters
    ----------
    data_dir : str
        The base directory containing the raw subject folders.
    output_dir : str
        The directory where the concatenated NIfTI files will be saved.
    subject_ids : list
        List of subject IDs.
    session_ids : list
        List of session IDs.

    Returns
    -------
    list
        A list of file paths pointing to the saved concatenated NIfTI images.
    """
    # Create the output directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize an empty array/list to collect the saved file paths
    saved_paths = []

    # 1. Loop over the 6 subjects
    for sub in subject_ids:
        session_files = []
        
        # 2. Loop over the 6 sessions for the current subject
        for ses in session_ids:
            file_name = f"sub-{sub}_ses-{ses}_bold.nii.gz"
            file_path = os.path.join(data_dir, f"sub-{sub}", f"ses-{ses}", file_name)
            
            if os.path.exists(file_path):
                print(f"Loading: {file_name}")
                nifti_img = nb.load(file_path)
                session_files.append(nifti_img)
            else:
                print(f"Warning: File missing at {file_path}")

        # 3. Concatenate and save files for this subject
        if session_files:
            print(f"--> Concatenating 6 sessions for subject: {sub}")
            concat_img = concat_images(session_files)
            
            # Define output filename and path
            output_name = f"sub-{sub}_desc-concatenated_bold.nii.gz"
            output_path = os.path.abspath(os.path.join(output_dir, output_name))
            
            # Save the new combined 4D image to disk
            nb.save(concat_img, output_path)
            print(f"Successfully saved: {output_name}\n")
            
            # Append the completed path to our tracking array
            saved_paths.append(output_path)
        else:
            print(f"Error: No sessions found for subject {sub}\n")

    # Return the completed array of file paths
    return saved_paths

# THE FOLLOWING CODE IS USED UNDER THE BSD 3-CLAUSE LICENSE
# Point process analysis for a signal. Values equal to 1 when the original value 
# is higher than the threshold (1.5*SD)
def point_process(signal):

    pp_signal = np.zeros_like(signal)
    th = np.std(signal) * 1.5

    pp_signal[signal > th] = 1

    return pp_signal

# Given an fMRI, extract timeseries, calculate Point Process and then the Rate and Map for each voxel given a seed
def maps_and_rates(in_file, seed_location_ecn, seed_location_dna, seed_location_dnb, outdir, sub, map_ecn_file_name = 'map_ecn', map_dna_file_name = 'map_dna', map_dnb_file_name = 'map_dnb'):

    # Treat fMRI image
    data, mask, img = ng.load_nifti_get_mask(in_file)

    # Extract seed and apply PP for the 3 networks
    seed_data_ecn = data[seed_location_ecn[0], seed_location_ecn[1], seed_location_ecn[2],:]
    pp_seed_data_ecn = point_process(seed_data_ecn)

    seed_data_dna = data[seed_location_dna[0], seed_location_dna[1], seed_location_dna[2],:]
    pp_seed_data_dna = point_process(seed_data_dna)

    seed_data_dnb = data[seed_location_dnb[0], seed_location_dnb[1], seed_location_dnb[2],:]
    pp_seed_data_dnb = point_process(seed_data_dnb)
   
    # Map and switch rates
    # Save map_ecn
    map_ecn = data[..., pp_seed_data_ecn != 0].mean(axis=-1)
    map_ecn_file_path = f'{outdir}/{map_ecn_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_ecn, img, map_ecn_file_path)
    print(f"Success! Map ECN saved to: {map_ecn_file_path}")

    # Save map_dna
    map_dna = data[..., pp_seed_data_dna != 0].mean(axis=-1)
    map_dna_file_path = f'{outdir}/{map_dna_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_dna, img, map_dna_file_path)
    print(f"Success! Map DNA saved to: {map_dna_file_path}")

    # Save map_dnb
    map_dnb = data[..., pp_seed_data_dnb != 0].mean(axis=-1)
    map_dnb_file_path = f'{outdir}/{map_dnb_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_dnb, img, map_dnb_file_path)
    print(f"Success! Map DNB saved to: {map_dnb_file_path}")

    rate_ecn_dna = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dna[1:]) == 2)
    rate_ecn_dnb = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dnb[1:]) == 2)
    
    return rate_ecn_dna, rate_ecn_dnb, seed_data_ecn, seed_data_dna, seed_data_dnb, pp_seed_data_ecn, pp_seed_data_dna, pp_seed_data_dnb

def plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices, sub_id):
    """
    Generates a 3-panel synchronized time-series plot with trigger arrows, 
    significance markers (*), and vertical alignment guidelines.
    """
    # Create a 3-row synchronized figure sharing the exact same X-axis
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=150)
    
    # Define styling profiles
    line_props = dict(marker='o', markersize=3, markerfacecolor='white', linewidth=1, alpha=0.7)
    dash_style = dict(color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # PANEL 1: ECN Time-Series (Top Channel)
    threshold = np.std(seed_data_ecn) * 1.5    
    axes[0].plot(time, seed_data_ecn, color='royalblue', label='ECN', **line_props)
    axes[0].set_ylabel('ECN', fontsize=12, fontweight='bold')
    axes[0].axhline(threshold, color='black', linestyle='--')
    axes[0].text(time[-1] * 1.01, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    axes[0].set_title(f"Subject {sub_id} Trigger Profiles", fontsize=14, fontweight="bold", pad=12)
    
    # Overlay trigger arrows at specific time points
    for t_idx in ecn_indices:
        axes[0].annotate('', xy=(time[t_idx], 2.2), xytext=(time[t_idx], 4.0),
                         arrowprops=dict(facecolor='black', arrowstyle='->', lw=2))
        
    # PANEL 2: DNA Time-Series (Middle Channel)
    threshold = np.std(seed_data_dna) * 1.5
    axes[1].plot(time, seed_data_dna, color='crimson', label='DNA', **line_props)
    axes[1].set_ylabel('DNA', fontsize=12, fontweight='bold')
    axes[1].axhline(threshold, color='black', linestyle='--')
    axes[1].text(time[-1] * 1.01, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    
    # Overlay asterisks for DNA
    for t_idx in dna_indices:
        axes[1].text(time[t_idx], 1.5, '*', fontsize=18, fontweight='bold', ha='center')

    # PANEL 3: DNB Time-Series (Bottom Channel)
    threshold = np.std(seed_data_dnb) * 1.5
    axes[2].plot(time, seed_data_dnb, color='lightcoral', label='DNB', **line_props)
    axes[2].set_ylabel('DNB', fontsize=12, fontweight='bold')
    axes[2].axhline(threshold, color='black', linestyle='--')
    axes[2].text(time[-1] * 1.01, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    
    # Overlay asterisks for DNB
    for t_idx in dnb_indices:
        axes[2].text(time[t_idx], 1.5, '*', fontsize=18, fontweight='bold', ha='center')
        

    # GLOBAL FORMATTING & VERTICAL ALIGNMENT LINES
    # Draw vertical dashed alignment markers across all subplots at trigger events
    for ax in axes:
        ax.set_ylim(-4.5, 4.5)           # check maximum and minimum z-scores
        ax.set_yticks([-4, -2, 0, 2, 4]) # check maximum and minimum z-scores 
        ax.grid(False)

        # Draw Royal Blue vertical lines for ECN triggers
        for t_idx in ecn_indices:
            ax.axvline(x=time[t_idx], color='royalblue', linestyle='--', linewidth=1, alpha=0.4)
            
        # Draw Crimson vertical lines for DNA triggers
        for t_idx in dna_indices:
            ax.axvline(x=time[t_idx], color='crimson', linestyle='--', linewidth=1, alpha=0.4)
            
        # Draw Light Coral vertical lines for DNB triggers
        for t_idx in dnb_indices:
            ax.axvline(x=time[t_idx], color='lightcoral', linestyle='--', linewidth=1, alpha=0.4)
    
    # Set shared X-axis parameters
    axes[2].set_xlabel('Time (sec)', fontsize=12)
    axes[2].set_xlim(0, time[-1]) # multiply the total number of volumes in the concatenated file by the scan's TR for the maximum limit
    
    # Add a unified Y axis label text block on the left
    fig.text(0.02, 0.5, 'BOLD (z)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0.04, 0, 0.95, 1])
    plot_name = f"Trigger_profile_sub-{sub_id}.png"
    plt.savefig(plot_name, dpi=150)
    plt.show()

def density (brain_mask_path, vessel_mask_path):
    _, brain_mask, _ = ng.load_nifti_get_mask(brain_mask_path, is_mask=True, ndim=3)
    brain_mask_voxels = np.sum(brain_mask)

    _, vessel_mask, _ = ng.load_nifti_get_mask(vessel_mask_path, is_mask=True, ndim=3)
    vessel_mask_voxels = np.sum(vessel_mask)

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

def plot_linear_regression(x, y, regression_results, label, title):
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
    title : str, optional
        The title for the generated plot.
    """
    # 1. Convert inputs to arrays to ensure they plot cleanly
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    y_pred = regression_results["predictions"]
    slope = regression_results["slope"]
    intercept = regression_results["intercept"]
    r_2 = regression_results["r_squared"]

    # 2. Initialize the plot figure
    plt.figure(figsize=(8, 5), dpi=100)

    # 3. Scatter plot for the actual raw data points
    plt.scatter(x_arr, y_arr, color="darkblue", alpha=0.7, edgecolors="k", s=80, label="Actual Data")

    # 4. Line plot for the predicted linear regression line
    # Sorting x makes sure the line draws smoothly from left to right
    sort_idx = np.argsort(x_arr)
    plt.plot(x_arr[sort_idx], y_pred[sort_idx], color="crimson", linewidth=2.5, 
             label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    # 5. Add text box in the upper left containing the R² score
    stats_text = f"$R^2$ = {r_2:.3f}\nRMSE = {regression_results['rmse']:.3f}"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    # 6. Labels, grid, and legend styling
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("X (Independent Variable)", fontsize=12)
    plt.ylabel("Y (Dependent Variable)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    
    # 7. Display the plot window
    plt.tight_layout()
    plot_name = f"Linear_Regression_{label}.png"
    plt.savefig(plot_name, dpi=150)
    plt.show()

def main ():
    
    # fMRI data concatenation and processing for functional networks and switching rates
    no_subjects = 6
    no_sessions = 6
    subjects = [f"{i:02d}" for i in range(1, no_subjects + 1)]
    sessions = [f"{i:02d}" for i in range(1, no_sessions + 1)]

    fmri_base_directory = "/fmri_data" # replace with real path
    fmri_output_directory = "/fmri_data/concatenated" # replace with real path
    
    seed_file = "seeds.txt"
    all_subject_seeds = load_subject_seeds(seed_file)

    concatenated_files = read_and_concatenate_subject_sessions(fmri_base_directory, fmri_output_directory, subjects, sessions)

    rate_ecn_dna_array = []
    rate_ecn_dnb_array = []
   
    for filename, sub in zip(concatenated_files, subjects):
        # Calculating rates
        sub_seeds = all_subject_seeds[sub]
        seed_ecn = sub_seeds['ecn']
        seed_dna = sub_seeds['dna']
        seed_dnb = sub_seeds['dnb']
        
        rate_ecn_dna, rate_ecn_dnb, seed_data_ecn, seed_data_dna, seed_data_dnb, pp_seed_data_ecn, pp_seed_data_dna, pp_seed_data_dnb = maps_and_rates(filename, seed_ecn, seed_dna, seed_dnb, fmri_output_directory, sub) 
        
        rate_ecn_dna_array.append(rate_ecn_dna)
        rate_ecn_dnb_array.append(rate_ecn_dnb)

        # Plotting the time-series with triggers  
        ecn_indices = np.where(pp_seed_data_ecn == 1)[0]
        dna_indices = np.where(pp_seed_data_dna == 1)[0]
        dnb_indices = np.where(pp_seed_data_dnb == 1)[0]

        time = np.arange(len(seed_data_ecn)) * 2.0 # Replace with real TR

        plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices, sub)

    #MRI data processing for vascular density
    vessel_segmentation_directory = "/Neurodata/M3PI/derivatives/vessels/segmentations/prediction/"
   
    density_array = []

    print("====================================")

    for sub in subjects:
        mri_base_directory = f"/Neurodata/M3PI/derivatives/vessels/sub-{sub}/ses-7T/anat/"
        file_name_brain = f"00.sub-{sub}_ses-7T_part-mag_T2starw_brain_mask.nii.gz"
        full_path_brain = os.path.abspath(os.path.join(mri_base_directory, file_name_brain))

        file_name_vessel = f"sub-{sub}_ses-7T_part-mag_T2starw_imgavg_preprocessed_bfcvb_brain.nii.gz"
        full_path_vessel = os.path.abspath(os.path.join(vessel_segmentation_directory, file_name_vessel))
        
        brain_mask_voxels, vessel_mask_voxels, vascular_density = density(full_path_brain, full_path_vessel)
        density_array.append(vascular_density)

        print(f"Subject {sub}")
        print(f"Total Brain Voxels : {brain_mask_voxels}")
        print(f"Total Vessel Voxels Subject : {vessel_mask_voxels}")
        print(f"Calculated Whole-Brain Vascular Density Subject : {vascular_density:.6f}")
        print("====================================")

    # Regression analysis between switching rates and vascular density
    metrics_ecn_dna = perform_linear_regression(density_array, rate_ecn_dna_array)
    print_regression_metrics(metrics_ecn_dna, label=f"ECN-DNA")
    
    metrics_ecn_dnb = perform_linear_regression(density_array, rate_ecn_dnb_array)
    print_regression_metrics(metrics_ecn_dnb, label=f"ECN-DNB")

    plot_linear_regression(density_array, rate_ecn_dna_array, metrics_ecn_dna, label=f"ECN-DNA", title="Linear Regression: Density vs ECN-DNA Switching Rate")
    plot_linear_regression(density_array, rate_ecn_dnb_array, metrics_ecn_dnb, label=f"ECN-DNB", title="Linear Regression: Density vs ECN-DNB Switching Rate")

    return 

if __name__ == "__main__":
    main()

# Copyright (c) 2014, Child Mind Institute, Inc. and C-PAC developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
