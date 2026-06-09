import numpy as np
import os
import nibabel as nb
import statsmodels.api as sm
import nigsp as ng
from nibabel.funcs import concat_images
import pyvane as pv
import networkx as nx
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def main ():
    
    # fMRI data processing for functional networks and switching rates
    no_subjects = 6
    no_sessions = 6
    subjects = [f"{i:02d}" for i in range(1, no_subjects + 1)]
    sessions = [str(i) for i in range(1, no_sessions + 1)]

    fmri_base_directory = "/fmri_data" # replace with real path
    fmri_output_directory = "/fmri_data/concatenated" # replace with real path

    concatenated_files = read_and_concatenate_subject_sessions(fmri_base_directory, fmri_output_directory, subjects, sessions)

    rate_ecn_dna_array = []
    rate_ecn_dnb_array = []
   
    for filename in concatenated_files:
        rate_ecn_dna, rate_ecn_dnb = maps_and_rates(filename, [0,0,0], [0,0,0], [0,0,0]) # replace seed_location_ecn, seed_location_dna, seed_location_dnb with real values
        rate_ecn_dna_array.append(rate_ecn_dna)
        rate_ecn_dnb_array.append(rate_ecn_dnb)

    # Plotting the time-series with triggers for subject 1
    example_file = concatenated_files[0]
    
    _, _, seed_data_ecn, seed_data_dna, seed_data_dnb, pp_seed_data_ecn, pp_seed_data_dna, pp_seed_data_dnb = maps_and_rates(example_file, [0,0,0], [0,0,0], [0,0,0]) # replace seed_location_ecn, seed_location_dna, seed_location_dnb with real values
    
    ecn_indices = np.where(pp_seed_data_ecn == 1)[0]
    dna_indices = np.where(pp_seed_data_dna == 1)[0]
    dnb_indices = np.where(pp_seed_data_dnb == 1)[0]

    time = np.arange(len(seed_data_ecn)) * 2.0 # Replace with real TR

    plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices) # replace with real data

    #MRI data processing for vascular density
    mri_base_directory = "/mri_data" # replace with real path

    density_array = []

    for sub in subjects:
        # Construct the target standardized filename
        file_name = f"sub-{sub}_mri.nii.gz"
        
        # Combine the base folder path with the filename
        full_path = os.path.abspath(os.path.join(mri_base_directory, file_name))
        
        data, mask, img = ng.load_nifti_get_mask(full_path)
        
        # Not sure how to use these
        data_masked = ng.apply_mask(data, mask)
        parcels = ng.apply_atlas(data, atlas)
        
        # Needs a graph?
        density = pv.vessel_density(graph, img.shape)
        density_array.append(density)

    # Regression analysis between switching rates and vascular density
    metrics_ecn_dna = perform_linear_regression(density_array, rate_ecn_dna_array)
    print_regression_metrics(metrics_ecn_dna, label=f"ECN-DNA")
    
    metrics_ecn_dnb = perform_linear_regression(density_array, rate_ecn_dnb_array)
    print_regression_metrics(metrics_ecn_dnb, label=f"ECN-DNB")

    plot_linear_regression(density_array, rate_ecn_dna_array, metrics_ecn_dna, title="Linear Regression: Density vs ECN-DNA Switching Rate")
    plot_linear_regression(density_array, rate_ecn_dnb_array, metrics_ecn_dnb, title="Linear Regression: Density vs ECN-DNB Switching Rate")

    return 

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

# Point process analysis for a signal. Values equal to 1 when the original value 
# is higher than the threshold (1.5*SD)
def point_process(signal):

    pp_signal = np.zeros(signal.shape[0])
    th = np.std(signal) * 1.5

    pp_signal[signal > th] = 1

    return pp_signal

# Given an fMRI, extract timeseries, calculate Point Process and then the Rate and Map for each voxel given a seed
def maps_and_rates(in_file, seed_location_ecn, seed_location_dna, seed_location_dnb, map_ecn_file_name = 'map_ecn', map_dna_file_name = 'map_dna', map_dnb_file_name = 'map_dnb'):

    # Treat fMRI image
    data, mask, img = ng.load_nifti_get_mask(in_file)

    # Extract seed and apply pp for the 3 networks
    seed_data_ecn = data[seed_location_ecn[0], seed_location_ecn[1], seed_location_ecn[2],:]
    pp_seed_data_ecn = point_process(seed_data_ecn)

    seed_data_dna = data[seed_location_dna[0], seed_location_dna[1], seed_location_dna[2],:]
    pp_seed_data_dna = point_process(seed_data_dna)

    seed_data_dnb = data[seed_location_dnb[0], seed_location_dnb[1], seed_location_dnb[2],:]
    pp_seed_data_dnb = point_process(seed_data_dnb)
   
    # Map and switch rates
    map_ecn = data[..., pp_seed_data_ecn != 0].mean(axis=-1)
    # Save map_ecn
    ng.export_nifti(map_ecn, img, map_ecn_file_name)
    print(f"Success! Map ecn saved to: {map_ecn_file_name}")

    map_dna = data[..., pp_seed_data_dna != 0].mean(axis=-1)
    # Save map_dna
    ng.export_nifti(map_dna, img, map_dna_file_name)
    print(f"Success! Map dna saved to: {map_dna_file_name}")

    map_dnb = data[..., pp_seed_data_dnb != 0].mean(axis=-1)
    # Save map_dnb
    ng.export_nifti(map_dnb, img, map_dnb_file_name)
    print(f"Success! Map dnb saved to: {map_dnb_file_name}")

    rate_ecn_dna = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dna[1:]) == 2)
    rate_ecn_dnb = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dnb[1:]) == 2)
    
    return rate_ecn_dna, rate_ecn_dnb, seed_data_ecn, seed_data_dna, seed_data_dnb, pp_seed_data_ecn, pp_seed_data_dna, pp_seed_data_dnb

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

def print_regression_metrics(metrics_dict, label="Dataset"):
    """
    Prints a cleanly formatted summary of the regression metrics dictionary.
    """
    print(f"\n====================================")
    print(f" REGRESSION SUMMARY: {label}")
    print(f"====================================")
    print(f" Line Equation:  y = {metrics_dict['slope']:.4f} * x + {metrics_dict['intercept']:.4f}")
    print(f" Slope (Beta 1): {metrics_dict['slope']:.4f}")
    print(f" Intercept (b):  {metrics_dict['intercept']:.4f}")
    print(f" R² Accuracy:    {metrics_dict['r_squared']:.4f} ({metrics_dict['r_squared']*100:.1f}%)")
    print(f" RMSE Error:     {metrics_dict['rmse']:.4f}")
    print(f"====================================\n")

def plot_linear_regression(x, y, regression_results, title="Linear Regression Plot"):
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
    plt.show()

def plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices):
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
    axes[0].text(605, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    
    # Overlay trigger arrows at specific time points
    for t_idx in ecn_indices:
        axes[0].annotate('', xy=(time[t_idx], 2.2), xytext=(time[t_idx], 4.0),
                         arrowprops=dict(facecolor='black', arrowstyle='->', lw=2))
        
    # PANEL 2: DNA Time-Series (Middle Channel)
    threshold = np.std(seed_data_dna) * 1.5
    axes[1].plot(time, seed_data_dna, color='crimson', label='DNA', **line_props)
    axes[1].set_ylabel('DNA', fontsize=12, fontweight='bold')
    axes[1].axhline(threshold, color='black', linestyle='--')
    axes[1].text(605, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    
    # Overlay asterisks for DNA
    for t_idx in dna_indices:
        axes[1].text(time[t_idx], 1.5, '*', fontsize=18, fontweight='bold', ha='center')

    # PANEL 3: DNB Time-Series (Bottom Channel)
    threshold = np.std(seed_data_dnb) * 1.5
    axes[2].plot(time, seed_data_dnb, color='lightcoral', label='DNB', **line_props)
    axes[2].set_ylabel('DNB', fontsize=12, fontweight='bold')
    axes[2].axhline(threshold, color='black', linestyle='--')
    axes[2].text(605, threshold, '1.5 SD Cutoff', verticalalignment='center', fontsize=11)
    
    # Overlay asterisks for DNB
    for t_idx in dnb_indices:
        axes[2].text(time[t_idx], 1.5, '*', fontsize=18, fontweight='bold', ha='center')
        

    # GLOBAL FORMATTING & VERTICAL ALIGNMENT LINES
    # Draw vertical dashed alignment markers across all subplots at trigger events
    all_events = sorted(list(set(ecn_indices + dna_indices + dnb_indices)))
    for ax in axes:
        ax.set_ylim(-4.5, 4.5)           # check maximum and minimum z-scores
        ax.set_yticks([-4, -2, 0, 2, 4]) # check maximum and minimum z-scores 
        ax.grid(False)
        # Apply the vertical line spans across panels
        for t_idx in all_events:
            ax.axvline(x=time[t_idx], **dash_style)
            
    # Set shared X-axis parameters
    axes[2].set_xlabel('Time (sec)', fontsize=12)
    axes[2].set_xlim(0, 0) # multiply the total number of volumes in the concatenated file by the scan's TR for the maximum limit
    
    # Add a unified Y axis label text block on the left
    fig.text(0.02, 0.5, 'BOLD (z)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0.04, 0, 0.95, 1])
    plt.show()

if __name__ == "__main__":
    main()