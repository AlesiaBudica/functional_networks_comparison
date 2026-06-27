#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np

from nigsp import io as ng


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


# Point process analysis for a signal. Values equal to 1 when the original value 
# is higher than the threshold (1.2*SD)
def point_process(signal):
    """
    The following code is taken from https://erramuzpe.github.io/C-PAC/blog/2015/08/07/integration-of-measures-and-point-process-developing/ and copyrighted to Asier Erramuzpe.
    """
    pp_signal = np.zeros_like(signal)
    th = 1.2  # It's normalised!

    pp_signal[signal > th] = 1

    return pp_signal


# Given an fMRI, extract timeseries, calculate Point Process and then the Rate and Map for each voxel given a seed
def maps_and_rates(in_file, seed_location_ecn, seed_location_dna, seed_location_dnb, outdir, sub, map_ecn_file_name = 'map_ecn', map_dna_file_name = 'map_dna', map_dnb_file_name = 'map_dnb'):
    """
    The following code is taken from https://erramuzpe.github.io/C-PAC/blog/2015/08/07/integration-of-measures-and-point-process-developing/ and copyrighted to Asier Erramuzpe.
    """
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
    map_ecn_mask = np.zeros_like(map_ecn)
    map_ecn_mask[map_ecn > .5] = 1
    map_ecn_file_path = f'{outdir}/{map_ecn_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_ecn, img, map_ecn_file_path)
    map_ecn_mask_file_path = f'{outdir}/{map_ecn_file_name}_mask_sub-{sub}.nii.gz'
    ng.export_nifti(map_ecn_mask, img, map_ecn_mask_file_path)
    print(f"Success! Map ECN saved to: {map_ecn_file_path}")

    # Save map_dna
    map_dna = data[..., pp_seed_data_dna != 0].mean(axis=-1)
    map_dna_mask = np.zeros_like(map_dna)
    map_dna_mask[map_dna > .5] = 1
    map_dna_file_path = f'{outdir}/{map_dna_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_dna, img, map_dna_file_path)
    map_dna_mask_file_path = f'{outdir}/{map_dna_file_name}_mask_sub-{sub}.nii.gz'
    ng.export_nifti(map_dna_mask, img, map_dna_mask_file_path)
    print(f"Success! Map DNA saved to: {map_dna_file_path}")

    # Save map_dnb
    map_dnb = data[..., pp_seed_data_dnb != 0].mean(axis=-1)
    map_dnb_mask = np.zeros_like(map_dnb)
    map_dnb_mask[map_dnb > .5] = 1
    map_dnb_file_path = f'{outdir}/{map_dnb_file_name}_sub-{sub}.nii.gz'
    ng.export_nifti(map_dnb, img, map_dnb_file_path)
    map_dnb_mask_file_path = f'{outdir}/{map_dnb_file_name}_mask_sub-{sub}.nii.gz'
    ng.export_nifti(map_dnb_mask, img, map_dnb_mask_file_path)
    print(f"Success! Map DNB saved to: {map_dnb_file_path}")

    rate_ecn_dna = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dna[1:]) == 2)
    rate_ecn_dnb = np.count_nonzero((pp_seed_data_ecn[:-1] + pp_seed_data_dnb[1:]) == 2)
    
    return rate_ecn_dna, rate_ecn_dnb, seed_data_ecn, seed_data_dna, seed_data_dnb, pp_seed_data_ecn, pp_seed_data_dna, pp_seed_data_dnb


def plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices, fmri_output_directory, sub_id):
    """
    Generates a 3-panel synchronized time-series plot with trigger arrows, 
    significance markers (*), and vertical alignment guidelines.
    """
    # Create a 3-row synchronized figure sharing the exact same X-axis
    fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True, dpi=300)
    
    # Define styling profiles
    line_props = dict(linestyle='-', linewidth=1, alpha=1)
    dash_style = dict(color='black', linestyle='--', linewidth=1, alpha=0.5)

    # PANEL 1: ECN Time-Series (Top Channel)
    threshold = 1.2
    axes[0].plot(time, seed_data_ecn, color='royalblue', label='ECN', **line_props)
    axes[0].set_ylabel('ECN', fontsize=12)
    axes[0].axhline(threshold, **dash_style)
    axes[0].set_title(f"Subject {sub_id} Trigger Profiles", fontsize=14, pad=12)
    
    # Overlay trigger arrows at specific time points
    for t_idx in ecn_indices:
        axes[0].text(time[t_idx], 1.2, '*', fontsize=9, ha='center')
        
    # PANEL 2: DNA Time-Series (Middle Channel)
    threshold = 1.2
    axes[1].plot(time, seed_data_dna, color='crimson', label='DNA', **line_props)
    axes[1].set_ylabel('DNA', fontsize=12)
    axes[1].axhline(threshold, **dash_style)
    
    # Overlay asterisks for DNA
    for t_idx in dna_indices:
        axes[1].text(time[t_idx], 1.2, '*', fontsize=9, ha='center')

    # PANEL 3: DNB Time-Series (Bottom Channel)
    threshold = 1.2
    axes[2].plot(time, seed_data_dnb, color='lightcoral', label='DNB', **line_props)
    axes[2].set_ylabel('DNB', fontsize=12)
    axes[2].axhline(threshold, **dash_style)
    
    # Overlay asterisks for DNB
    for t_idx in dnb_indices:
        axes[2].text(time[t_idx], 1.2, '*', fontsize=9, ha='center')
        
    # GLOBAL FORMATTING & VERTICAL ALIGNMENT LINES
    # Draw vertical dashed alignment markers across all subplots at trigger events
    for ax in axes:
        ax.grid(False)
    
    # Set shared X-axis parameters
    axes[2].set_xlabel('Time (sec)', fontsize=12)
    axes[2].set_xlim(0, time[-1])
    
    # Add a unified Y axis label text block on the left
    fig.supylabel('BOLD (z)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_name = os.path.join(fmri_output_directory, f"Trigger_profile_sub-{sub_id}.png")
    plt.savefig(plot_name, dpi=300)
    plt.show()


def main ():
    
    # fMRI data concatenation and processing for functional networks and switching rates
    fmri_base_directory = "/data/func_net_comp"
    fmri_output_directory = "/data/func_net_comp"
    
    seed_file = "seeds.txt"
    all_subject_seeds = load_subject_seeds(seed_file)

    concatenated_files = [
        f"{fmri_base_directory}/sub-01_concatenated_bold.nii.gz",
        f"{fmri_base_directory}/sub-02_concatenated_bold.nii.gz",
        f"{fmri_base_directory}/sub-03_concatenated_bold.nii.gz",
        f"{fmri_base_directory}/sub-04_concatenated_bold.nii.gz",
        f"{fmri_base_directory}/sub-05_concatenated_bold.nii.gz",
        f"{fmri_base_directory}/sub-06_concatenated_bold.nii.gz",
    ]

    subjects = ['01', '02', '03', '04', '05', '06']

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

        time = np.arange(len(seed_data_ecn)) * 1.5  # Replace with real TR

        plot_trigger(time, seed_data_ecn, seed_data_dna, seed_data_dnb, ecn_indices, dna_indices, dnb_indices, fmri_output_directory, sub)

    np.savetxt(os.path.join(fmri_output_directory, "rate_ecn_dna"), rate_ecn_dna_array)
    np.savetxt(os.path.join(fmri_output_directory, "rate_ecn_dnb"), rate_ecn_dnb_array)

    return 


if __name__ == "__main__":
    main()

"""
Copyright [2026] [Alesia Maria Budica]
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
