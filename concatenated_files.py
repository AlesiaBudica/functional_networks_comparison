#!/usr/bin/env python3

import os

import nibabel as nb
from nibabel.funcs import concat_images


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

    # Loop over the 6 subjects
    for sub in subject_ids:
        session_files = []
        
        # Loop over the 6 sessions for the current subject
        for ses in session_ids:
            file_name = f"00.sub-{sub}_ses-{ses}_task-simon_optcom_native_preprocessed.nii.gz"
            file_path = os.path.join(data_dir, f"sub-{sub}", f"ses-{ses}", "func", file_name)
            
            if os.path.exists(file_path):
                print(f"Loading: {file_name}")
                nifti_img = nb.load(file_path)
                session_files.append(nifti_img)
            else:
                print(f"Warning: File missing at {file_path}")

        # Concatenate and save files for the current subject
        if session_files:
            print(f"--> Concatenating 6 sessions for subject: {sub}")
            concat_file = concat_images(session_files, axis=-1)
            
            # Define output filename and path
            output_name = f"sub-{sub}_concatenated_bold.nii.gz"
            output_path = os.path.abspath(os.path.join(output_dir, output_name))
            
            # Save the new combined 4D image to disk
            nb.save(concat_file, output_path)
            print(f"Successfully saved: {output_name}\n")
            
            # Append the completed path to our tracking array
            saved_paths.append(output_path)
        else:
            print(f"Error: No sessions found for subject {sub}\n")

    # Return the completed array of file paths
    return saved_paths


def main ():
    
    # fMRI data concatenation
    no_subjects = 6
    no_sessions = 6
    subjects = [f"{i:02d}" for i in range(1, no_subjects + 1)]
    sessions = [f"{i:02d}" for i in range(1, no_sessions + 1)]

    fmri_base_directory = "/data/vessels"  # /data will be bound to /Neurodata/M3PI/derivatives
    fmri_output_directory = "/data/func_net_comp"

    read_and_concatenate_subject_sessions(fmri_base_directory, fmri_output_directory, subjects, sessions)

    return


if __name__ == "__main__":
    main()

"""
Copyright [2026] [Alesia Maria Budica]
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
