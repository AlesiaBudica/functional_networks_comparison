import numpy as np
import nigsp as ng

def main():
    brain_mask_path = "/Neurodata/M3PI/derivatives/vessels/sub-03/ses-7T/anat/00.sub-03_ses-7T_part-mag_T2starw_brain_mask.nii.gz"
    _, brain_mask, _ = ng.io.load_nifti_get_mask(brain_mask_path, is_mask=True)
    brain_mask_voxels = np.sum(brain_mask)

    vessel_mask_path = "/Neurodata/M3PI/derivatives/vessels/sub-03/ses-7T/anat/sub-03_ses-7T_acq-invRO_run-1_optcom_part-mag_T2starw2vesselref.nii.gz"
    _, vessel_mask, _ = ng.io.load_nifti_get_mask(vessel_mask_path, is_mask=True)
    vessel_mask_voxels = np.sum(vessel_mask * brain_mask_voxels)

    vascular_density = vessel_mask_voxels / brain_mask_voxels

    print(f"Total Brain Voxels: {brain_mask_voxels}")
    print(f"Total Vessel Voxels (within the given mask): {vessel_mask_voxels}")
    print(f"Calculated Whole-Brain Vascular Density: {vascular_density:.6f}")

    return 

if __name__ == "__main__":
    main()

"""
Copyright [2026] [Alesia Maria Budica]
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""