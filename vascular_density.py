import numpy as np
import nigsp as ng

def main():
    brain_mask_path = "/Neurodata/M3PI/derivatives/vessels/sub-03/ses-7T/anat/00.sub-03_ses-7T_part-mag_T2starw_brain_mask.nii.gz"
    _, brain_mask, _ = ng.io.load_nifti_get_mask(brain_mask_path, is_mask=True)
    brain_mask_voxels = np.sum(brain_mask)

    vessel_mask_path = "/Neurodata/M3PI/derivatives/vessels/sub-03/ses-7T/anat/sub-03_ses-7T_acq-invRO_run-1_optcom_part-mag_T2starw2vesselref.nii.gz"
    _, vessel_mask, _ = ng.io.load_nifti_get_mask(vessel_mask_path, is_mask=True)
    vessel_mask_voxels = np.sum(vessel_mask)

    vascular_density = vessel_mask_voxels / brain_mask_voxels

    print(f"Total Brain Voxels: {brain_mask_voxels}")
    print(f"Total Vessel Voxels: {vessel_mask_voxels}")
    print(f"Calculated Whole-Brain Vascular Density: {vascular_density:.6f}")

    return 

if __name__ == "__main__":
    main()