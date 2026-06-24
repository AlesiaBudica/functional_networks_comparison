#!/usr/bin/env bash

for sub in $(seq -f %02g 1 6)
do
	anat_reference=/data/vessels/sub-${sub}/ses-7T/reg/sub-${sub}_vesselref.nii.gz

	T1w2T2starw=/data/vessels/sub-${sub}/ses-7T/reg/sub-${sub}_ses-02_UNIT12vesselref0GenericAffine.mat
	T2w2T1w=/data/vessels/sub-${sub}/ses-02/reg/sub-${sub}_ses-02_T2w2UNIT10GenericAffine.mat
	T2w2func=/data/vessels/sub-${sub}/ses-01/reg/sub-${sub}_ses-02_T2w2sbref0GenericAffine.mat

	for map in ecn dna dnb
	do
		antsApplyTransforms -d 3 -i /data/func_net_comp/map_${map}_mask_sub-${sub}.nii.gz \
							-r ${anat_reference}.nii.gz \
							-o /data/func_net_comp/map_${map}_mask_sub-${sub}_vesselres.nii.gz \
							-n MultiLabel \
							-t ${T1w2T2starw} \
							-t ${T2w2T1w} \
							-t [${T2w2func},1]
	done
done




# Copyright 2026, Alesia Budica.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.