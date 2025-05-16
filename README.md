# Analysis of nematic ordering of HUVEC

This repository contains custom scripts for analyzing endothelial cell (HUVEC) alignment. It relies on a scientific tool to analyise nematic ordering.
See: [AFT - Alignment by Fourier Transform](https://github.com/OakesLab/AFT-Alignment_by_Fourier_Transform)

## Workflow Summary

### 1. Imaging & Data Extraction
- HUVEC were cultured, fixed, and fluorescently stained.
- Imaging was performed using a Nikon SDCM microscope.
- `.nd2` files were processed in FIJI to extract VE-cadherina and nuclei signals.

### 3. Cellsegmentation
- HUVEC were segmented using cellpose.

### 2. Nematic Ordering Analysis
- **Script**: `nematic_ordering.ipynb`
- **Steps**:
  1. Analyze nematic ordering using AFT.
  2. Calculate order parameters and visualize alignment.
  3. Use MATLAB's `AFT_batch.m` for alignment vectors and heatmaps.

## Requirements

| Software/Tool   | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Python 3.x**  | Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-image`, `cellpose`, `SciPy`, `Statsmodels`, `Scikit-posthocs`. |
| **MATLAB**      | Needed for running `AFT_batch.m` in the Nematic Ordering Analysis section.  |
| **FIJI/ImageJ** | Used for preprocessing and segmenting microscopy images.                   |
| **AFT Tool**    | Required for nematic ordering analysis using the AFT method.               |
| **Cellpose**    | Used to segment HUVEC and extract their outlines             |


### Conda Environments
| Environment File         | Purpose                                      |
|---------------------------|----------------------------------------------|
| `analysis_env.yml`        | Processing nematic ordering data.            |
| `aft_312_env.yml`         | Calculating order parameters in AFT scripts. |

# Citation

If you use this code for your own work, please cite this repo, the corresponding paper and/or the AFT method where appropriate.
DOI:

# License
The content of this project itself is licensed under the MIT license.
