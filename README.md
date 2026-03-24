# coTISja
TIS characterization across genes and cell lines

# Setup

We use a conda-managed environment with python 3.11, saved to `riboseq.yml`. A new environment can be created and activated from this file by running: 

```
conda env create -f riboseq.yml
conda activate riboseq
```

# src

Contains exploratory notebooks for analysis and scripts for major processing steps.

## notebooks

Consists primarily of prototypes for processing scripts, running analyses, or figure generation. The following processes and analyses have been prototyped:
- Filtering putative start sites called by ribotish (`filter_ribotish.ipynb`)
- Differential translation initiation analysis using DESeq2 (`differential_tis.ipynb`, `sample_comparison.ipynb`)
- Predictive modeling of translation efficiency (`tis_gene_analysis.ipynb`, `leaky_scanning_model.ipynb`)
- Outlier analyses of translation efficiency (`sample_comparison.ipynb`)

Other ad-hoc tasks include:
- Preparing manifest CSVs for arrayed sbatch scripts (`filter_ribotish.ipynb`)
- Formatting tracks for IGV (`prepare_igv_files.ipynb`)

Refer to the README in this folder for more details and suggestions for future directions.

## scripts

Scripts for implementing key pre-processing steps are located here:

- Filtering putative start sites called by ribotish (`filter_ribotish.py`)
- Differential analysis across samples ()
- Fit models for predicting canonical TIS translation efficiency from sequence-based determinants (`fit_leaky_scanning_model.py`)

Refer to the README in this folder for more details.
