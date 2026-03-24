import os
import sys
from pathlib import Path
sys.path.append('/lab/barcheese01/smaffa/coTISja/src')

from scripts.plotting import *
from scripts.filter_utils import *
from scripts.analysis_pipeline_helpers import define_global_tis_reference, read_transcript_sequences, assign_canonical_kozak_annotations, featurize_by_canonical_sites, write_glm_results
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf

##### Define filepaths and model parameters #####
REPLICATE_LEVEL_TIS_METADATA_FILE = '/lab/barcheese01/smaffa/coTISja/data/filtered_tis_data/per_rep/all_samples_filtered_with_protein_seq.csv'
GLOBAL_INPUT_DIRECTORY = '/lab/barcheese01/smaffa/coTISja/data/tisdiff_results/deseq2/global'
ANALYSIS_DIRECTORY = '/lab/barcheese01/smaffa/coTISja/data/translation_models'
MODEL_LABEL = 'canonical_translation'
FORMULA = """
    TISCounts ~ 
    KozakMajorHammingDistance +
    NumUpstreamATG +
    NumUpstreamNonATG +
    CanonicalUTRLength +
    Sample +
    KozakMajorHammingDistance:Sample +
    NumUpstreamATG:Sample +
    NumUpstreamNonATG:Sample +
    CanonicalUTRLength:Sample
"""

# Load data to format into appropriate inputs

# metadata
experiment_table, _, replicate_df = load_experiment_manifest()

# replicate level TIS data
all_replicate_tis_df = pd.read_csv(REPLICATE_LEVEL_TIS_METADATA_FILE)
all_replicate_tis_df = all_replicate_tis_df.assign(
    IsoformID=all_replicate_tis_df['Tid'] + ':' + all_replicate_tis_df['Start'].astype(int).astype(str),
    TIS_ID=all_replicate_tis_df['Gid'] + ':' + all_replicate_tis_df['GenomeStart'],
    TIS=all_replicate_tis_df.apply(lambda x: f'{x["Tid"]}_{int(x["Start"])}_{x["GenomeStart"]}:{x["GenomePos"].split(":")[-1]}', axis=1)
)

# assign logRPM RNA-seq values per TIS
all_replicate_tis_df = calculate_normalization_factors(
    all_replicate_tis_df, 
    id_columns = ['Sample', 'Replicate'],
    reference_sample = ('HeLa', 'rep1'),
    experiment_table=replicate_df[replicate_df['condition'] == 'TIS'].assign(rnaseq_count_file=lambda x: x['rnaseq_count_file'].apply(lambda y: [y])),
    id_cols = ['sample', 'replicate']
)

# load transcript sequences to extract Kozak contexts
reference_transcript_sequences = read_transcript_sequences()

# load TIS and RNAseq data to filter on sufficient reads
vst_log_te_matrix = pd.read_csv(os.path.join(GLOBAL_INPUT_DIRECTORY, 'translation_efficiency_vst_matrix.csv'), index_col=0)
riboseq_counts = pd.read_csv(os.path.join(GLOBAL_INPUT_DIRECTORY, 'rpf_summed_replicate_counts.csv'), index_col=0)
tis_mask = riboseq_counts >= 5
rna_counts = pd.read_csv(os.path.join(GLOBAL_INPUT_DIRECTORY, 'rna_summed_replicate_counts.csv'), index_col=0)
rna_mask = rna_counts >= 5
masked_te_matrix = vst_log_te_matrix[tis_mask & rna_mask].dropna(how='all')

# construct the input dataframe, where each transcript constitutes one observation
global_tis_reference = define_global_tis_reference(all_replicate_tis_df, masked_te_matrix)
global_tis_reference_df = assign_canonical_kozak_annotations(global_tis_reference, reference_transcript_sequences)
feature_table = featurize_by_canonical_sites(tid_summary_metadata=global_tis_reference_df, tis_df=all_replicate_tis_df)

# export input dataframe before modeling
global_tis_reference_df.to_csv(os.path.join(ANALYSIS_DIRECTORY, 'transcript_tis_vector_annotations.csv'), index=False)
feature_table.to_csv(os.path.join(ANALYSIS_DIRECTORY, 'glm_model_inputs.csv'), index=False)

# construct missing features and normalize some features
feature_table = feature_table.assign(
    CanonicalUTRLength = lambda x: x['CanonicalUTRLength'] / 1000,
    NumUpstreamNonATG=lambda x: x['NumUpstreamTIS'] - x['NumUpstreamATG']
)

# construct and fit the model
model = smf.glm(
    formula=FORMULA,
    data=feature_table,
    family=sm.families.NegativeBinomial(), # use the negative binomial family, because TISs are in discrete counts
    offset=feature_table["GeneRNASeqLogRPM"] # set intercept terms using the RNA expression (efficiency will be relative to expression)
).fit(cov_type='HC0')

write_glm_results(model, output_dir=ANALYSIS_DIRECTORY, model_prefix=MODEL_LABEL)