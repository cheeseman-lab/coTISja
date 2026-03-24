##### IMPORTS #####
import os
import sys
from pathlib import Path
sys.path.append('/lab/barcheese01/smaffa/coTISja/src')

from scripts.filter_utils import *
from scripts.analysis_pipeline_helpers import generate_coldata_table, assign_tids_to_tis_matrix
import re

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

deseq2 = importr("DESeq2")
apeglm = importr("apeglm")

### Define file inputs/outputs
MANIFEST_FILE = '/lab/barcheese01/smaffa/coTISja/data/ribotish_tisdiff_manifest.tsv'
GLOBAL_ANALYSIS_DIRECTORY = '/lab/barcheese01/smaffa/coTISja/data/tisdiff_results/deseq2/global'
CELL_LINE_ANALYSIS_DIRECTORY = '/lab/barcheese01/smaffa/coTISja/data/tisdiff_results/deseq2/cell_lines'
RPE1_ANALYSIS_DIRECTORY = '/lab/barcheese01/smaffa/coTISja/data/tisdiff_results/deseq2/RPE1_states'

# Import metadata & define parameters
experiment_table, sample_df, replicate_df = load_experiment_manifest()
tisdiff_manifest = pd.read_csv(MANIFEST_FILE, sep='\t')
samples = experiment_table['sample'].tolist()
codon_order = ['ATG', 'ATA', 'ATC', 'ATT', 'ACG', 'AAG', 'AGG', 'GTG', 'TTG', 'CTG']

# compile a global count matrix of TIS counts and RNAseq gene counts
pairs_to_count_matrix = dict()
for _, r in tisdiff_manifest.iterrows():
    baseline_s = r['baseline']
    test_s = r['test']
    counts_file = r['export_output_file']

    counts_matrix = pd.read_csv(counts_file, sep='\t', index_col=0)

    sample_pair  = [baseline_s, test_s]
    n_reps = counts_matrix.shape[1] // 4
    assays = ['rpf', 'rna']
    
    column_names = []
    for a in assays:
        for s in sample_pair:
            for i in range(1, n_reps+1):
                column_names.append(f'{s}__rep{i}__{a}')
    counts_matrix.columns = column_names
    pairs_to_count_matrix[(test_s, baseline_s)] = counts_matrix

# need to backfill RNA counts by GENE (for TISs which may only be absent in some samples), which requires a TIS ID to GENE ID mapping
# can get this mapping from a global concatenation of predict files
global_mapping = None
for pf in replicate_df[replicate_df['condition'] == 'TIS']['predict_file'].tolist():
    id_map = pd.read_csv(pf, sep='\t')[['Gid', 'Tid']].drop_duplicates()
    if global_mapping is None:
        global_mapping = id_map
    else:
        global_mapping = pd.concat([global_mapping, id_map], axis=0).drop_duplicates()
global_mapping = global_mapping.reset_index(drop=True)

# a bit convoluted because ribotish has already merged per-TIS Riboseq (rpf) counts with per-gene RNAseq (rna) counts, but we need to re-extract TIS rpf counts and gene rna counts
# 1) extract this rpf per TIS and rna per gene
# 2) concatenate all rpf counts (with TIS index), then map the TIS ids to corresponding transcript and gene ids
# 3) concatenate all RNA counts (with gene index)
# 4) merge the rpf matrix with the rna matrix
rpf_counts_by_sample = dict() 
rna_counts_by_replicate = dict()
for pair, counts in pairs_to_count_matrix.items():
    s1, s2 = pair

    if s1 not in rpf_counts_by_sample:
        all_sample_columns = [c for c in counts.columns if s1 in c]
        rpf_columns = [c for c in all_sample_columns if 'rpf' in c]
        rna_columns = [c for c in all_sample_columns if 'rna' in c]

        rpf_counts_by_sample[s1] = counts.loc[:, rpf_columns] # for rpf we concatenate first before mapping
        replicate_rna_counts = assign_tids_to_tis_matrix(counts.loc[:, rna_columns]).merge(global_mapping, how='left')  # for rna we map to transcript and gene ids
        for rep in rna_columns:
            rna_counts_by_replicate[rep] = replicate_rna_counts.loc[:, ['Gid', rep]].drop_duplicates().set_index('Gid')[rep] # get non-duplicated entries per gene
    if s2 not in rpf_counts_by_sample:
        all_sample_columns = [c for c in counts.columns if s2 in c]
        rpf_columns = [c for c in all_sample_columns if 'rpf' in c]
        rna_columns = [c for c in all_sample_columns if 'rna' in c]

        rpf_counts_by_sample[s2] = counts.loc[:, rpf_columns]
        replicate_rna_counts = assign_tids_to_tis_matrix(counts.loc[:, rna_columns]).merge(global_mapping, how='left')
        for rep in rna_columns:
            rna_counts_by_replicate[rep] = replicate_rna_counts.loc[:, ['Gid', rep]].drop_duplicates().set_index('Gid')[rep]

# merge the rpf counts and rna counts independently
all_rpf_counts = assign_tids_to_tis_matrix(pd.concat(rpf_counts_by_sample.values(), axis=1)).merge(global_mapping, how='left').fillna(0)
all_rna_counts = pd.concat(rna_counts_by_replicate, axis=1).fillna(0)
# map the counts together on gene id
all_counts = all_rpf_counts.merge(all_rna_counts, how='left', left_on='Gid', right_index=True).drop(['Tid', 'Gid'], axis=1).set_index('TIS')
all_counts = all_counts.astype(int)
# generate a sample metadata file for DESeq2
all_coldata = generate_coldata_table(all_counts.columns)

# export the counts matrix and the sample metadata table
# group by condition and mask based on sufficient readcounts for TIS and RNA (>= 5)
condition_counts = all_counts.T.merge(all_coldata[['condition', 'assay']], left_index=True, right_index=True).groupby(['condition', 'assay']).sum().T.loc[:, pd.IndexSlice[:, 'rpf']].droplevel(1, axis=1)
tis_mask = condition_counts >= 5
condition_rna_counts = all_counts.T.merge(all_coldata[['condition', 'assay']], left_index=True, right_index=True).groupby(['condition', 'assay']).sum().T.loc[:, pd.IndexSlice[:, 'rna']].droplevel(1, axis=1)
rna_mask = condition_rna_counts >= 5

# save readcounts by condition
condition_counts.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'rpf_summed_replicate_counts.csv'))
condition_counts[tis_mask].to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'rpf_summed_replicate_counts_masked.csv'))
condition_rna_counts.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'rna_summed_replicate_counts.csv'))
condition_rna_counts[rna_mask].to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'rna_summed_replicate_counts_masked.csv'))

# save full replicate-wise matrix
all_counts.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'input_counts_matrix.csv'))
all_coldata.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'input_sample_metadata.csv'))

##### Differential analysis #1: Global DESeq2 fit over all conditions, likelihood ratio test for any difference in TE across samples #####

# convert to R
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(all_counts)
    r_coldata = pandas2ri.py2rpy(all_coldata)

# Run DESeq2
r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_lrt_dds = deseq2.DESeq(r_dds, test='LRT', reduced=ro.Formula('~ assay + condition'))
r_lrt_res_df = ro.r("as.data.frame")(deseq2.results(r_lrt_dds))

# convert back to python
with localconverter(ro.default_converter + pandas2ri.converter):
    lrt_res_df = ro.conversion.rpy2py(r_lrt_res_df)

lrt_res_df.sort_values('padj').to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'lrt_results.csv'))


##### Differential analysis #2: Global DESeq2 fit over all conditions for translation efficiency (Ribo/RNA), export all pairwise comparisons #####

# convert to R
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(all_counts)
    r_coldata = pandas2ri.py2rpy(all_coldata)

# Run DESeq2
r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_wald_dds = deseq2.DESeq(r_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_dict = dict()
# conversion can't recognize string operators (-) --> need to instead run the pairwise comparisons in r eval strings, then convert to python
for pair in tqdm(pairs_to_count_matrix):
    # one sample (reference sample) is the "intercept term", figure out for each pair how to extract the comparison appropriately
    s1, s2 = pair
    if s1 == reference_sample:
        # swap the variables
        temp = s1
        s1 = s2
        s2 = temp
    if s2 == reference_sample: # reference sample is the intercept
        coeff_name = f'assayrpf.condition{s1}'
        res = ro.r(f'as.data.frame(results(r_wald_dds, name="{coeff_name}"))')
    else: # neither sample is the intercept
        contrast = f'list("assayrpf.condition{s1}", "assayrpf.condition{s2}")'
        res = ro.r(f'as.data.frame(results(r_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
    
    # convert to python
    with localconverter(ro.default_converter + pandas2ri.converter):
        pair_res_df = pandas2ri.rpy2py(res)

    pairwise_wald_dict[(s1, s2)] = pair_res_df
    pair_res_df.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, f'pairwise_wald/{s1}_vs_{s2}.csv'))


##### Differential analysis #3: Global DESeq2 fit over all conditions for RNAseq, export all pairwise comparisons #####

rna_counts = assign_tids_to_tis_matrix(all_counts[[c for c in all_counts.columns if 'rna' in c]]).merge(global_mapping, how='left')
rna_counts = rna_counts.drop_duplicates(subset=['Gid']).set_index('Gid').drop(['TIS', 'Tid'], axis=1)
rna_coldata = all_coldata[all_coldata['assay'] == 'rna']

# convert to R
with localconverter(ro.default_converter + pandas2ri.converter):
    r_rna_counts = pandas2ri.py2rpy(rna_counts)
    r_rna_coldata = pandas2ri.py2rpy(rna_coldata)

# Run DESeq2
r_rna_dds = deseq2.DESeqDataSetFromMatrix(countData=r_rna_counts, colData=r_rna_coldata, design=ro.Formula('~ condition'))
r_rna_wald_dds = deseq2.DESeq(r_rna_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_rna_coldata"] = r_rna_coldata
ro.globalenv["r_rna_wald_dds"] = r_rna_wald_dds
reference_sample = ro.r('levels(factor(r_rna_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_rna_dict = dict()

# conversion can't recognize string operators (-) well --> need to instead run this in r eval strings
for pair in tqdm(pairs_to_count_matrix):
    s1, s2 = pair
    if s1 == reference_sample:
        # swap the variables
        temp = s1
        s1 = s2
        s2 = temp
    if s2 == reference_sample: # reference sample is the intercept
        coeff_name = f'condition_{s1}_vs_{reference_sample}'
        res = ro.r(f'as.data.frame(results(r_rna_wald_dds, name="{coeff_name}"))')
    else: # neither sample is the intercept
        contrast = f'list("condition_{s1}_vs_{reference_sample}", "condition_{s2}_vs_{reference_sample}")'
        res = ro.r(f'as.data.frame(results(r_rna_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
    
    # convert to python
    with localconverter(ro.default_converter + pandas2ri.converter):
        pair_res_df = pandas2ri.rpy2py(res)

    pairwise_wald_rna_dict[(s1, s2)] = pair_res_df
    pair_res_df.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, f'pairwise_wald_rna/{s1}_vs_{s2}.csv'))


##### Extract the model's fitted TE values (without a variance-stabilizing transform) #####

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]
non_reference_samples = sorted(list(set(samples) - {reference_sample}))

# pull out coefficients per sample
te_coefficients = []
with localconverter(ro.default_converter + pandas2ri.converter):
    baseline = ro.conversion.rpy2py(ro.r('coef(r_wald_dds)[, "assay_rpf_vs_rna"]')) # reference sample
    te_coefficients.append(baseline)
    for s in non_reference_samples:
        addend_coef = ro.conversion.rpy2py(ro.r(f'coef(r_wald_dds)[, "assayrpf.condition{s}"]'))
        te_coefficients.append(baseline + addend_coef)
log_te = pd.DataFrame(te_coefficients, index=[reference_sample] + non_reference_samples, columns=all_counts.index).T
log_te.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'log_translation_efficiency_coeffs.csv'))


##### Extract the model's fitted TE values (after a variance-stabilizing transform) #####

# run the transformation
r_vst = deseq2.vst(r_dds, blind=True)

# extract the transformed matrix
r_vst_mtx = ro.r('assay')(r_vst)
with localconverter(ro.default_converter + pandas2ri.converter):
    vst_mtx = pd.DataFrame(ro.conversion.rpy2py(r_vst_mtx), index=all_counts.index, columns=all_counts.columns)

# both RNA and Riboseq reads have been variance stabilized and logged (i.e. adjusted for mean-variance relationship), average over replicates per sample
vst_avg_mtx = vst_mtx.T.groupby([all_coldata['condition'], all_coldata['assay']]).mean().T

# subtracting log(RPF) - log(RNA) gives log(RPF/RNA) ~= log(TE)
vst_avg_te = vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rpf']].droplevel(1, axis=1) - vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rna']].droplevel(1, axis=1)

vst_avg_te.to_csv(os.path.join(GLOBAL_ANALYSIS_DIRECTORY, 'translation_efficiency_vst_matrix.csv'))


##### Differential analysis on subsets #####

### Cell lines ###

# inputs
subset_sample_names = ['HeLa', 'K562', 'RPE1_Async', 'U2OS']
subset_coldata = all_coldata[all_coldata['condition'].isin(subset_sample_names)]
subset_counts = all_counts.loc[:, subset_coldata.index.tolist()]
subset_counts = subset_counts[(subset_counts[[c for c in subset_counts.columns if 'rpf' in c]].sum(axis=1) > 0)]

subset_counts.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, 'input_counts_matrix.csv'))
subset_coldata.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, 'input_sample_metadata.csv'))

# LRT test
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(subset_counts)
    r_coldata = pandas2ri.py2rpy(subset_coldata)

r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_lrt_dds = deseq2.DESeq(r_dds, test='LRT', reduced=ro.Formula('~ assay + condition'))
r_lrt_res_df = ro.r("as.data.frame")(deseq2.results(r_lrt_dds))

with localconverter(ro.default_converter + pandas2ri.converter):
    lrt_res_df = ro.conversion.rpy2py(r_lrt_res_df)

lrt_res_df.sort_values('padj').to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, 'lrt_results.csv'))

# Pairwise TIS
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(subset_counts)
    r_coldata = pandas2ri.py2rpy(subset_coldata)

r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_wald_dds = deseq2.DESeq(r_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_dict = dict()

# conversion can't recognize string operators (-) well --> need to instead run this in r eval strings
for pair in tqdm(pairs_to_count_matrix):
    s1, s2 = pair
    if s1 in subset_sample_names and s2 in subset_sample_names:
        if s1 == reference_sample:
            # swap the variables
            temp = s1
            s1 = s2
            s2 = temp
        if s2 == reference_sample:
            coeff_name = f'assayrpf.condition{s1}'
            res = ro.r(f'as.data.frame(results(r_wald_dds, name="{coeff_name}"))')
        else:
            contrast = f'list("assayrpf.condition{s1}", "assayrpf.condition{s2}")'
            res = ro.r(f'as.data.frame(results(r_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            pair_res_df = pandas2ri.rpy2py(res)

        pairwise_wald_dict[(s1, s2)] = pair_res_df
        pair_res_df.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, f'pairwise_wald/{s1}_vs_{s2}.csv'))

# pairwise RNA
rna_counts = assign_tids_to_tis_matrix(subset_counts[[c for c in subset_counts.columns if 'rna' in c]]).merge(global_mapping, how='left')
rna_counts = rna_counts.drop_duplicates(subset=['Gid']).set_index('Gid').drop(['TIS', 'Tid'], axis=1)
rna_coldata = subset_coldata[subset_coldata['assay'] == 'rna']

with localconverter(ro.default_converter + pandas2ri.converter):
    r_rna_counts = pandas2ri.py2rpy(rna_counts)
    r_rna_coldata = pandas2ri.py2rpy(rna_coldata)

r_rna_dds = deseq2.DESeqDataSetFromMatrix(countData=r_rna_counts, colData=r_rna_coldata, design=ro.Formula('~ condition'))
r_rna_wald_dds = deseq2.DESeq(r_rna_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_rna_coldata"] = r_rna_coldata
ro.globalenv["r_rna_wald_dds"] = r_rna_wald_dds
reference_sample = ro.r('levels(factor(r_rna_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_rna_dict = dict()

# conversion can't recognize string operators (-) well --> need to instead run this in r eval strings
for pair in tqdm(pairs_to_count_matrix):
    s1, s2 = pair
    if s1 in subset_sample_names and s2 in subset_sample_names:
        if s1 == reference_sample:
            # swap the variables
            temp = s1
            s1 = s2
            s2 = temp
        if s2 == reference_sample:
            coeff_name = f'condition_{s1}_vs_{reference_sample}'
            res = ro.r(f'as.data.frame(results(r_rna_wald_dds, name="{coeff_name}"))')
        else:
            contrast = f'list("condition_{s1}_vs_{reference_sample}", "condition_{s2}_vs_{reference_sample}")'
            res = ro.r(f'as.data.frame(results(r_rna_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            pair_res_df = pandas2ri.rpy2py(res)

        pairwise_wald_rna_dict[(s1, s2)] = pair_res_df
        pair_res_df.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, f'pairwise_wald_rna/{s1}_vs_{s2}.csv'))

# Fitted TE values

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]
non_reference_samples = sorted(list(set(subset_sample_names) - {reference_sample}))

te_coefficients = []
with localconverter(ro.default_converter + pandas2ri.converter):
    baseline = ro.conversion.rpy2py(ro.r('coef(r_wald_dds)[, "assay_rpf_vs_rna"]'))
    te_coefficients.append(baseline)
    for s in non_reference_samples:
        addend_coef = ro.conversion.rpy2py(ro.r(f'coef(r_wald_dds)[, "assayrpf.condition{s}"]'))
        te_coefficients.append(baseline + addend_coef)
log_te = pd.DataFrame(te_coefficients, index=[reference_sample] + non_reference_samples, columns=subset_counts.index).T

log_te.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, 'log_translation_efficiency_coeffs.csv'))


# run the transformation
r_vst = deseq2.vst(r_dds, blind=True)

# extract the transformed matrix
r_vst_mtx = ro.r('assay')(r_vst)
with localconverter(ro.default_converter + pandas2ri.converter):
    vst_mtx = pd.DataFrame(ro.conversion.rpy2py(r_vst_mtx), index=subset_counts.index, columns=subset_counts.columns)

# both RNA and Riboseq reads have been variance stabilized and logged (i.e. adjusted for mean-variance relationship), average over replicates per sample
vst_avg_mtx = vst_mtx.T.groupby([subset_coldata['condition'], subset_coldata['assay']]).mean().T

# subtracting log(RPF) - log(RNA) gives log(RPF/RNA) ~= log(TE)
vst_avg_te = vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rpf']].droplevel(1, axis=1) - vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rna']].droplevel(1, axis=1)

vst_avg_te.to_csv(os.path.join(CELL_LINE_ANALYSIS_DIRECTORY, 'translation_efficiency_vst_matrix.csv'))


### RPE1 states ###

# inputs
subset_sample_names = ['RPE1_Async', 'RPE1_Que', 'RPE1_Sen']
subset_coldata = all_coldata[all_coldata['condition'].isin(subset_sample_names)]
subset_counts = all_counts.loc[:, subset_coldata.index.tolist()]
subset_counts = subset_counts[(subset_counts[[c for c in subset_counts.columns if 'rpf' in c]].sum(axis=1) > 0)]

subset_counts.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, 'input_counts_matrix.csv'))
subset_coldata.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, 'input_sample_metadata.csv'))

# LRT test
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(subset_counts)
    r_coldata = pandas2ri.py2rpy(subset_coldata)

r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_lrt_dds = deseq2.DESeq(r_dds, test='LRT', reduced=ro.Formula('~ assay + condition'))
r_lrt_res_df = ro.r("as.data.frame")(deseq2.results(r_lrt_dds))

with localconverter(ro.default_converter + pandas2ri.converter):
    lrt_res_df = ro.conversion.rpy2py(r_lrt_res_df)

lrt_res_df.sort_values('padj').to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, 'lrt_results.csv'))

# Pairwise TIS
with localconverter(ro.default_converter + pandas2ri.converter):
    r_counts = pandas2ri.py2rpy(subset_counts)
    r_coldata = pandas2ri.py2rpy(subset_coldata)

r_dds = deseq2.DESeqDataSetFromMatrix(countData=r_counts, colData=r_coldata, design=ro.Formula('~ assay + condition + assay:condition'))
r_wald_dds = deseq2.DESeq(r_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_dict = dict()

# conversion can't recognize string operators (-) well --> need to instead run this in r eval strings
for pair in tqdm(pairs_to_count_matrix):
    s1, s2 = pair
    if s1 in subset_sample_names and s2 in subset_sample_names:
        if s1 == reference_sample:
            # swap the variables
            temp = s1
            s1 = s2
            s2 = temp
        if s2 == reference_sample:
            coeff_name = f'assayrpf.condition{s1}'
            res = ro.r(f'as.data.frame(results(r_wald_dds, name="{coeff_name}"))')
        else:
            contrast = f'list("assayrpf.condition{s1}", "assayrpf.condition{s2}")'
            res = ro.r(f'as.data.frame(results(r_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            pair_res_df = pandas2ri.rpy2py(res)

        pairwise_wald_dict[(s1, s2)] = pair_res_df
        pair_res_df.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, f'pairwise_wald/{s1}_vs_{s2}.csv'))

# pairwise RNA
rna_counts = assign_tids_to_tis_matrix(subset_counts[[c for c in subset_counts.columns if 'rna' in c]]).merge(global_mapping, how='left')
rna_counts = rna_counts.drop_duplicates(subset=['Gid']).set_index('Gid').drop(['TIS', 'Tid'], axis=1)
rna_coldata = subset_coldata[subset_coldata['assay'] == 'rna']

with localconverter(ro.default_converter + pandas2ri.converter):
    r_rna_counts = pandas2ri.py2rpy(rna_counts)
    r_rna_coldata = pandas2ri.py2rpy(rna_coldata)

r_rna_dds = deseq2.DESeqDataSetFromMatrix(countData=r_rna_counts, colData=r_rna_coldata, design=ro.Formula('~ condition'))
r_rna_wald_dds = deseq2.DESeq(r_rna_dds)

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_rna_coldata"] = r_rna_coldata
ro.globalenv["r_rna_wald_dds"] = r_rna_wald_dds
reference_sample = ro.r('levels(factor(r_rna_coldata$condition))')[0]

# for all pairwise comparisons, extract the results
pairwise_wald_rna_dict = dict()

# conversion can't recognize string operators (-) well --> need to instead run this in r eval strings
for pair in tqdm(pairs_to_count_matrix):
    s1, s2 = pair
    if s1 in subset_sample_names and s2 in subset_sample_names:
        if s1 == reference_sample:
            # swap the variables
            temp = s1
            s1 = s2
            s2 = temp
        if s2 == reference_sample:
            coeff_name = f'condition_{s1}_vs_{reference_sample}'
            res = ro.r(f'as.data.frame(results(r_rna_wald_dds, name="{coeff_name}"))')
        else:
            contrast = f'list("condition_{s1}_vs_{reference_sample}", "condition_{s2}_vs_{reference_sample}")'
            res = ro.r(f'as.data.frame(results(r_rna_wald_dds, contrast={contrast}, listValues=c(1, -1)))')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            pair_res_df = pandas2ri.rpy2py(res)

        pairwise_wald_rna_dict[(s1, s2)] = pair_res_df
        pair_res_df.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, f'pairwise_wald_rna/{s1}_vs_{s2}.csv'))

# Fitted TE values

# determine which sample was used as the reference by DESeq2 (simply alphabetical)
ro.globalenv["r_coldata"] = r_coldata
ro.globalenv["r_wald_dds"] = r_wald_dds
reference_sample = ro.r('levels(factor(r_coldata$condition))')[0]
non_reference_samples = sorted(list(set(subset_sample_names) - {reference_sample}))

te_coefficients = []
with localconverter(ro.default_converter + pandas2ri.converter):
    baseline = ro.conversion.rpy2py(ro.r('coef(r_wald_dds)[, "assay_rpf_vs_rna"]'))
    te_coefficients.append(baseline)
    for s in non_reference_samples:
        addend_coef = ro.conversion.rpy2py(ro.r(f'coef(r_wald_dds)[, "assayrpf.condition{s}"]'))
        te_coefficients.append(baseline + addend_coef)
log_te = pd.DataFrame(te_coefficients, index=[reference_sample] + non_reference_samples, columns=subset_counts.index).T

log_te.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, 'log_translation_efficiency_coeffs.csv'))


# run the transformation
r_vst = deseq2.vst(r_dds, blind=True)

# extract the transformed matrix
r_vst_mtx = ro.r('assay')(r_vst)
with localconverter(ro.default_converter + pandas2ri.converter):
    vst_mtx = pd.DataFrame(ro.conversion.rpy2py(r_vst_mtx), index=subset_counts.index, columns=subset_counts.columns)

# both RNA and Riboseq reads have been variance stabilized and logged (i.e. adjusted for mean-variance relationship), average over replicates per sample
vst_avg_mtx = vst_mtx.T.groupby([subset_coldata['condition'], subset_coldata['assay']]).mean().T

# subtracting log(RPF) - log(RNA) gives log(RPF/RNA) ~= log(TE)
vst_avg_te = vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rpf']].droplevel(1, axis=1) - vst_avg_mtx.loc[:, pd.IndexSlice[:, 'rna']].droplevel(1, axis=1)

vst_avg_te.to_csv(os.path.join(RPE1_ANALYSIS_DIRECTORY, 'translation_efficiency_vst_matrix.csv'))