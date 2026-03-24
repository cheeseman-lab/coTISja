import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm
import os

from Bio import SeqIO, SeqUtils

# Define IUPAC ambiguity codes
ambiguity_dict = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'}, 'U': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'M': {'A', 'C'},
    'K': {'G', 'T'}, 'S': {'C', 'G'}, 'W': {'A', 'T'},
    'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'}, 'H': {'A', 'C', 'T'},
    'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'},
    'a': {'A'}, 'c': {'C'}, 'g': {'G'}, 't': {'T'}, 'u': {'T'}
}

kozak_pattern = 'gccgccRccATGG'
partial_weights = [0.1,0.1,0.1,0.1,0.1,0.1,1,0.1,0.1,1,1,1,1]
major_weights = [0,0,0,0,0,0,1,0,0,1,1,1,1]

##### Differential translation & expression #####

# make metadata table from sample names, assuming that names are in order of sample, replicate, assay
def generate_coldata_table(column_names, sep='__'):
    metadata_df = pd.DataFrame({
        'colname': column_names,
        'condition': [x.split(sep)[0] for x in column_names],
        'replicate': [x.split(sep)[1] for x in column_names],
        'assay': [x.split(sep)[2] for x in column_names]
    }).set_index('colname')
    return metadata_df

# Convert the indices of the TIS matrix into transcript IDS
def assign_tids_to_tis_matrix(tis_matrix):
    matrix_copy = tis_matrix.copy()
    matrix_copy['Tid'] = tis_matrix.index.str.split('_').str[0]
    return matrix_copy.reset_index()


##### TIS variance across samples #####

# function to pull out statistics for a TIS across all pairwise comparisons
def extract_TIS_matrix(TIS, pairwise_dict, masked_tis_matrix, comparison_metric='log2FoldChange', ):
    single_samples = set()
    for s1, s2 in pairwise_dict:
        single_samples.add(s1)
        single_samples.add(s2)
    single_samples = sorted(list(single_samples))
    n_samples = len(single_samples)

    output_matrix = np.empty(shape=(n_samples, n_samples))
    output_matrix.fill(np.nan)
    output_matrix = pd.DataFrame(output_matrix, index=single_samples, columns=single_samples)

    for pair, df in pairwise_dict.items():
        s1, s2 = pair
        if TIS in df.index.tolist():
            output_matrix.loc[s1, s2] = df.loc[TIS, comparison_metric]
        if masked_tis_matrix is not None:
            if TIS in masked_tis_matrix.index.tolist() and s1 in masked_tis_matrix.columns.tolist() and s2 in masked_tis_matrix.columns.tolist():
                if np.isnan(masked_tis_matrix.loc[TIS, s1]) and np.isnan(masked_tis_matrix.loc[TIS, s2]):
                    output_matrix.loc[s1, s2] = np.nan

    return output_matrix

# function to pull out a stacked dataframe of values for each TIS, stacking across all individual TIS matrices
def extract_TIS_block(pairwise_dict, masked_tis_matrix, comparison_metric='log2FoldChange'):
    single_samples = set()
    tis_superindex = set()
    for s1, s2 in pairwise_dict:
        single_samples.add(s1)
        single_samples.add(s2)
        tis_superindex = tis_superindex.union(pairwise_dict[(s1, s2)].index.tolist())
    single_samples = sorted(list(single_samples))
    tis_superindex = sorted(list(tis_superindex))
    n_samples = len(single_samples)
    n_tis = len(tis_superindex)

    sample_index = pd.Series(range(n_samples), index=single_samples)
    tis_index = pd.Series(range(n_tis), index=tis_superindex)

    output_matrix = np.empty(shape=(n_tis, n_samples, n_samples))
    output_matrix.fill(np.nan)

    for pair, df in tqdm(pairwise_dict.items()):
        s1, s2 = pair
        output_matrix[:, sample_index.loc[s1], sample_index.loc[s2]] = df.reindex(index=tis_superindex)[comparison_metric].values

        if masked_tis_matrix is not None:
            na_mask = masked_tis_matrix.reindex(index=tis_superindex)[[s1, s2]].isna().all(axis=1).tolist()
            output_matrix[na_mask, sample_index.loc[s1], sample_index.loc[s2]] = np.nan

    output_block = pd.DataFrame(
        output_matrix.reshape(tis_index.shape[0] * sample_index.shape[0], sample_index.shape[0]),
        index=pd.MultiIndex.from_product([tis_index.index.tolist(), sample_index.index.tolist()], names=['TIS', 'TestSample']),
        columns=sample_index.index.tolist()
    )
    output_block.columns.name='RefSample'
    return output_block

# compute changes in variance using a leave-one-out procedure
def calculate_variance_shifts(matrix, baseline_variance=None):
    """
    Calculates the changes in variance across the rows after removing each column
    """
    if baseline_variance is None:
        baseline_variance = matrix.var(axis=1)
    column_dropped_to_remaining_variance = dict()
    for c in matrix.columns:
        column_dropped_to_remaining_variance[c] = matrix.drop([c], axis=1).var(axis=1).rename(c)
    dropped_variances = pd.concat(column_dropped_to_remaining_variance, axis=1)

    variance_shifts = (dropped_variances.T - baseline_variance).T
    variance_shifts[~(matrix.isna()) & dropped_variances.isna()] = -np.inf
    return variance_shifts, dropped_variances

# compute changes in range using a leave-one-out procedure
def calculate_range_shifts(matrix):
    """
    Calculates the changes in range across the rows after removing each column
    """
    baseline_range = matrix.max(axis=1) - matrix.min(axis=1)
    column_dropped_to_remaining_range = dict()
    for c in matrix.columns:
        dropped_matrix = matrix.drop([c], axis=1)
        column_dropped_to_remaining_range[c] = (dropped_matrix.max(axis=1) - dropped_matrix.min(axis=1)).rename(c)
    dropped_ranges = pd.concat(column_dropped_to_remaining_range, axis=1)

    range_shifts = (dropped_ranges.T - baseline_range).T
    return range_shifts, dropped_ranges

# construct a dataframe of summary metrics per TIS across all pairwise comparisons
def summarize_tis_stats_across_samples(
    te_matrix, 
    lfc_stack, qval_stack, 
    tis_counts, total_tis_counts, tis_metadata_df, 
    sample_subset=None
):
    if sample_subset is None:
        sample_subset = te_matrix.columns.tolist()
    # reshape inputs depending on sample subset
    te_matrix = te_matrix.loc[:, sample_subset]
    lfc_stack = lfc_stack.loc[pd.IndexSlice[:, sample_subset], sample_subset]
    qval_stack = qval_stack.loc[pd.IndexSlice[:, sample_subset], sample_subset]
    tis_counts = tis_counts.loc[:, sample_subset]
    total_tis_counts = total_tis_counts.loc[sample_subset]

    # overall metrics on presence and variability across samples
    n_samples_per_tis = (~(te_matrix.isna())).sum(axis=1).rename('NSamples')
    variance_per_tis = te_matrix.var(axis=1).rename('LogTEVariance')

    # outlier detection by changes in spread
    variance_shifts, dropped_variances = calculate_variance_shifts(te_matrix, baseline_variance=variance_per_tis)
    min_delta_variance = variance_shifts.min(axis=1).rename('LeaveOneOutLargestDeltaVariance')
    range_shifts, dropped_ranges = calculate_range_shifts(te_matrix)
    min_delta_range = range_shifts.min(axis=1).rename('LeaveOneOutLargestDeltaRange')
    min_delta_variance_sample = variance_shifts.dropna(how='all').idxmin(axis=1).rename('CandidateOutlier').reindex(index=min_delta_variance.index)

    # Ribo/RNA values
    min_te_per_tis = te_matrix.min(axis=1).rename('MinLogRiboOverRNA')
    min_te_sample_per_tis = te_matrix.dropna(how='all').idxmin(axis=1).rename('MinLogTESample')
    max_te_per_tis = te_matrix.max(axis=1).rename('MaxLogRiboOverRNA')
    max_te_sample_per_tis = te_matrix.dropna(how='all').idxmax(axis=1).rename('MaxLogTESample')

    # metrics of fold changes and q-values for all pairwise comparisons
    lfc_reshaped = lfc_stack.melt(ignore_index=False).reset_index().pivot(index='TIS', columns=['TestSample', 'RefSample'], values='value')
    qval_reshaped = qval_stack.melt(ignore_index=False).reset_index().pivot(index='TIS', columns=['TestSample', 'RefSample'], values='value')
    min_lfc_per_tis = lfc_reshaped.min(axis=1).rename('MinLFC')
    min_lfc_pair_per_tis = lfc_reshaped.dropna(how='all').idxmin(axis=1).apply(lambda x: '/'.join(x)).rename('MinLFCSamplePair')
    max_lfc_per_tis = lfc_reshaped.max(axis=1).rename('MaxLFCSamplePair')
    max_lfc_pair_per_tis = lfc_reshaped.dropna(how='all').idxmax(axis=1).apply(lambda x: '/'.join(x)).rename('MaxLFCSamplePair')
    min_diffq_per_tis = qval_reshaped.min(axis=1).rename('MinDiffQVal')
    min_diffq_pair_per_tis = qval_reshaped.dropna(how='all').idxmin(axis=1).apply(lambda x: '/'.join(x)).rename('MinDiffQValSamplePair')
    median_diffq_per_tis = qval_reshaped.median(axis=1).rename('MedianDiffQVal')
    max_diffq_per_tis = qval_reshaped.max(axis=1).rename('MaxDiffQVal')
    min_median_diffq_magnitude_diff_per_tis = (np.log10(min_diffq_per_tis) - np.log10(median_diffq_per_tis)).abs().rename('QValMinMedianMagnitudeDiff')

    # direction of outlier change
    directions = []
    for tis, min_idx in tqdm(qval_reshaped.dropna(how='all').idxmin(axis=1).items()):
        directions.append(np.sign(lfc_reshaped.loc[tis, min_idx]))
    most_sig_direction_per_tis = pd.Series(directions, index=qval_reshaped.dropna(how='all').index).rename('MinDiffQValDirection')

    # data on raw counts
    max_riboseq_counts_per_tis = tis_counts.max(axis=1).rename('MaxRiboseqCounts')
    max_riboseq_rpm_per_tis = ((tis_counts / total_tis_counts) * 1e6).max(axis=1).rename('MaxRiboseqRPM')
    max_riboseq_sample_per_tis = ((tis_counts / total_tis_counts) * 1e6).idxmax(axis=1).rename('MaxRiboseqSample')
    min_riboseq_counts_per_tis = tis_counts.min(axis=1).rename('MinRiboseqCounts')
    min_riboseq_rpm_per_tis = ((tis_counts / total_tis_counts) * 1e6).min(axis=1).rename('MinRiboseqRPM')
    min_riboseq_sample_per_tis = ((tis_counts / total_tis_counts) * 1e6).idxmin(axis=1).rename('MinRiboseqSample')

    # merge metrics together
    tis_stats = pd.concat([
        n_samples_per_tis, variance_per_tis,
        max_riboseq_counts_per_tis, max_riboseq_rpm_per_tis, max_riboseq_sample_per_tis, min_riboseq_counts_per_tis, min_riboseq_rpm_per_tis, min_riboseq_sample_per_tis,
        max_te_per_tis, max_te_sample_per_tis, min_te_per_tis, min_te_sample_per_tis,
        max_lfc_per_tis, max_lfc_pair_per_tis, min_lfc_per_tis, min_lfc_pair_per_tis, most_sig_direction_per_tis,
        min_diffq_per_tis, min_diffq_pair_per_tis, median_diffq_per_tis, max_diffq_per_tis, min_median_diffq_magnitude_diff_per_tis,
        min_delta_variance, min_delta_range, min_delta_variance_sample
    ], axis=1)
    tis_summary = tis_metadata_df.merge(tis_stats, left_on='TIS', right_index=True)
    tis_summary = tis_summary.dropna(subset=['NSamples'])

    return tis_summary

##### Vector comparison pipeline #####

# construct a dictionary mapping transcripts or genes to a matrix of sample-wise vectors, having shape [num samples x num TIS per transcript/gene]
def create_tis_vectors(te_matrix, tis_metadata_df, sample_subset=None, level='Gene'):
    if sample_subset is None:
        sample_subset = te_matrix.columns.tolist()
    else:
        sample_subset = [s for s in sample_subset if s in te_matrix.columns.tolist()]

    annotated_te_matrix = te_matrix.fillna(0).merge(tis_metadata_df, left_index=True, right_on='TIS', how='left')
    annotated_te_matrix['GenomeStart'] = annotated_te_matrix['TIS'].str.split('_').str[-1]
    annotated_te_matrix['ChromosomePosition'] = annotated_te_matrix['GenomeStart'].str.split(':').str[1].astype(int)
    annotated_te_matrix['Orientation'] = annotated_te_matrix['GenomeStart'].str.split(':').str[-1]

    group_to_tis_vector = dict()
    gene_groups = annotated_te_matrix.groupby('Symbol').groups

    for gene_id, gene_subset_idxs in tqdm(gene_groups.items()):
        gene_subset = annotated_te_matrix.loc[gene_subset_idxs, :]
        if gene_subset.iloc[0].loc['Orientation'] == '+': # forward, use an ascending sort on the chromosome position
            sort_ascending = True
        else:
            sort_ascending = False
        gene_subset_ordered = gene_subset.sort_values('ChromosomePosition', ascending=sort_ascending)

        if level.lower() in ['gene', 'symbol']:
            average_log_te_vector = gene_subset_ordered.groupby('GenomeStart')[sample_subset].mean()
            group_to_tis_vector[gene_id] = average_log_te_vector.T
        elif level.lower() == 'transcript':
            for transcript_id, transcript_subset_idxs in gene_subset_ordered.groupby('Tid').groups.items():
                transcript_subset = gene_subset_ordered.loc[transcript_subset_idxs, :]
                group_to_tis_vector[transcript_id] = transcript_subset.set_index('TIS').loc[:, sample_subset].T
    return group_to_tis_vector

# softmax on a log-space vector
def convert_log_to_probs(vector, base=2):
    unlogged_vector = base ** vector
    probs = unlogged_vector / unlogged_vector.sum()
    return probs

# calculate Jensen-Shannon divergence between a pair of probability distributions
def js_divergence(log_P, log_Q, base=2):
    """
    Jensen-Shannon divergence, a symmetric measure of the difference between probability distributions
    JSD(P || Q) = 0.5 * KLD(P || M) + 0.5 * KLD(Q || M)
    M = 0.5 + (P + Q)
    logM = log(0.5) + log(P + Q) = log(P + Q) - log(2)
    """
    log_P = log_P.copy()
    log_Q = log_Q.copy()

    if base != np.e:
        # convert to natural log
        log_P = log_P * np.log(base)
        log_Q = log_Q * np.log(base)

    log_pP = log_P - logsumexp(log_P)
    log_pQ = log_Q - logsumexp(log_Q)

    log_mixture = np.logaddexp(log_pP, log_pQ) - np.log(2)

    pP = np.exp(log_pP)
    pQ = np.exp(log_pQ)

    KL_PvM = np.sum(pP * (log_pP - log_mixture))
    KL_QvM = np.sum(pQ * (log_pQ - log_mixture))

    return 0.5 * (KL_PvM + KL_QvM)

# generate a matrix from a pairwise operation on a table of samples
def apply_pairwise(df, func, axis=0, symmetric=True):
    df_copy = df.copy()
    if axis == 0: # treats indices as samples
        df_copy = df_copy.T
    elif axis != 1: 
        raise ValueError('axis must be 0 or 1')
    elements = df_copy.columns.tolist()

    output = np.empty((len(elements), len(elements)))
    output.fill(np.nan)

    for i1 in range(len(elements)):
        if symmetric: # only compute the lower diagonal
            other_range = range(i1+1, len(elements))
        else:
            other_range = range(len(elements))
        for i2 in other_range:
            output[i1, i2] = func(df_copy[elements[i1]], df_copy[elements[i2]])
    output_df = pd.DataFrame(output, index=elements, columns=elements)
    return output_df

# convert an upper- or lower-triangular matrix to a symmetric matrix by copying values over the main diagonal
def symmetrize_matrix(mtx, diag_value=0):
    symm_mtx = np.maximum(mtx.fillna(0).values, mtx.T.fillna(0).values)
    np.fill_diagonal(symm_mtx, diag_value)
    return pd.DataFrame(symm_mtx, index=mtx.index, columns=mtx.columns)

# summarize pairwise statistics on vectors across samples
def summarize_vector_differences(group_to_vector_matrix, group_to_distance_matrix=None, group_type='Gene'):
    # compute disatnces between vectors if not provided
    if group_to_distance_matrix is None:
        group_to_distance_matrix = dict()
        for g in tqdm(group_to_vector_matrix):
            nonnull_matrix = group_to_vector_matrix[g].dropna(how='all').fillna(-np.inf)
            if nonnull_matrix.shape[0] > 1:
                group_to_distance_matrix[g] = apply_pairwise(
                    nonnull_matrix, js_divergence, axis=0, symmetric=True
                )

    # determine the appropriate labels based on the grouping type
    if group_type.lower() in ['gene', 'symbol']:
        column_prefix = 'Gene'
        index_name = 'Symbol'
    elif group_type.lower() == 'transcript':
        column_prefix = 'Transcript'
        index_name = 'Tid'
    
    # construct a dictionary to collect values
    group_vector_summary = {
        index_name: [],
        f'{column_prefix}MeanPairwiseTEVectorDifference': [],
        f'{column_prefix}NumUniqueTIS': [],
        f'{column_prefix}NumSamplesWithAllTIS': [],
        f'{column_prefix}NumSamplesWithHalfTIS': [],
        f'{column_prefix}NumSamplesWithAnyTIS': [],
    }

    # iterate through groups and add appropriate statistics
    for g in tqdm(group_to_distance_matrix):
        group_vector_summary[index_name].append(g)
        group_vector_summary[f'{column_prefix}MeanPairwiseTEVectorDifference'].append(np.nanmean(group_to_distance_matrix[g].values.flatten()))
        group_vector_summary[f'{column_prefix}NumUniqueTIS'].append(group_to_vector_matrix[g].shape[1])
        group_vector_summary[f'{column_prefix}NumSamplesWithAllTIS'].append(((~(group_to_vector_matrix[g].isna())).sum(axis=1) == group_to_vector_matrix[g].shape[1]).sum())
        group_vector_summary[f'{column_prefix}NumSamplesWithHalfTIS'].append(((~(group_to_vector_matrix[g].isna())).sum(axis=1) >= (group_to_vector_matrix[g].shape[1] / 2)).sum())
        group_vector_summary[f'{column_prefix}NumSamplesWithAnyTIS'].append(group_to_distance_matrix[g].shape[0])
    group_vector_summary = pd.DataFrame.from_dict(group_vector_summary).sort_values(
        by=[f'{column_prefix}NumSamplesWithHalfTIS', f'{column_prefix}NumUniqueTIS', f'{column_prefix}MeanPairwiseTEVectorDifference'], ascending=False
    ).set_index(index_name)

    return group_to_distance_matrix, group_vector_summary

# construct an annotation table per transcript based on individual TISs mapped to the transcript
def create_tis_annotations(group_to_tis_vector, tis_metadata_df, group_type='Gene'):
    if group_type.lower() in ['gene', 'symbol']:
        index_name = 'Symbol'
        tis_id = 'GenomeStart'
    elif group_type.lower() == 'transcript':
        index_name = 'Tid'
        tis_id = 'TIS'
    tis_type_annotations = []
    tis_structure_annotations = {index_name: [], 'StartCodons': [], 'InterTISDistances': [], 'UTRLengths': []}
    for g in tqdm(group_to_tis_vector):
        if group_type.lower() in ['gene', 'symbol']:
            id_list = [':'.join(x.split(':')[:-1]) for x in group_to_tis_vector[g].columns]
        elif group_type.lower() == 'transcript':
            id_list = group_to_tis_vector[g].columns

        tis_information = tis_metadata_df[tis_metadata_df[tis_id].isin(id_list)]

        if tis_information['TIS'].iloc[0][-1] == '+':
            forward = True
            sign_flip = 1
        else:
            forward = False
            sign_flip = -1
        
        # summaries of TIS classes
        tis_type_annotations.append(
            tis_information.groupby('RecatTISType')[tis_id].nunique().reindex(index=['Annotated', 'Extended', 'Truncated', 'uORF']).fillna(0).astype(int).rename(g)
        )

        # structural summaries of TISs (codon identities, relative spacing, 5'UTR lengths)
        tis_structure_annotations[index_name].append(g)
        sorted_tis_information = tis_information.drop_duplicates(subset=[tis_id]).sort_values('GenomeStart', ascending=forward)
        tis_structure_annotations['StartCodons'].append(
            '|'.join(sorted_tis_information['StartCodon'].tolist())
        )
        tis_structure_annotations['InterTISDistances'].append(
            '|'.join( # extract locus, convert to int for differencing, convert back to string, concat
                (sorted_tis_information['GenomeStart'].str.split(':').str[-1].astype(int).diff().dropna() * sign_flip).astype(int).astype(str).tolist() 
            )
        )
        tis_structure_annotations['UTRLengths'].append(
            '|'.join(sorted_tis_information['Start'].astype(int).astype(str).tolist())
        )
    tis_type_annotations = pd.concat(tis_type_annotations, axis=1).T
    tis_type_annotations = tis_type_annotations.add_prefix('Num').add_suffix('Sites').reset_index(names=[index_name]).set_index(index_name)

    tis_structure_annotations = pd.DataFrame.from_dict(tis_structure_annotations).set_index(index_name)
    
    tis_combined_annotations = tis_structure_annotations.merge(tis_type_annotations, left_index=True, right_index=True, how='outer')
    return tis_combined_annotations


##### Vector comparison I/O #####

# convert a dictionary of transcript/gene -> matrix into a table where variable-length vectors are encoded by string separators
def encode_tis_vector_df(tis_vector_dict):
    global_table = []
    for group_label, subset_to_encode in tqdm(tis_vector_dict.items()):
        tis_id_string = '|'.join(subset_to_encode.columns.tolist())
        value_strings = subset_to_encode.apply(lambda x: '|'.join(x.astype(str)), axis=1).rename('Values')
        nonzero_values = (subset_to_encode != 0).sum(axis=1).rename('NumNonzeroValues')
        n_values = subset_to_encode.shape[1]

        encoded_table = pd.concat([value_strings, nonzero_values], axis=1).assign(TISString=tis_id_string, NumTIS=n_values, GroupID=group_label).reset_index(names=['Sample']).loc[
            :, ['GroupID', 'NumTIS', 'TISString', 'Sample', 'NumNonzeroValues', 'Values']
        ]
        global_table.append(encoded_table)
    full_table = pd.concat(global_table, axis=0, ignore_index=True)

    return full_table

# convert a table of encoded vectors to a dictionary mapping transcripts/genes -> matrix
def parse_tis_table(encoded_tis_table, mask_zeroes=False, group_labels=None):
    if group_labels is None:
        group_labels = encoded_tis_table['GroupID'].unique().tolist()
    groups_to_idxs = encoded_tis_table.groupby('GroupID').groups
    group_labels = list(set(group_labels).intersection(groups_to_idxs.keys()))

    group_to_vector_matrix = dict()
    for group in tqdm(group_labels):
        group_idxs = groups_to_idxs[group]
        group_subset = encoded_tis_table.loc[group_idxs, :]
        sample_names = group_subset['Sample'].tolist()
        column_names = group_subset['TISString'].iloc[0].split('|')
        value_matrix = group_subset['Values'].str.split('|', expand=True)
        value_matrix = value_matrix.astype(float)
        value_matrix.index = sample_names
        value_matrix.columns = column_names
        if mask_zeroes:
            value_matrix[value_matrix == 0] = np.nan
        group_to_vector_matrix[group] = value_matrix
    
    return group_to_vector_matrix


##### Model fitting on transcript-level data

# construct a set of metadata representing all TIS identified across any sample
def define_global_tis_reference(longform_tis_table, masked_logte_matrix, sample_subset=None):
    # define a subsample if provided
    if sample_subset is None: 
        sample_subset = masked_logte_matrix.columns.tolist()
    tis_with_sufficient_support = masked_logte_matrix[sample_subset].dropna(how='all').index.tolist() # TE defined in any line
    
    # preserve only TISs with sufficient evidence
    filtered_longform_tis_table = longform_tis_table[
        longform_tis_table['Sample'].isin(sample_subset) & 
        longform_tis_table['TIS'].isin(tis_with_sufficient_support)
    ].copy()
    filtered_longform_tis_table['GenomeStart'] = filtered_longform_tis_table['TIS'].str.split('_').str[-1]
    filtered_longform_tis_table['ChromosomePosition'] = filtered_longform_tis_table['GenomeStart'].str.split(':').str[1].astype(int)
    filtered_longform_tis_table['Orientation'] = filtered_longform_tis_table['GenomeStart'].str.split(':').str[-1]

    # iterate through transcripts and encode variable-length information about TISs using strings with separators
    transcript_to_tis_vector_information = []
    tis_groups = filtered_longform_tis_table.groupby('Tid').groups
    for tid, tid_idxs in tqdm(tis_groups.items()):
        tid_subset = filtered_longform_tis_table.loc[tid_idxs, :]
        if tid_subset.iloc[0].loc['Orientation'] == '+': # forward, use an ascending sort on the chromosome position
            sort_ascending = True
        else:
            sort_ascending = False
        tid_subset_ordered = tid_subset.sort_values('ChromosomePosition', ascending=sort_ascending)
        ordered_unique_tis_subset = tid_subset_ordered.drop_duplicates(subset=['GenomeStart'])
        codon_string = '|'.join([s for s in ordered_unique_tis_subset['StartCodon'].tolist()])
        utr_length_string = '|'.join([str(int(s)) for s in ordered_unique_tis_subset['Start'].tolist()])
        tistype_string = '|'.join([s for s in ordered_unique_tis_subset['RecatTISType'].tolist()])
        transcript_to_tis_vector_information.append([tid, utr_length_string, codon_string, tistype_string])
    transcript_to_tis_vector_information = pd.DataFrame(transcript_to_tis_vector_information, columns=['Tid', 'UTRLengths', 'StartCodons', 'TISTypes'])
    return transcript_to_tis_vector_information

# pull all transcript sequences from the reference assembly
def read_transcript_sequences(transcript_fasta='/lab/barcheese01/smaffa/coTISja/data/reference/gencode.v49.pc_transcripts.fa'):
    gencode_transcript_sequences = dict()
    for record in SeqIO.parse(transcript_fasta, format='fasta'):
        sequence_ids = record.id
        transcript_id = [tag for tag in sequence_ids.split('|') if 'ENST' in tag][0]
        gencode_transcript_sequences[transcript_id] = str(record.seq)
    gencode_transcript_sequences = pd.DataFrame(pd.Series(gencode_transcript_sequences), columns=['Seq']).reset_index(names=['Tid'])
    gencode_transcript_sequences['Len'] = gencode_transcript_sequences['Seq'].apply(lambda x: len(x))
    return gencode_transcript_sequences

# calculate hamming distance while allowing for IUPAC ambiguity codes
def hamming_distance_ambiguous(s1, s2, weights=None):
    """Calculate Hamming distance between two sequences with ambiguous bases."""
    if len(s1) != len(s2):
        raise ValueError("Sequences must be of equal length")
    
    if weights is None:
        weights = [1] * len(s1)

    distance = 0
    position_i = 0
    for a, b in zip(s1, s2):
        set_a = ambiguity_dict.get(a, {a})
        set_b = ambiguity_dict.get(b, {b})
        if not set_a.intersection(set_b):  # no overlap → mismatch
            distance += weights[position_i]
        position_i += 1
    return distance

# extract Kozak context sequences for canonical sites and assemble statistics for thei similarity to the Kozak consensus sequence
def assign_canonical_kozak_annotations(tis_vector_annotations, transcript_sequences):
    annotated_tis_vector_metadata = tis_vector_annotations.merge(transcript_sequences, left_on='Tid', right_on='Tid', how='left')

    canonical_subset = annotated_tis_vector_metadata[
        annotated_tis_vector_metadata['TISTypes'].str.contains('Annotated')
    ].copy()
    canonical_subset['CanonicalIndex'] = canonical_subset['TISTypes'].apply(lambda x: x.split('|').index('Annotated'))
    canonical_subset['CanonicalStart'] = canonical_subset.apply(lambda x: int(x['UTRLengths'].split('|')[int(x['CanonicalIndex'])]), axis=1)
    canonical_subset['CanonicalKozakContext'] = canonical_subset.apply(
        lambda x: x['Seq'][(int(x['CanonicalStart']) - 9):(int(x['CanonicalStart']) + 4)], axis=1
    )
    canonical_subset['GCContentCanonicalUTR'] = canonical_subset.apply(
        lambda x: SeqUtils.gc_fraction(x['Seq'][:(int(x['CanonicalStart']))]), axis=1
    )

    annotated_tis_vector_metadata = annotated_tis_vector_metadata.merge(canonical_subset, how='left')
    hamming_subset = annotated_tis_vector_metadata[
        annotated_tis_vector_metadata['CanonicalKozakContext'].apply(lambda x: len(x) == 13 if isinstance(x, str) else False)
    ].copy()
    hamming_subset['KozakHammingDistance'] = hamming_subset.apply(
        lambda x: hamming_distance_ambiguous(kozak_pattern, x['CanonicalKozakContext'], weights=None), axis=1
    )
    hamming_subset['KozakMajorHammingDistance'] = hamming_subset.apply(
        lambda x: hamming_distance_ambiguous(kozak_pattern, x['CanonicalKozakContext'], weights=major_weights), axis=1
    )

    annotated_tis_vector_metadata = annotated_tis_vector_metadata.merge(hamming_subset, how='left')
    return annotated_tis_vector_metadata

# Create the input features for model fitting, compatible with a formula object
def featurize_by_canonical_sites(tid_summary_metadata, tis_df, 
                                 sample_id_columns=['Sample', 'Replicate'], 
                                 response_variable_columns=['TISCounts', 'GeneRNASeqCounts', 'GeneRNASeqLogRPM'], 
                                 extra_transcript_columns=['MANE_Select']):
    
    # extract features from the annotation metadata summary
    canonical_subset = tid_summary_metadata[
        tid_summary_metadata['TISTypes'].str.contains('Annotated')
    ].dropna()

    predictors_table = pd.concat([
        canonical_subset['Tid'],
        canonical_subset.apply(lambda x: len(x['StartCodons'].split('|')[:int(x['CanonicalIndex'])]), axis=1).rename('NumUpstreamTIS'),
    ] + [
        canonical_subset.apply(lambda x: len([y for y in x['StartCodons'].split('|')[:int(x['CanonicalIndex'])] if y == codon]), axis=1).rename(f'NumUpstream{codon}') for codon in codon_order
    ] + [
        canonical_subset.apply(lambda x: len([y for y in x['StartCodons'].split('|')[:int(x['CanonicalIndex'])] if y == tistype]), axis=1).rename(f'NumUpstream{tistype}') for tistype in ['uORF', 'Extension', 'Other']
    ] + [
        canonical_subset['UTRLengths'].apply(lambda x: int(x.split('|')[0])).rename('FirstUTRLength'),
        canonical_subset.apply(lambda x: int(x['UTRLengths'].split('|')[int(x['CanonicalIndex'])]), axis=1).rename('CanonicalUTRLength'),
        canonical_subset['GCContentCanonicalUTR'],
        canonical_subset['KozakHammingDistance'],
        canonical_subset['KozakMajorHammingDistance']
    ], axis=1)

    # extract observations from the longform tis df
    observations_table = tis_df[
        tis_df['Tid'].isin(canonical_subset['Tid']) &
        (tis_df['RecatTISType'] == 'Annotated')
    ][['Tid'] + sample_id_columns + response_variable_columns + extra_transcript_columns]

    feature_table = observations_table.merge(predictors_table, left_on='Tid', right_on='Tid', how='outer')
    return feature_table

# Format an input data table for statsmodels such that a formula is not a necessary input (unused in the current pipeline)
def create_model_inputs(feature_df, formula_string):
    from pandas.api.types import is_string_dtype

    exog_var = formula_string.split('~')[0].strip()
    endog_var_list = [term.strip() for term in formula_string.split('~')[-1].split('+')]
    noninteraction_terms = [term for term in endog_var_list if ':' not in term]
    interaction_terms = [(combo.split(':')[0].strip(), combo.split(':')[1].strip()) for combo in endog_var_list if ':' in combo]
    products = {f'{s[0]}__x__{s[1]}': s for s in interaction_terms}
    all_columns = list(set(noninteraction_terms + [s[0] for s in interaction_terms] + [s[1] for s in interaction_terms]))
    categorical_columns = [col for col in all_columns if is_string_dtype(feature_df[col])]
    categorical_to_categories = {col: sorted(feature_df[col].unique().tolist()) for col in categorical_columns}
    categorical_to_reference_value = {col: cats[0] for col, cats in categorical_to_categories.items()}
    categorical_to_other_values = {col: cats[1:] for col, cats in categorical_to_categories.items()}

    if len(categorical_columns) > 0:
        categorical_encodings = pd.concat([
            pd.get_dummies(feature_df[col]).drop([categorical_to_reference_value[col]], axis=1).rename(
                {cat: f'{col}[T.{cat}]' for cat in categorical_to_other_values[col]}, axis=1
            ) for col in categorical_columns
        ], axis=1)
    else:
        categorical_encodings = pd.DataFrame()

    interaction_encodings = []
    for prod, pair in products.items():
        s1, s2 = pair
        if s1 in categorical_columns and s2 in categorical_columns:
            for cat1 in categorical_to_other_values[s1]:
                for cat2 in categorical_to_other_values[s2]:
                    interaction_encodings.append((categorical_encodings[f'{s1}[T.{cat1}]'] * categorical_encodings[f'{s2}[{cat2}]']).rename(f'{s1}[T.{cat1}]__x__{s2}[T.{cat2}]'))
        elif s1 in categorical_columns:
            for cat in categorical_to_other_values[s1]:
                interaction_encodings.append((categorical_encodings[f'{s1}[T.{cat}]'] * feature_df[s2]).rename(f'{s1}[T.{cat}]__x__{s2}'))
        elif s2 in categorical_columns:
            for cat in categorical_to_other_values[s2]:
                interaction_encodings.append((feature_df[s1] * categorical_encodings[f'{s2}[T.{cat}]']).rename(f'{s1}__x__{s2}[T.{cat}]'))
        else:
            interaction_encodings.append((feature_df[s1] * feature_df[s2]).rename(f'{s1}__x__{s2}'))
    interaction_encodings = pd.concat(interaction_encodings, axis=1)

    endog_df = pd.concat([
        categorical_encodings
    ] + [
        feature_df[col] for col in noninteraction_terms if col not in categorical_columns
    ] + [
        interaction_encodings
    ], axis=1)
    exog_df = feature_df[exog_var]

    return endog_df.astype(float), exog_df.astype(float)

# extract fold changes with confidence intervals from a fitted GLM
def compute_fold_changes(result, factor_names, cell_lines, baseline_cell_line):
    """
    Computes fold-change and 95% CI for specified factors per cell line.
    
    Parameters:
        result: fitted GLM NB (with cov_type='HC0')
        factor_names: list of main factors to interpret
        cell_lines: list of all samples
        baseline_cell_line: the reference sample
    Returns:
        pandas DataFrame with β, fold-change, 95% CI per factor per cell line
    """
    cov = result.cov_params()
    params = result.params
    
    records = []

    for factor in factor_names:
        # main effect
        beta_main = params.get(factor, 0.0)
        
        for cell_line in cell_lines:
            # start with main effect
            beta_eff = beta_main
            
            if cell_line != baseline_cell_line:
                # check for interaction term
                interaction_term = f"{factor}:Sample[T.{cell_line}]"
                beta_int = params.get(interaction_term, 0.0)
                beta_eff += beta_int
                
                # variance propagation
                sub_cov = cov.reindex(index=[factor, interaction_term], columns=[factor, interaction_term], fill_value=0.0)

                var_eff = sub_cov.loc[factor, factor] + sub_cov.loc[interaction_term, interaction_term] + 2 * sub_cov.loc[factor, interaction_term]
            else:
                # baseline cell line variance
                var_eff = cov.loc[factor, factor]
            
            se_eff = np.sqrt(var_eff)
            # 95% CI in log space
            ci_low = beta_eff - 1.96 * se_eff
            ci_high = beta_eff + 1.96 * se_eff
            
            # convert to fold-change
            fold = np.exp(beta_eff)
            fold_low = np.exp(ci_low)
            fold_high = np.exp(ci_high)
            
            records.append({
                'sample': cell_line,
                'factor': factor,
                'beta': beta_eff,
                'fold_change': fold,
                'fold_change_CI_low': fold_low,
                'fold_change_CI_high': fold_high
            })
    
    return pd.DataFrame.from_records(records)

# save outputs of a statsmodels GLM
def write_glm_results(model, output_dir, model_prefix):
    summary_file = os.path.join(output_dir, f'{model_prefix}_model_summary.csv')
    covariance_file = os.path.join(output_dir, f'{model_prefix}_covariance_matrix.csv')
    with open(summary_file, "w") as f:
        f.write(model.summary().as_csv())
    model.cov_params().to_csv(covariance_file, index=True)