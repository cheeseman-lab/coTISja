import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm


### TIS variance across samples ###

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

### Vector comparison pipeline ###

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


def convert_log_to_probs(vector, base=2):
    unlogged_vector = base ** vector
    probs = unlogged_vector / unlogged_vector.sum()
    return probs


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

def apply_pairwise(df, func, axis=0, symmetric=True):
    df_copy = df.copy()
    if axis == 0:
        df_copy = df_copy.T
    elif axis != 1:
        raise ValueError('axis must be 0 or 1')
    elements = df_copy.columns.tolist()

    output = np.empty((len(elements), len(elements)))
    output.fill(np.nan)

    for i1 in range(len(elements)):
        if symmetric:
            other_range = range(i1+1, len(elements))
        else:
            other_range = range(len(elements))
        for i2 in other_range:
            output[i1, i2] = func(df_copy[elements[i1]], df_copy[elements[i2]])
    output_df = pd.DataFrame(output, index=elements, columns=elements)
    return output_df

def symmetrize_matrix(mtx, diag_value=0):
    symm_mtx = np.maximum(mtx.fillna(0).values, mtx.T.fillna(0).values)
    np.fill_diagonal(symm_mtx, diag_value)
    return pd.DataFrame(symm_mtx, index=mtx.index, columns=mtx.columns)


def summarize_vector_differences(group_to_vector_matrix, group_to_distance_matrix=None, group_type='Gene'):
    if group_to_distance_matrix is None:
        group_to_distance_matrix = dict()
        for g in tqdm(group_to_vector_matrix):
            nonnull_matrix = group_to_vector_matrix[g].dropna(how='all').fillna(-np.inf)
            if nonnull_matrix.shape[0] > 1:
                group_to_distance_matrix[g] = apply_pairwise(
                    nonnull_matrix, js_divergence, axis=0, symmetric=True
                )

    if group_type.lower() in ['gene', 'symbol']:
        column_prefix = 'Gene'
        index_name = 'Symbol'
    elif group_type.lower() == 'transcript':
        column_prefix = 'Transcript'
        index_name = 'Tid'
    
    group_vector_summary = {
        index_name: [],
        f'{column_prefix}MeanPairwiseTEVectorDifference': [],
        f'{column_prefix}NumUniqueTIS': [],
        f'{column_prefix}NumSamplesWithAllTIS': [],
        f'{column_prefix}NumSamplesWithHalfTIS': [],
        f'{column_prefix}NumSamplesWithAnyTIS': [],
    }

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


### Vector comparison I/O ###

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