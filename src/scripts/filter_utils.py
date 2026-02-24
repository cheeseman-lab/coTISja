# standard data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

# convenience
from tqdm import tqdm

GTF_FILE = '/lab/barcheese01/aTIS_data/ribotish/gencode.v49.primary_assembly.annotation.gtf'
GENOME_FILE = '/lab/barcheese01/smaffa/coTISja/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa'
PROTEIN_FASTA = '/lab/barcheese01/smaffa/coTISja/data/reference/gencode.v49.pc_translations.fa'
REPLICATE_MANIFEST_FILE = '/lab/barcheese01/smaffa/coTISja/data/ribotish_replicate_manifest.csv'
SAMPLE_MANIFEST_FILE = '/lab/barcheese01/smaffa/coTISja/data/ribotish_sample_manifest.csv'

# Adapted from swissisoform.genome without the GenomeHandler object
def load_transcript_annotations(gtf_path: str, nrows=None, feature_type='transcript') -> pd.DataFrame:
    """Load and parse GTF annotations into a pandas DataFrame.

    Args:
        gtf_path (str): Path to the genome annotation GTF file.

    Returns:
        annotations (pd.DataFrame)
    """
    features_list = []
    if nrows is None:
        nrows = np.inf
    counter = 0

    with open(gtf_path) as handle:
        for line in handle:
            counter += 1

            # skip the header
            if line.startswith("#"):
                continue

            # stop parsing if nrows exceeded
            if counter > nrows:
                break

            # only look at transcript annotations
            if f'\t{feature_type}\t' in line:
                fields = line.strip().split("\t")

                if len(fields) != 9:
                    continue

                # leave attributes as a sting to parse
                attribute_string = fields[8]

                feature_dict = {
                    "chromosome": fields[0],
                    "source": fields[1],
                    "feature_type": fields[2],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "score": fields[5],
                    "strand": fields[6],
                    "frame": fields[7],
                    "attributes": fields[8],
                }

                features_list.append(feature_dict)
    
    annotations = pd.DataFrame(features_list)

    base_columns = ['chromosome', 'source', 'feature_type', 'start', 'end', 'score', 'strand', 'frame']
    extracted_columns = ['gene_id', 'gene_type', 'transcript_id', 'transcript_type', 'transcript_support_level']
    tags = ['MANE_Select']

    for col in extracted_columns:
        annotations[col] = annotations['attributes'].str.extract(f'{col} "([^"]*)"')
    for tag in tags:
        annotations[tag] = annotations['attributes'].str.contains(tag)

    annotations_sorted = annotations[base_columns + extracted_columns + tags + ['attributes']]

    return annotations_sorted

def annotate_tis_locus(tis_df, genome_position_column='GenomePos'):
    """
    Extract key components (chromosome, strand, start locus relative to strand) from a genome position string
    
    :param tis_df: Input dataframe
    :param genome_position_column: string name of the genome position column
    """
    tis_df = tis_df.copy()
    tis_df['Chromosome'] = tis_df['GenomePos'].str.split(':').str[0]
    tis_df['Strand'] = tis_df['GenomePos'].str.get(-1)
    forward_mask = tis_df['Strand'] == '+'
    reverse_mask = tis_df['Strand'] == '-'
    tis_df.loc[forward_mask, 'Locus'] = tis_df.loc[forward_mask, 'GenomePos'].str.split(':').str[1].str.split('-').str[0]
    tis_df.loc[reverse_mask, 'Locus'] = tis_df.loc[reverse_mask, 'GenomePos'].str.split(':').str[1].str.split('-').str[1]
    return tis_df

# Load a csv with ribotish results
def import_ribotish_results(filepath, gtf_df=None):
    """
    Load a table of ribotish outputs and merge it with an annotation file
    
    :param filepath: table of outputs from `ribotish predict`
    :param gtf_df: an output df from load_transcript_annotations()
    """
    tis_df = pd.read_csv(filepath, sep='\t')

    tis_df = annotate_tis_locus(tis_df)

    if gtf_df is not None:
        tis_columns = tis_df.columns.tolist()

        tis_df = tis_df.merge(
            gtf_df, how='left', left_on=['Gid', 'Tid'], right_on=['gene_id', 'transcript_id']
        )

        # simplify df to relevant columns
        gtf_columns = ['gene_type', 'transcript_type', 'source', 'start', 'end', 'MANE_Select', 'transcript_support_level', 'attributes']
        gtf_rename_dict = {'start': 'txn_start', 'end': 'txn_end'}

        tis_df = tis_df[tis_columns + gtf_columns].rename(gtf_rename_dict, axis=1)
    
    return tis_df

def recategorize_tis_type(tis_df, original_column='TisType', output_column='RecatTISType'):
    """
    Reassign labels for TIS type
    
    :param tis_df: Description
    :param original_column: Description
    :param output_column: Description
    """
    tis_df = tis_df.copy()
    tis_df[output_column] = tis_df[original_column].apply(lambda x: 
        'Annotated' if 'Annotated' in x else
        'Truncated' if 'Truncated' in x else
        'uORF' if "5'UTR" in x else
        'Extended' if 'Extended' in x else
        'Other'
    )
    return tis_df

def load_experiment_manifest(
    sample_file=SAMPLE_MANIFEST_FILE, 
    replicate_file=REPLICATE_MANIFEST_FILE
):
    replicate_df = pd.read_csv(replicate_file)
    sample_df = pd.read_csv(sample_file)
    experiment_table = sample_df.merge(
        replicate_df[
            replicate_df['condition'] == 'TIS'
        ].groupby('sample').apply(lambda x: list(x['rnaseq_count_file'])).rename('rnaseq_count_file'), left_on='sample', right_index=True
    )
    return experiment_table, sample_df, replicate_df

def calculate_read_totals_from_bam_qc(
    bam_qc_filepath, 
    # offset_filepath=None
):
    """
    Extract mapped readcount totals from corresponding BAM QC outputs from `ribotish quality`
    
    :param bam_qc_filepath: Description
    """
    with open(bam_qc_filepath, 'r') as f:
        line = f.readline()
    read_totals_per_length = {int(x.split(': ')[0]): int(x.split(': ')[1]) for x in line.strip().strip('{|}').split(', ')}

    # I think this is wrong - should be using all of the reads, not just the length-wise groups whose P-site offset has its default value overwritten

    # offset_dict = dict()
    # if offset_filepath is not None:
    #     with open(offset_filepath, 'r') as f:
    #         line = f.readline()
    
    #     match = re.search('\{(.+)\}', line)
    #     if match:
    #         offset_string = match.groups()[0]
    #         offset_dict = dict()
    #         for keyval in offset_string.split(', '):
    #             key, val = keyval.split(': ')
    #             if 'm0' in key:
    #                 offset_dict[key.strip("'")] = val
    #             else:
    #                 offset_dict[int(key)] = int(val)
        
    # return np.sum([c for l, c in read_totals_per_length.items() if l in list(offset_dict.keys())])

    return np.sum([c for c in read_totals_per_length.values()])

def read_rnaseq_counts(
    rnaseq_qc_filepath,
):
    """
    Extract mapped readcount totals from corresponding RNA-seq mapping results
    
    :param rnaseq_qc_filepath: Description
    """
    gene_counts = pd.read_csv(rnaseq_qc_filepath, sep='\t', header=None, names=['Gid', 'counts'])
    gene_counts = gene_counts.set_index('Gid')['counts']
    qc_indices = ['__no_feature', '__ambiguous', '__too_low_aQual', '__not_aligned', '__alignment_not_unique']
    qc_counts = gene_counts.loc[gene_counts.index.isin(qc_indices)]
    gene_counts = gene_counts.loc[~gene_counts.index.isin(qc_indices)]
    return gene_counts

def normalize_tis_counts(tis_df, bam_qc_files=None, total=None, divisor_column=None, scale_factor=1e6, count_col = 'TISCounts', output_col='NormTISCounts'):
    """
    Normalize TIS counts in a ribotish result table, using an RPM transformation by default
    
    :param tis_df: Description
    :param bam_qc_files: Description
    :param total: Description
    :param scale_factor: Description
    :param count_col: Description
    :param output_col: Description
    """
    tis_df = tis_df.copy()
    if divisor_column is not None: # divide by another column
        tis_df[output_col] = tis_df[count_col] / tis_df[divisor_column]
        return tis_df
    elif bam_qc_files is not None: # divide by mapped reads (read depth)
        if not isinstance(bam_qc_files, list):
            bam_qc_files = [bam_qc_files]
        total = 0
        for bam_qc_file in bam_qc_files:
            total += calculate_read_totals_from_bam_qc(bam_qc_file)
    if total is None: # nothing external provided to normalize
        total = tis_df.drop_duplicates(['Chromosome', 'Locus', 'Strand'])[count_col].sum()
    # if total is provided manually, use this as the denominator
    tis_df[output_col] = (tis_df[count_col] / total) * scale_factor
    return tis_df
    
# Define a set of reference transcripts to map start sites to
def identify_reference_transcripts(
    tis_df, 
    transcript_support_levels=['1','2','3'], 
    min_tis_counts=None,
    min_percentile_tis_counts=None,
    count_col='TISCounts',
    tis_enrichment_max_p=0.01,
    frame_test_max_p=0.01,
    combined_test_max_q=0.05
):
    """
    Determine a set of transcripts from which TISs will be selected:
        1) Filter by data-independent annotations: MANE_Select and TSL
        2) Filter by readcounts from any TIS mapped to that transcript (default is minimum readcounts of 0, aka all TISs)
        3) Filter by ribotish significance of any TIS mapped to that transcript (default is original strategy for *filtered_predictor files)
    Return all transcript IDs for any TIS remaining
    
    :param tis_df: Description
    :param transcript_support_levels: Description
    :param min_tis_counts: Description
    :param min_percentile_tis_counts: Description
    :param count_col: Description
    :param tis_enrichment_max_p: Description
    :param frame_test_max_p: Description
    :param combined_test_max_q: Description
    """
    # potential filters on readcounts
    if min_percentile_tis_counts is not None:
        min_tis_counts = np.percentile(tis_df[count_col], min_percentile_tis_counts)
    if min_tis_counts is None:
        min_tis_counts = 0

    print(f'Selecting transcripts to use from {tis_df.shape[0]} mapped TISs...')

    # MANE annotations
    mane_mask = tis_df['MANE_Select'] == True
    print(f'Keeping {mane_mask.sum()} TISs from MANE_Select transcripts...')

    # transcript support level annotations
    tsl_mask = (tis_df['transcript_support_level'].isin(transcript_support_levels))
    print(f'Keeping {tsl_mask.sum()} remaining TISs from transcripts with support level {"/".join(transcript_support_levels)}...')
    
    # combine annotated totals
    tis_to_keep = tis_df[mane_mask | tsl_mask]
    print(f'Total of {tis_to_keep.shape[0]} TISs are mapped to transcripts with high-quality evidence...')

    # filter for readcount support
    support_mask = tis_to_keep[count_col] >= min_tis_counts
    print(f'Keeping {support_mask.sum()} TISs with readcount support of {count_col} >= {min_tis_counts}...')
    tis_to_keep = tis_to_keep[support_mask]

    # filter for significance
    enrichment_mask = tis_to_keep['TISPvalue'] <= tis_enrichment_max_p
    frame_mask = tis_to_keep['RiboPvalue'] <= frame_test_max_p
    combined_q_mask = tis_to_keep['FisherQvalue'] <= combined_test_max_q
    significance_mask = enrichment_mask & frame_mask & combined_q_mask
    print(f'Keeping {significance_mask.sum()} TISs meeting significance of (TISPvalue <= {tis_enrichment_max_p}) & (RiboPvalue <= {frame_test_max_p}) & (FisherQvalue <= {combined_test_max_q})...')
    tis_to_keep = tis_to_keep[significance_mask]

    tids_to_keep = tis_to_keep['Tid'].unique().tolist()
    print(f'Keeping a total of {len(tids_to_keep)} unique transcript IDs represented in this set of TISs...')
    
    return tids_to_keep

def append_tag(series, tag, target_indices=None, target_mask=None, separator='|'):
    """
    Helper function to append a string to an annotation column
    
    :param series: Description
    :param tag: Description
    :param target_indices: Description
    :param target_mask: Description
    :param separator: Description
    """
    series = series.copy()
    if target_mask is None:
        if target_indices is not None:
            target_mask = series.index.isin(target_indices)
        else:
            target_mask = pd.Series(True, index=series.index)
    null_target_mask = target_mask & series.isnull()
    string_target_mask = target_mask & ~series.isnull()
    series.loc[null_target_mask] = tag # set the null values to the intended tag
    series.loc[string_target_mask] = series.loc[string_target_mask] + separator + tag # append the intended tag to the existing string values
    return series


# Filter TISs based on a reference set of transcripts, using a exclusion window from iteratively kept transcripts
def filter_ribotish_results(
    tis_df, 
    transcript_support_levels=['1','2','3'], 
    reference_min_tis_counts=None,
    reference_min_percentile_tis_counts=None,
    reference_count_col='TISCounts',
    reference_tis_enrichment_max_p=1,
    reference_frame_test_max_p=1,
    reference_combined_test_max_q=1,
    min_putative_tis_counts=0.25,
    count_col='NormTISCounts',
    tis_enrichment_max_p=0.01,
    frame_test_max_p=0.01,
    combined_test_max_q=0.05,
    tis_distance_buffer=30,
    return_dropped=False
):
    """
    Filter to a set of TISs for downstream analysis:
        1) Select a set of transcripts to use (see identify_reference_transcripts())
        2) Filter TISs by readcounts (default is minimum of 0.25 RPM)
        3) Filter TISs by significance (default is original strategy for *filtered_predictor files)
        4) Keep the annotated start site for all transcripts remaining
        5) For each transcript, iteratively select the highest readcount TIS, then exclude all other nearby TISs downstream of it
    
    :param tis_df: Description
    :param transcript_support_levels: Description
    :param reference_min_tis_counts: Description
    :param reference_min_percentile_tis_counts: Description
    :param reference_count_col: Description
    :param reference_tis_enrichment_max_p: Description
    :param reference_frame_test_max_p: Description
    :param reference_combined_test_max_q: Description
    :param min_putative_tis_counts: Description
    :param count_col: Description
    :param tis_enrichment_max_p: Description
    :param frame_test_max_p: Description
    :param combined_test_max_q: Description
    :param tis_distance_buffer: Description
    :param return_dropped: Description
    """
    reference_tids = identify_reference_transcripts(
        tis_df, 
        transcript_support_levels=transcript_support_levels,
        min_tis_counts=reference_min_tis_counts,
        min_percentile_tis_counts=reference_min_percentile_tis_counts,
        count_col=reference_count_col,
        tis_enrichment_max_p=reference_tis_enrichment_max_p,
        frame_test_max_p=reference_frame_test_max_p,
        combined_test_max_q=reference_combined_test_max_q,
    )

    tis_df = tis_df.copy()
    tis_df['GenomeStart'] = tis_df['GenomePos'].apply(
        lambda x: f'{x.split("-")[0]}' if (x.split(":")[-1] == '+') else f'{x.split(":")[0]}:{x.split(":")[1].split("-")[1]}'
    )
    tis_df['DropReason'] = None
    print(f'Identified {len(reference_tids)} transcript IDs to use...')
    reference_mask = tis_df['Tid'].isin(reference_tids)
    print(f'{reference_mask.sum()} TISs are mapped to these transcript IDs...')
    tis_df['DropReason'] = append_tag(tis_df['DropReason'], 'NotReferenceTranscript', target_mask=~reference_mask)

    # filter for readcount support
    support_mask = tis_df[count_col] >= min_putative_tis_counts
    print(f'Filtering down to {support_mask.sum()} TISs with readcount support of {count_col} >= {min_putative_tis_counts}...')
    tis_df['DropReason'] = append_tag(tis_df['DropReason'], 'LowReadcounts', target_mask=~support_mask)

    # filter for significance
    enrichment_mask = tis_df['TISPvalue'] <= tis_enrichment_max_p
    frame_mask = tis_df['RiboPvalue'] <= frame_test_max_p
    combined_q_mask = tis_df['FisherQvalue'] <= combined_test_max_q
    significance_mask = enrichment_mask & frame_mask & combined_q_mask
    print(f'Filtering down to {significance_mask.sum()} TISs meeting significance of (TISPvalue <= {tis_enrichment_max_p}) & (RiboPvalue <= {frame_test_max_p}) & (FisherQvalue <= {combined_test_max_q})...')
    tis_df['DropReason'] = append_tag(tis_df['DropReason'], 'NotSignificant', target_mask=~significance_mask)

    reference_tis_df = tis_df[tis_df['DropReason'].isnull()]
    tis_idx_to_keep = []
    print(f'Per transcript, keeping the TISs with the highest readcount and excluding downstream TISs within {tis_distance_buffer} nt...')
    for tid, idxs in tqdm(reference_tis_df.groupby('Tid').groups.items()):
        subset = reference_tis_df.loc[idxs, :].sort_values([count_col], ascending=False)
        while subset.shape[0] > 0:
            # If the annotated start site is present for any transcript, then remove it and its neighbors
            annotated_mask = subset['TisType'].str.contains('Annotated')
            assert annotated_mask.sum() <= 1, f"transcript {tid} has more than 1 annotated canonical start site"
            if annotated_mask.sum() > 0:
                selected_tis_idx = subset[annotated_mask].index[0] # should only ever be 1
            else:
                selected_tis_idx = subset.index[0]

            # identify TISs downstream of it
            selected_tis_start = subset.loc[selected_tis_idx, 'Start']
            downstream_mask = (subset['Start'] >= selected_tis_start) & (subset['Start'] <= (selected_tis_start + tis_distance_buffer)) # should be inclusive of the selected_tis entry

            # keep the top TIS and remove the rest
            tis_idx_to_keep.append(selected_tis_idx)
            subset = subset[~downstream_mask]

    downstream_exclusion_idxs = reference_tis_df[~reference_tis_df.index.isin(tis_idx_to_keep)].index.tolist()
    tis_df['DropReason'] = append_tag(tis_df['DropReason'], 'UpstreamTIS', target_indices=downstream_exclusion_idxs)

    inclusion_mask = tis_df['DropReason'].isnull()
    filtered_tis_df = tis_df[inclusion_mask].reset_index(drop=True)
    dropped_tis_df = tis_df[~inclusion_mask].reset_index(drop=True)
    print(f'Keeping a total of {filtered_tis_df.shape[0]} TISs representing unique protein isoforms')
    print(f'These TISs represent {len(filtered_tis_df["GenomeStart"].unique())} unique genomic positions')

    if return_dropped:
        return filtered_tis_df, dropped_tis_df
    else:
        return filtered_tis_df

# UNUSED
def peak_finding_filter_ribotish_results(
    tis_df, 
    transcript_support_levels=['1','2','3'], 
    min_tis_counts=None,
    min_percentile_tis_counts=90,
    min_putative_tis_counts=6, # 6 is estimated from the 25th percentile counts of annotated MANE start sites
    peak_qval_magnitude=4, # 4 is estimated from 95th percentile of qvalue change for all OBSERVED neighbor TIS pairs
):
    reference_tids = identify_reference_transcripts(
        tis_df, 
        transcript_support_levels=transcript_support_levels,
        min_tis_counts=min_tis_counts,
        min_percentile_tis_counts=min_percentile_tis_counts
    )

    print(f'Identified {len(reference_tids)} transcript IDs to use...')
    reference_mask = tis_df['Tid'].isin(reference_tids)
    print(f'{reference_mask.sum()} TISs are mapped to these transcript IDs...')
    reference_tis_df = tis_df[reference_mask]
    reference_tis_df = reference_tis_df.sort_values(['Tid', 'Start'])
    preceded_mask = reference_tis_df['Start'].diff() == 3
    reference_tis_df['preceded'] = preceded_mask
    reference_tis_df['count_diff'] = reference_tis_df['TISCounts']
    reference_tis_df.loc[preceded_mask, 'count_diff'] = reference_tis_df['TISCounts'].diff()[preceded_mask]
    reference_tis_df['log_qval_diff'] = -np.log10(reference_tis_df['TISQvalue'])
    reference_tis_df.loc[preceded_mask, 'log_qval_diff'] = (-np.log10(reference_tis_df['TISQvalue'])).diff()[preceded_mask]

    tis_to_keep = []
    for tid, idxs in tqdm(reference_tis_df.groupby('Tid').groups.items()):
        subset = reference_tis_df.loc[idxs, :].sort_values(['Start'], ascending=False)
        # always accept the annotated start site
        annotated_mask = subset['TisType'].str.contains('Annotated')
        tis_to_keep.append(subset[annotated_mask])
        subset = subset[~annotated_mask].reset_index(drop=True).copy()

        current_peak_i = None
        for i, r in subset.iterrows():
            if r['preceded']: # continuity, logic of peak finding depends on the prior state
                if current_peak_i is not None: # we already called a peak
                    if r['count_diff'] >= 0: # if the count continues to increase, update the tracker to the endmost peak
                        current_peak_i = i
                    else: # otherwise, add the putative peak and reset the tracker
                        tis_to_keep.append(subset[subset.index == current_peak_i]) 
                        current_peak_i = None
                else: # we have not previously called a peak
                    if (r['TISCounts'] >= min_putative_tis_counts) and (r['log_qval_diff'] >= peak_qval_magnitude): # found a new peak
                        current_peak_i = i
            else: # no continuity, logic of peak finding is decoupled from the prior state
                if current_peak_i is not None: # add the last found peak and reset the tracker since the current position is irrelevant to the peak determination of the last
                    tis_to_keep.append(subset[subset.index == current_peak_i])
                    current_peak_i = None
                if (r['TISCounts'] >= min_putative_tis_counts) and (r['log_qval_diff'] >= peak_qval_magnitude): # found a new peak
                    current_peak_i = i
                
        if current_peak_i is not None:
            tis_to_keep.append(subset[subset.index == current_peak_i])

    filtered_tis_df = pd.concat(tis_to_keep, axis=0)
    print(f'Keeping {filtered_tis_df.shape[0]} TISs from reference transcripts, using a peak finding algorithm that selects TIS with at least {min_putative_tis_counts} counts and a 10^{peak_qval_magnitude} increase in TISDiff significance from the prior position')
    return filtered_tis_df.reset_index(drop=True)

######## I/O ########

def write_csv_to_wig_file(input_file, output_file, track_name, track_description, 
                          value_column='TISCounts', transform=None, 
                          input_dirpath='/lab/barcheese01/smaffa/filtered_tis_data',
                          output_dirpath='/lab/barcheese01/smaffa/igv_files'):
    """
    Write an output metric for ribotish results to a .wig file, which can be imported into IGV for visualization 
    
    :param input_file: Description
    :param output_file: Description
    :param track_name: Description
    :param track_description: Description
    :param value_column: Description
    :param transform: Description
    :param input_dirpath: Description
    :param output_dirpath: Description
    """
    if '.csv' in input_file:
        tis_df = pd.read_csv(os.path.join(input_dirpath, input_file))
    elif '.txt' in input_file:
        tis_df = pd.read_csv(os.path.join(input_dirpath, input_file), sep='\t')

    # trim the input table into the necessary columns for the output wig file
    df_slim = tis_df.loc[:, [value_column, 'GenomePos']]
    genome_annotations = df_slim['GenomePos'].str.split(':', expand=True)
    genome_annotations.columns = ['chr', 'span', 'strand']
    spans = genome_annotations['span'].str.split('-', expand=True)
    spans.columns = ['start', 'end']
    df_slim = pd.concat([df_slim, genome_annotations, spans], axis=1)
    df_slim['locus'] = df_slim.apply(lambda x: x['start'] if x['strand'] == '+' else x['end'] if x['strand'] == '-' else 'None', axis=1)

    if transform is not None:
        df_slim[value_column] = transform(df_slim[value_column])

    with open(os.path.join(output_dirpath, output_file), "w") as f:
        for chr, idx in df_slim.groupby('chr').groups.items():
            if 'chr' in chr:
                f.write(f'track type=wiggle_0 name={track_name} description={track_description}\n')
                f.write(f'variableStep chrom={chr} span=1\n')
                subset = df_slim.loc[idx, :].sort_values('locus')
                for locus, counts in zip(subset['locus'].tolist(), subset[value_column].tolist()):
                    if counts != 0:
                        f.write(f'{locus} {counts}\n')
    print(f'Saved file to {os.path.join(output_dirpath, output_file)}')


##### GTF handling #####

def load_gtf_annotations(gtf_path=GTF_FILE, features=['start_codon', 'CDS', 'UTR']):
    """
    Returns a list of dataframes extracting fields from a .gtf annotation file, one dataframe per feature type. Each dataframe is a filtering on the full
    annotation table, keeping only the entries with the feature_type corresponding to the elements of `features`
    
    :param gtf_path: Description
    :param features: Description
    """
    annotation_tables = []
    if isinstance(features, str):
        features = [features]
    for feature in features:
        print(f'Reading {feature}s from {gtf_path}')
        annotation_tables.append(load_transcript_annotations(gtf_path=gtf_path, feature_type=feature))
    if len(annotation_tables) == 1:
        return annotation_tables[0]
    else:
        return annotation_tables
    
def get_start_codons(start_codon_annotations=None, genome_file=GENOME_FILE, gtf_path=GTF_FILE):
    """
    Extract start codon sequences for the `start_codon` annotations in a .gtf file
    
    :param start_codon_annotations: Description
    :param genome_file: Description
    :param gtf_path: Description
    """
    from pyfaidx import Fasta
    from Bio.Seq import Seq

    assert os.path.exists(genome_file), 'must supply an assembly fasta file as reference'
    genome = Fasta(genome_file)

    if start_codon_annotations is None:
        assert os.path.exists(gtf_path), 'if `start_codon_annotations` not provided, must provide a GTF file'
        start_codon_annotations = load_gtf_annotations(gtf_path=gtf_path, features=['start_codon'])

    transcript_to_start_codon = dict()
    for i in tqdm(start_codon_annotations.index.tolist()):
        chrom = start_codon_annotations.loc[i, 'chromosome']
        start = start_codon_annotations.loc[i, 'start']
        end = start_codon_annotations.loc[i, 'end']
        strand = start_codon_annotations.loc[i, 'strand']
        transcript_id = start_codon_annotations.loc[i, 'transcript_id']

        seq = Seq(genome[chrom][start-1:end].seq)

        if strand == '-':
            seq = seq.reverse_complement()
            
        if transcript_id in transcript_to_start_codon:
            transcript_to_start_codon[transcript_id] = transcript_to_start_codon[transcript_id] + str(seq)
        else:
            transcript_to_start_codon[transcript_id] = str(seq)
    transcript_to_start_codon = pd.DataFrame(pd.Series(transcript_to_start_codon, name='StartCodon')).reset_index(names=['Tid'])
    return transcript_to_start_codon

def get_utr_lengths(cds_annotations=None, utr_annotations=None, gtf_path=GTF_FILE):
    """
    Calculate 5' UTR lengths, which is equivalent to the `Start` position of the annotated start codon relative to the spliced transcript
    """
    # Load annotations for UTRs and CDSs (from which to identify 5' UTRs)
    if cds_annotations is None or utr_annotations is None:
        assert os.path.exists(gtf_path), 'if `cds_annotations` or `utr_annotations` not provided, must provide a GTF file'
    if utr_annotations is None:
        utr_annotations = load_gtf_annotations(gtf_path=gtf_path, features=['UTR'])
    if cds_annotations is None:
        cds_annotations = load_gtf_annotations(gtf_path=gtf_path, features=['CDS'])

    # Get 5' CDS annotations
    forward_cds_start_idxs = cds_annotations[cds_annotations['strand'] == '+'].reset_index().sort_values(['transcript_id', 'start']).groupby('transcript_id').first()['index']
    reverse_cds_start_idxs = cds_annotations[cds_annotations['strand'] == '-'].reset_index().sort_values(['transcript_id', 'start'], ascending=[False, True]).groupby('transcript_id').first()['index']
    first_cds = cds_annotations.loc[forward_cds_start_idxs.tolist() + reverse_cds_start_idxs.tolist()]

    # Merge the CDS start positions with the UTR annotations and filter based on strand
    utr_df = utr_annotations.merge(first_cds[['start', 'transcript_id']].rename({'start': 'cds_start'}, axis=1), left_on='transcript_id', right_on='transcript_id')
    utr_df = utr_df[
        ((utr_df['strand'] == '+') & (utr_df['end'] < utr_df['cds_start'])) | # forward strand, keep UTRs upstream of CDS
        ((utr_df['strand'] == '-') & (utr_df['end'] > utr_df['cds_start'])) # reverse strand, keep UTRs upstream of CDS in reverse direction
    ]
    # Calculate the total segment length of UTRs on the 5' end
    utr_df['utr_length'] = (utr_df['end'] - utr_df['start'] + 1)
    concat_utr_length = pd.DataFrame(utr_df.groupby('transcript_id')['utr_length'].sum()).reset_index(names=['Tid'])
    return concat_utr_length

def get_canonical_genome_positions(cds_annotations=None, gtf_path=GTF_FILE):
    """
    Generate strings detailing the genomic position of the outermost CDS regions in each transcript
    
    :param cds_annotations: Description
    :param gtf_path: Description
    """
    if cds_annotations is None:
        assert os.path.exists(gtf_path), 'if `cds_annotations` not provided, must provide a GTF file'
        cds_annotations = load_gtf_annotations(gtf_path=gtf_path, features=['CDS'])
    
    sorted_cds_annotations = cds_annotations.sort_values(['transcript_id', 'start']) # used to find start and end of coding sequence
    unique_transcript_cds_annotations = sorted_cds_annotations.drop_duplicates(subset=['transcript_id']) # used to extract common annotations

    transcript_starts = sorted_cds_annotations.groupby('transcript_id').first()['start']
    transcript_ends = sorted_cds_annotations.groupby('transcript_id').last()['end']
    transcript_chromosomes = unique_transcript_cds_annotations.set_index('transcript_id')['chromosome']
    transcript_strands = unique_transcript_cds_annotations.set_index('transcript_id')['strand']

    # combine annotations and assemble genome position string relative to strand
    cds_genome_pos_df = pd.concat([transcript_chromosomes, transcript_starts, transcript_ends, transcript_strands], axis=1)
    cds_genome_pos_df['GenomePos'] = cds_genome_pos_df.apply(lambda x: f'{x["chromosome"]}:{x["start"]}-{x["end"]}:{x["strand"]}', axis=1)
    cds_genome_pos_df.index.name = 'Tid'
    cds_genome_pos_df = cds_genome_pos_df.reset_index()

    return cds_genome_pos_df

def get_protein_products(protein_fasta=PROTEIN_FASTA):
    """
    Reads in a fasta of all protein sequences to create a mapping of transcript IDs to translated protein sequences
    
    :param protein_fasta: Description
    """
    from Bio import SeqIO

    print(f'Reading protein sequences from {protein_fasta}')

    gencode_protein_products = dict()
    for record in SeqIO.parse(PROTEIN_FASTA, format='fasta'):
        sequence_ids = record.id
        transcript_id = [tag for tag in sequence_ids.split('|') if 'ENST' in tag][0]
        gencode_protein_products[transcript_id] = str(record.seq)
    gencode_protein_products = pd.DataFrame(pd.Series(gencode_protein_products), columns=['AASeq']).reset_index(names=['Tid'])
    gencode_protein_products['AALen'] = gencode_protein_products['AASeq'].apply(lambda x: len(x))

    return gencode_protein_products

##### Imputation #####

def impute_missing_canonical_starts(
    tis_df, 
    transcript_ids=None, 
    genome_pos=None, utr_lengths=None, start_codons=None, protein_products=None,
    start_codon_annotations=None, cds_annotations=None, utr_annotations=None, genome_file=GENOME_FILE, gtf_path=GTF_FILE, protein_fasta=PROTEIN_FASTA,
    static_annotations={'TisType': 'Annotated', 'RecatTISType': 'Annotated', 'TISGroup': 0, 'TISCounts': 0, 'NormTISCounts': 0}
):
    """
    Given a filtered output from ribotish, creates a table of entries for canonical starts that were undetected by `ribotish predict`
    
    :param tis_df: Description
    :param transcript_ids: Description
    :param genome_pos: Description
    :param utr_lengths: Description
    :param start_codons: Description
    :param protein_products: Description
    :param start_codon_annotations: Description
    :param cds_annotations: Description
    :param utr_annotations: Description
    :param genome_file: Description
    :param gtf_path: Description
    :param protein_fasta: Description
    :param static_annotations: Description
    """
    if transcript_ids is None:
        transcript_ids = tis_df['Tid'].unique().tolist() # initialize to all
    missing_canonical_ids = tis_df.assign(has_annotated=lambda x: x['RecatTISType'] == 'Annotated').groupby('Tid')['has_annotated'].sum().loc[lambda x: x == 0].index.tolist()
    transcript_ids = set(missing_canonical_ids).intersection(transcript_ids)

    # Get existing annotations per transcript id 
    existing_tis_annotations = tis_df[tis_df['Tid'].isin(transcript_ids)].drop_duplicates(subset=['Tid']).loc[
        :, ['Gid', 'Tid', 'Symbol', 'GeneType', 'MANE_Select', 'transcript_support_level']
    ]

    # For each of the additional columns, run the corresponding function if not provided
    if genome_pos is None:
        genome_pos = get_canonical_genome_positions(cds_annotations=cds_annotations, gtf_path=gtf_path)
    genome_pos = genome_pos[['Tid', 'GenomePos']]

    if utr_lengths is None:
        utr_lengths = get_utr_lengths(cds_annotations=cds_annotations, utr_annotations=utr_annotations, gtf_path=gtf_path)
    utr_lengths = utr_lengths.rename({'utr_length': 'Start'}, axis=1)

    if start_codons is None:
        start_codons = get_start_codons(start_codon_annotations=start_codon_annotations, genome_file=genome_file, gtf_path=gtf_path)
    
    if protein_products is None:
        protein_products = get_protein_products(protein_fasta=protein_fasta)
    protein_products = protein_products[['Tid', 'AALen']]

    # merge all annotations
    imputed_tis_df = existing_tis_annotations.merge(genome_pos, how='left').merge(utr_lengths, how='left').merge(start_codons, how='left').merge(protein_products, how='left')
    imputed_tis_df = imputed_tis_df.dropna(subset=['GenomePos', 'Start', 'StartCodon', 'AALen'])

    # assign static annotations
    for k, v in static_annotations.items():
        imputed_tis_df[k] = v

    imputed_tis_df['Imputed'] = True
    
    return imputed_tis_df


def tmm_normalization_factors(input_sample, reference_sample, experiment_table=None, m_trim=0.3, a_trim=0.1, **experiment_table_kws):
    if experiment_table is None:
        experiment_table, _, _ = load_experiment_manifest(**experiment_table_kws)

    # readcounts per gene
    unique_gene_rnaseq_counts = pd.concat([read_rnaseq_counts(f) for f in experiment_table.set_index('sample').loc[input_sample, 'rnaseq_count_file']], axis=1).sum(axis=1)
    unique_gene_rnaseq_counts_ref = pd.concat([read_rnaseq_counts(f) for f in experiment_table.set_index('sample').loc[reference_sample, 'rnaseq_count_file']], axis=1).sum(axis=1)

    # total readcounts for sample and reference
    n = unique_gene_rnaseq_counts.sum()
    n_ref = unique_gene_rnaseq_counts_ref.sum()

    # depth-normalized counts for sample and reference
    norm_y = unique_gene_rnaseq_counts / n 
    norm_y_ref = unique_gene_rnaseq_counts / n_ref

    # M values and A values
    A = 0.5 * np.log2(norm_y * norm_y_ref) # A values per gene
    A[(norm_y == 0) | (norm_y_ref == 0)] = np.nan 
    M = np.log2(norm_y / norm_y_ref) # M values per gene

    # weights in the sum for the scale factor
    w = ((n - unique_gene_rnaseq_counts) * norm_y) + ((n_ref - unique_gene_rnaseq_counts_ref) * norm_y_ref)

    # trim most extreme M and A values
    M_sorted = M.dropna().sort_values()
    A_sorted = A.dropna().sort_values()
    n_mtrim = int(m_trim * len(M_sorted)) + 1
    n_atrim = int(a_trim * len(A_sorted)) + 1
    m_idx_to_drop = M_sorted.head(n_mtrim).index.tolist() + M_sorted.tail(n_mtrim).index.tolist()
    a_idx_to_drop = A_sorted.head(n_atrim).index.tolist() + A_sorted.tail(n_atrim).index.tolist()
    idx_to_keep = list(set(M_sorted.index.tolist() + A_sorted.index.tolist()).difference(set(m_idx_to_drop + a_idx_to_drop)))

    # calculate TMM over remaining values after trimming
    TMM = (w[idx_to_keep] * M[idx_to_keep]).sum() / w[idx_to_keep].sum()

    return 2 ** TMM