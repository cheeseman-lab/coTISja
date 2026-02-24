from filter_utils import *

# Filter to a set of TISs for downstream analysis:
#     1) Determine a set of transcripts from which to keep TISs
#         a) Filter by data-independent annotations: MANE_Select and TSL
#         b) Filter by readcounts from any TIS mapped to that transcript (threshold used is minimum readcounts of 0, aka all TISs)
#         c) Filter by ribotish significance of any TIS mapped to that transcript (threshold used ignores all significance values for this step)
#     2) Filter TISs by readcounts (threshold used is minimum of 0.1 RPM)
#     3) Filter TISs by significance (threshold used is original strategy for *filtered_predictor files)
#     4) Keep the annotated start site for all transcripts remaining
#     5) For each transcript, iteratively select the highest readcount TIS, then exclude all other nearby TISs downstream of it

# mapping of input files to filtered outputs is in the table /lab/barcheese01/smaffa/ribotish_sample_manifest.csv

# file manifest: maps inputs to outputs
REPLICATE_LEVEL = False

replicate_df = pd.read_csv('ribotish_replicate_manifest.csv')
sample_df = pd.read_csv('ribotish_sample_manifest.csv')
if REPLICATE_LEVEL:
    experiment_table = replicate_df.dropna(subset=['predict_file'])
else:
    experiment_table = sample_df.merge(
        replicate_df[
            replicate_df['condition'] == 'TIS'
        ].groupby('sample').apply(lambda x: list(x['rnaseq_count_file'])).rename('rnaseq_count_file'), left_on='sample', right_index=True
    )

# common annotation file
gtf_df = load_transcript_annotations(GTF_FILE)

# define columns to save in output files
columns_to_keep = [
    'Gid', 'Tid', 'Symbol', 'GeneType', 'GenomePos', 
    'Start', 'StartCodon', 'TisType', 'RecatTISType', 'TISGroup', 
    'TISCounts', 'NormTISCounts', 'GeneRNASeqCounts', 'TotalRNASeqCounts',
    'AALen', 'MANE_Select', 'transcript_support_level'
]

columns_to_keep = [
    'Gid', 'Tid', 'Symbol', 'GeneType', 'GenomePos', 
    'Start', 'StartCodon', 'TisType', 'RecatTISType', 'TISGroup', 
    'TISCounts', 'NormTISCounts', 'GeneRNASeqCounts', 'TotalRNASeqCounts',
    'AALen', 'MANE_Select', 'transcript_support_level'
]

# iterate through experiments
for i, exp_row in experiment_table.iterrows():
    # extract input and output filepaths
    if REPLICATE_LEVEL:
        sample_name = f'{exp_row["sample"]}_rep{exp_row["replicate"]}'
        rnaseq_count_files = [exp_row['rnaseq_count_file']]
    else:
        sample_name = exp_row['sample']
        rnaseq_count_files = exp_row['rnaseq_count_file']
    predict_file = exp_row['predict_file']
    output_file = exp_row['filtered_file']
    dropped_file = exp_row['dropped_file']
    print(f'Processing {sample_name}...')

    # import data
    print(f'Reading TIS table from: {predict_file}')
    all_tis_df = import_ribotish_results(predict_file, gtf_df=gtf_df)
    all_tis_df = recategorize_tis_type(all_tis_df, original_column='TisType', output_column='RecatTISType')

    # normalize readcounts
    print('Normalizing using total counts from:')
    print('\n'.join(rnaseq_count_files))
    rnaseq_counts = pd.concat([read_rnaseq_counts(f) for f in rnaseq_count_files], axis=1).sum(axis=1).rename('GeneRNASeqCounts')
    all_tis_df = all_tis_df.merge(rnaseq_counts, left_on=['Gid'], right_index=True, how='left')
    all_tis_df['GeneRNASeqCounts'] = all_tis_df['GeneRNASeqCounts'].fillna(0)
    all_tis_df['TotalRNASeqCounts'] = rnaseq_counts.sum()
    norm_tis_df = normalize_tis_counts(all_tis_df, divisor_column='TotalRNASeqCounts')
    norm_tis_df['NormTISCounts']  = norm_tis_df['NormTISCounts'] * 1e6

    # perform the filtering procedure
    filtered_tis_df, dropped_tis_df = filter_ribotish_results(
        norm_tis_df, 
        transcript_support_levels=['1','2','3'], 
        reference_min_tis_counts=None,
        reference_min_percentile_tis_counts=None,
        reference_count_col='TISCounts',
        reference_tis_enrichment_max_p=1,
        reference_frame_test_max_p=1,
        reference_combined_test_max_q=1,
        min_putative_tis_counts=0.1, # 0.1 calibrated to ~ 5 counts in HeLa normalized over the sum of mapped reads to the TIS replicates
        count_col='NormTISCounts',
        tis_enrichment_max_p=0.01,
        frame_test_max_p=0.01,
        combined_test_max_q=0.05,
        tis_distance_buffer=30,
        return_dropped=True
    )

    # subset columns to the desired output
    filtered_tis_df = filtered_tis_df.loc[:, columns_to_keep]
    dropped_tis_df = dropped_tis_df.loc[:, columns_to_keep + ['DropReason']]

    # save output files
    print(f'Saving filtered results to: {output_file}')
    filtered_tis_df.to_csv(output_file, index=False)
    if dropped_file != output_file:
        print(f'Saving exclusion annotations to: {dropped_file}')
        dropped_tis_df.to_csv(dropped_file, index=False)

    # write to a .wig file for visualization in IGV
    write_csv_to_wig_file(
        input_file = output_file,
        output_file = f'{sample_name}_filtered_readcounts.wig',
        track_name = f'{sample_name}_filtered_readcounts',
        track_description = '20260213 filtering procedure',
        input_dirpath = ''
    )
    
