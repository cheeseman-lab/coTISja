# standard data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import seaborn as sns

import os
import re

# convenience
from tqdm import tqdm

def plot_tis_over_cell_lines(transcript_id, tis_summary_df, te_matrix, 
                             figsize=(16, 3), range_extension=0.01, cds_height_scale=0.05, position_width=2.4, 
                             sample_palette='tab10', tis_type_palette='Set2', upstream_crop=100):
    """
    Custom plot for translational efficiency per TIS for a given transcript. Plots a transcript track parallel to TIS-measurements along the transcript body.
    """
    
    def extend_lims(lims, extension_scale):
        if isinstance(extension_scale, tuple):
            left_ext_scale, right_ext_scale = extension_scale
        else:
            left_ext_scale = extension_scale
            right_ext_scale = extension_scale
        lim_range = lims[1] - lims[0]
        extended_lim = (lims[0] - left_ext_scale * lim_range, lims[1] + right_ext_scale * lim_range)
        return extended_lim

    df = tis_summary_df[
        tis_summary_df['Tid'].str.contains(transcript_id)
    ].sort_values('Start')[['TIS', 'Start', 'RecatTISType']].merge(
        te_matrix, left_on='TIS', right_index=True
    ).set_index(['TIS', 'Start', 'RecatTISType']).melt(ignore_index=False, var_name='Sample', value_name='LogTE').reset_index().dropna(subset=['LogTE'])
    symbol = tis_summary_df[tis_summary_df['Tid'].str.contains(transcript_id)]['Symbol'].iloc[0]

    # get key x value markers
    gene_5utr_body = (0, df['Start'].max())

    # get yrange
    ylims = (df['LogTE'].min(), df['LogTE'].max())
    extended_ylim = extend_lims(ylims, range_extension)
    # yrange = ylims[1] - ylims[0]
    # extended_ylim = (ylims[0] - range_extension * yrange, ylims[1] + range_extension * yrange)
    gene_cds_height = cds_height_scale * (extended_ylim[1] - extended_ylim[0])

    # organize all of the starts and associated metadata for plotting
    unique_starts = sorted(df['Start'].unique().tolist())
    unique_samples = sorted(df['Sample'].unique().tolist())
    unique_tis_types = sorted(df['RecatTISType'].unique().tolist())
    sample_width = position_width / len(unique_samples)
    sample_rect_xpos = pd.Series(np.arange(-position_width/2, position_width/2, sample_width), index=unique_samples)
    start_to_tis_type = df.drop_duplicates(['Start', 'RecatTISType']).set_index('Start')['RecatTISType']

    # assign colors to the TE bars
    if isinstance(sample_palette, str):
        sample_palette = pd.Series(sns.color_palette(sample_palette, n_colors=len(unique_samples)), index=unique_samples)
    elif isinstance(sample_palette, dict):
        sample_palette = pd.Series(sample_palette)
    elif not isinstance(sample_palette, pd.Series):
        raise TypeError('`sample_palette`')
    
    # assign colors to the TIS annotations
    if isinstance(tis_type_palette, str):
        tis_type_palette = pd.Series(sns.color_palette(tis_type_palette, n_colors=len(unique_tis_types)), index=unique_tis_types)
    elif isinstance(tis_type_palette, dict):
        tis_type_palette = pd.Series(tis_type_palette)
    elif not isinstance(tis_type_palette, pd.Series):
        raise TypeError('`tis_type_palette`')

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[1, 9])
    
    # plot the TIS annotations
    for start in unique_starts:
        axs[1].axvline(start-position_width/2, color='black', linestyle='dotted', alpha=0.2)
        axs[1].axvline(start+position_width/2, color='black', linestyle='dotted', alpha=0.2)
        tis_type = start_to_tis_type.loc[start]
        block_width = position_width/3
        axs[0].add_patch(Rectangle((start-block_width/2, -gene_cds_height), block_width, 2*gene_cds_height, facecolor=tis_type_palette.loc[tis_type], linewidth=0, fill=True, zorder=1))

    # may need to readjust figure bounds depending on the interaction between start positions and sample offsets
    empirical_min_x = df['Start'].iloc[0]
    empirical_max_x = gene_5utr_body[1]
    for i, r in df.iterrows():
        start = r['Start']
        sample = r['Sample']
        height = r['LogTE']

        if height < 0:
            xy = (start + sample_rect_xpos.loc[sample], height)
            ht = -height
        else:
            xy = (start + sample_rect_xpos.loc[sample], 0)
            ht = height
        axs[1].add_patch(Rectangle(xy, sample_width, ht, edgecolor='black', facecolor=sample_palette.loc[sample], linewidth=0.1, fill=True))
        if xy[0] < empirical_min_x:
            empirical_min_x = xy[0]
        if (xy[0] + sample_width) > empirical_max_x:
            empirical_max_x = xy[0] + sample_width
            
    # set axis labels
    axs[1].set_xlabel('TIS position relative to transcript start')
    axs[1].set_ylabel('Log2(Ribo / RNA)')
    axs[0].set_title(f'{transcript_id} ({symbol})')
    axs[0].set_axis_off()

    # set axis limits
    if (gene_5utr_body[0] + upstream_crop) < empirical_min_x:
        # nothing within `upstream_crop`` upstream of the first TIS, re-crop the figure close to the earliest TIS
        final_xlims = extend_lims((empirical_min_x, empirical_max_x), range_extension)
    else:
        final_xlims = extend_lims((gene_5utr_body[0], empirical_max_x), range_extension)
    axs[0].set_xlim(final_xlims[0], final_xlims[1])
    axs[1].set_xlim(final_xlims[0], final_xlims[1])
    axs[1].set_ylim(extended_ylim[0], extended_ylim[1])
    axs[1].axhline(0, color='black', linestyle='dashed')
    axs[0].plot([gene_5utr_body[0], final_xlims[1]], [0, 0], color='black', linewidth=1, zorder=0) # full transcript body
    if df[df['RecatTISType'] == 'Annotated'].shape[0] > 0: # we know where the start is
        annotated_start = df[df['RecatTISType'] == 'Annotated']['Start'].iloc[0]
        axs[0].add_patch(Rectangle((annotated_start, -gene_cds_height/2), final_xlims[1] - annotated_start, gene_cds_height, edgecolor='black', facecolor='black', fill=True, zorder=0)) # annotated CDS

    axs[1].legend(
        handles = [Patch(facecolor='white', edgecolor='white', label='TIS Type')] + 
        [
            Patch(facecolor=c, edgecolor='white', label=t) for t, c in tis_type_palette.items()
        ] + [Patch(facecolor='white', edgecolor='white', label='Sample')] + [
            Patch(facecolor=c, edgecolor='white', label=s) for s, c in sample_palette.items()
        ], loc='lower left', bbox_to_anchor=(1, -0.1)
    )

def plot_volcano(df, x='log2FoldChange', y='padj', transform_y=lambda x: -np.log10(x), min_x=1, max_p=0.05, **plt_kws):
    plot_df = df.copy()
    plot_df['x'] = plot_df[x]
    
    if transform_y is None:
        transform_y = lambda x: x
    plot_df['y'] = transform_y(plot_df[y])
    hline = transform_y(max_p)

    plot_df['highlight'] = (plot_df['x'].abs() > min_x) & (plot_df['y'] > hline)

    sns.scatterplot(plot_df, x='x', y='y', hue='highlight', palette={True: 'tab:blue', False: 'tab:gray'}, **plt_kws)
    plt.gca().axhline(hline, color='red', linestyle='dotted', alpha=0.5)
    plt.gca().axvline(-min_x, color='red', linestyle='dotted', alpha=0.5)
    plt.gca().axvline(min_x, color='red', linestyle='dotted', alpha=0.5)
    return plt.gca()


def plot_cross_scatter(merged_tis_rna_df, max_lfc_se = 1):
    plot_df = merged_tis_rna_df.copy()

    plot_df.loc[:, 'Significant'] = 'Neither'
    plot_df.loc[plot_df['SigRibo'], 'Significant'] = 'Ribo'
    plot_df.loc[plot_df['SigRNA'], 'Significant'] = 'RNA'
    plot_df.loc[plot_df['SigRibo'] & plot_df['SigRNA'], 'Significant'] = 'Both'

    filtered_plot_df = plot_df[
        (plot_df['Ribo_lfcSE'] < max_lfc_se) &
        (plot_df['RNA_lfcSE'] < max_lfc_se)
    ]

    sns.scatterplot(
        filtered_plot_df[filtered_plot_df['Significant'] == 'Neither'],
        x='RNA_log2FoldChange', y='Ribo_log2FoldChange', 
        color='tab:gray', alpha=0.5, size=2, legend=False
    )
    sns.scatterplot(
        filtered_plot_df[filtered_plot_df['Significant'] != 'Neither'], 
        x='RNA_log2FoldChange', y='Ribo_log2FoldChange', 
        hue='Significant', palette={'Ribo': 'tab:blue', 'RNA': 'tab:red', 'Both': 'tab:purple'}
    )
    plt.gca().axhline(0, color='black', linestyle='dashed', alpha=0.5)
    plt.gca().axvline(0, color='black', linestyle='dashed', alpha=0.5)
    return plt.gca(), filtered_plot_df

def plot_geneset_stem(gsea_df_up, gsea_df_down, top_n=5, 
                      term_column='Term', size_column='Odds Ratio', p_column='Adjusted P-value', significance_threshold=0.05,
                      figsize=(12, 6)):
    up_df = gsea_df_up.sort_values(p_column).head(top_n).copy()
    up_df['p_transform'] = -np.log10(up_df[p_column])
    up_df['Geneset'] = up_df[term_column].apply(lambda x: x.split('_')[0] + ': ' + ' '.join([y.title() for y in x.split('_')[1:]]))
    
    down_df = gsea_df_down.sort_values(p_column).head(top_n).copy()
    down_df['p_transform'] = -np.log10(down_df[p_column])
    down_df['Geneset'] = down_df[term_column].apply(lambda x: x.split('_')[0] + ': ' + ' '.join([y.title() for y in x.split('_')[1:]]))

    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0})
    sns.barplot(down_df, x='p_transform', y='Geneset', color='tab:blue', ax=axs[0])
    sns.barplot(up_df, x='p_transform', y='Geneset', color='tab:red', ax=axs[1])

    max_lim = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
    axs[0].set_xlim(0, max_lim)
    axs[1].set_xlim(0, max_lim)

    axs[0].invert_xaxis()
    axs[0].set_xlabel('-log10 (FDR-adjusted p-value)')
    axs[1].set_xlabel('-log10 (FDR-adjusted p-value)')
    axs[1].yaxis.set_label_position('right')
    axs[1].yaxis.tick_right()

    axs[0].axvline(-np.log10(significance_threshold), color='black', linestyle='dashed', alpha=0.5)
    axs[1].axvline(-np.log10(significance_threshold), color='black', linestyle='dashed', alpha=0.5)

    return fig, axs