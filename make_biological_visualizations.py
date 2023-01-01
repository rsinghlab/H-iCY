from re import sub
import numpy as np
import os

from torch import normal
from src.matrix_ops.ops import normalize

import seaborn as sns
import matplotlib.pyplot as plt










def make_heatmap_visualization(file_path, cutoff, chromosome_id=20, sub_mat=0, sub_mat_size=100):
    chrom_file_path = os.path.join(file_path, 'chr{}.npz'.format(chromosome_id))
    data = normalize(chrom_file_path, cutoff)
    sub_mat_data = data[sub_mat:sub_mat+sub_mat_size, sub_mat:sub_mat+sub_mat_size]
    dataset = file_path.split('/')[-1]

    ax = sns.heatmap(sub_mat_data, cbar=False)
    plt.axis('off')

    plt.savefig('results/figures/visualizations/encode2/{}_chr{}_loc-{}:{}.png'.format(dataset, chromosome_id, sub_mat, sub_mat+sub_mat_size))





# baseline_fp  = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_synthetic50'  
# smoothing_fp = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_smoothing-gaussian_synthetic50'
# hicplus_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_hicplus-50_synthetic50'
# hicnn_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_hicnn-50_synthetic50'
# hicnn2_fp    = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_hicnn2-25_synthetic25'
# deephic_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_deephic-50_synthetic50'
# vehicle_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_vehicle_synthetic50'
# target_fp    = '/media/murtaza/ubuntu1/hic_data/chromosome_files/synthetic/GM12878_rao_et_al'

# make_heatmap_visualization(baseline_fp, 50)
# make_heatmap_visualization(smoothing_fp, 50)
# make_heatmap_visualization(hicplus_fp, 255)
# make_heatmap_visualization(hicnn_fp, 255)
# make_heatmap_visualization(hicnn2_fp, 255)
# make_heatmap_visualization(deephic_fp, 255)
# make_heatmap_visualization(vehicle_fp, 255)
# make_heatmap_visualization(target_fp, 255)





# baseline_fp  = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_encode2'  
# smoothing_fp = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_smoothing-gaussian_encode2'
# hicplus_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicplus-16_encode2'
# hicnn_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn-16_encode2'
# hicnn2_fp    = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn2-50_encode2'
# deephic_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_deephic-50_encode2'
# vehicle_fp   = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_vehicle_encode2'
# target_fp    = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_rao_et_al'

# make_heatmap_visualization(baseline_fp, 96)
# make_heatmap_visualization(smoothing_fp, 96)
# make_heatmap_visualization(hicplus_fp, 255)
# make_heatmap_visualization(hicnn_fp, 255)
# make_heatmap_visualization(hicnn2_fp, 255)
# make_heatmap_visualization(deephic_fp, 255)
# make_heatmap_visualization(vehicle_fp, 255)
# make_heatmap_visualization(target_fp, 255)


baseline_fp  = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_encode2'  
hicnn_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn-16_encode2'
hicnn_encode0_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn-encode0_encode2'
hicnn_encode2_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn-encode2_encode2'
hicnn_hic026_fp     = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_hicnn-hic026_encode2'
target_fp    = '/media/murtaza/ubuntu1/hic_data/chromosome_files/real/GM12878_rao_et_al'

make_heatmap_visualization(baseline_fp, 96)
make_heatmap_visualization(hicnn_fp, 255)
make_heatmap_visualization(hicnn_encode0_fp, 255)
make_heatmap_visualization(hicnn_encode2_fp, 255)
make_heatmap_visualization(hicnn_hic026_fp, 255)
make_heatmap_visualization(target_fp, 255)



