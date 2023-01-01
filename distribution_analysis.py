from src.evaluation import distribution_analysis
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import os
from src.utils import downsample_chromsosome_file

# real_base_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/real'
# synthetic_base_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/synthetic'

# downsample_chromsosome_file(os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), os.path.join(synthetic_base_path, 'GM12878_synthetic44/'), 44, 'test')




method_to_color = {
    'Downsampled': '#C10001',
    'LRC': '#B7AC44',
    'HiCNN-Baseline': '#009E73',
    'HiCNN-LRC-3': '#C10001',
}
dataset_cutoff_values = {
    'GM12878_rao_et_al':            -1,
    'GM12878_rao_et_al_replicate':  -1,
    'IMR90_rao_et_al':              -1,
    'K562_rao_et_al':               -1,
    'GM12878_encode0':              -1,
    'GM12878_encode1':              -1,
    'GM12878_encode2':              -1,
    'GM12878_hic026':               -1,  
    'GM12878_hic033':               -1, 
    'IMR90_hic057':                 -1,   
    'K562_hic073':                  -1,
    'deeplearning_upscaled':        255,
    'GM12878_synthetic16':          -1,
    'GM12878_synthetic25':          -1,
    'GM12878_synthetic44':          -1,
    'GM12878_synthetic50':          -1,
    'GM12878_synthetic100':         -1,
    'IMR90_synthetic16':            -1,
    'IMR90_synthetic25':            -1,
    'IMR90_synthetic44':            -1,
    'IMR90_synthetic50':            -1,
    'IMR90_synthetic100':           -1,
    'K562_synthetic16':             -1,
    'K562_synthetic25':             -1,
    'K562_synthetic44':             -1,
    'K562_synthetic50':             -1,
    'K562_synthetic100':            -1
}






def interpolate(y):
    return gaussian_filter1d(y, sigma=2)

real_base_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/real'
synthetic_base_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/synthetic'

# file_pairs = [
#     (
#         'encode0', 
#         (os.path.join(real_base_path, 'GM12878_encode0/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'GM12878_synthetic44/'),  os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'encode1', 
#         (os.path.join(real_base_path, 'GM12878_encode1/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'GM12878_synthetic50/'),  os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'encode2', 
#         (os.path.join(real_base_path, 'GM12878_encode2/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'GM12878_synthetic25/'),  os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'hic026', 
#         (os.path.join(real_base_path, 'GM12878_hic026/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'GM12878_synthetic16/'),  os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'hic033', 
#         (os.path.join(real_base_path, 'GM12878_hic033/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'GM12878_synthetic100/'),  os.path.join(synthetic_base_path, 'GM12878_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'hic057', 
#         (os.path.join(real_base_path, 'IMR90_hic057/'), os.path.join(real_base_path, 'IMR90_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'IMR90_synthetic16/'),  os.path.join(synthetic_base_path, 'IMR90_rao_et_al/'), 'Downsampled')
#     ),
#     (
#         'hic073', 
#         (os.path.join(real_base_path, 'K562_hic073/'), os.path.join(real_base_path, 'K562_rao_et_al/'), 'LRC'), 
#         (os.path.join(synthetic_base_path, 'K562_synthetic16/'),  os.path.join(synthetic_base_path, 'K562_rao_et_al/'), 'Downsampled')
#     ),
# ]



file_pairs = [
    (
        'encode0', 
        (os.path.join(real_base_path, 'GM12878_encode0/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'GM12878_hicnn-100_encode0/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'GM12878_hicnn-encode2_encode0/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'encode1', 
        (os.path.join(real_base_path, 'GM12878_encode1/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'GM12878_hicnn-100_encode1/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'GM12878_hicnn-encode2_encode1/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'encode2', 
        (os.path.join(real_base_path, 'GM12878_encode2/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'GM12878_hicnn-100_encode2/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'GM12878_hicnn-encode2_encode2/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'hic026', 
        (os.path.join(real_base_path, 'GM12878_hic026/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'GM12878_hicnn-100_hic026/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'GM12878_hicnn-encode2_hic026/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'hic033', 
        (os.path.join(real_base_path, 'GM12878_hic033/'), os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'GM12878_hicnn-100_hic033/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'GM12878_hicnn-encode2_hic033/'),  os.path.join(real_base_path, 'GM12878_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'hic057', 
        (os.path.join(real_base_path, 'IMR90_hic057/'), os.path.join(real_base_path, 'IMR90_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'IMR90_hicnn-100_hic057/'),  os.path.join(real_base_path, 'IMR90_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'IMR90_hicnn-encode2_hic057/'),  os.path.join(real_base_path, 'IMR90_rao_et_al/'), 'HiCNN-LRC-3')
    ),
    (
        'hic073', 
        (os.path.join(real_base_path, 'K562_hic073/'), os.path.join(real_base_path, 'K562_rao_et_al/'), 'LRC'), 
        (os.path.join(real_base_path, 'K562_hicnn-100_hic073/'),  os.path.join(real_base_path, 'K562_rao_et_al/'), 'HiCNN-Baseline'),
        (os.path.join(real_base_path, 'K562_hicnn-encode2_hic073/'),  os.path.join(real_base_path, 'K562_rao_et_al/'), 'HiCNN-LRC-3')
    )
]



count = 0


for file_pair in file_pairs:
    fig = plt.figure()
    ax = plt.axes()
    x = range(200)
    dataset = file_pair[0]
    file_pair = file_pair[1:]
    print(file_pair)

    for base, target, method in file_pair:
        
        base_dataset = base.split('/')[-2]
        target_dataset = target.split('/')[-2]
        print(base_dataset, target_dataset)


        if base_dataset not in dataset_cutoff_values.keys():
            base_dataset = 'deeplearning_upscaled'

        
        corr = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base, target, 'distribution_analysis', 
            dataset_cutoff_values[base_dataset], dataset_cutoff_values[target_dataset])
        
        ax.plot(x, interpolate(corr), linestyle='solid' ,label=method, linewidth=5, color=method_to_color[method])
    

    plt.ylim([0, 1])
    plt.ylabel("PCC", size=18)
    plt.xlabel("Genomic Distance (10KB)", size=18)


    if count==0:
        plt.legend(fontsize=22)
    
    plt.savefig('results/figures/{}-dist-analysis.png'.format(dataset))
    plt.close()
    plt.cla()
    plt.clf()

    count += 1






























































