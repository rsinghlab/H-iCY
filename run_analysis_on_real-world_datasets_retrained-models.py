import os

from numpy.core.numeric import full
from src import globals
from src.evaluation import common

steps = [
    # 'correlation_metrics',
    # 'hicrep',
    # 'biological_metrics',
    # '3d_similarity',
    # 'significant_interaction'
    # 'insulation_score',
    'conserved_features'
]

DATA_DIRECTORY = '/media/murtaza/ubuntu/hic_data/chromosome_files/real'

upscaled_experiments = [
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode0_encode0/'), # HiCNN Encode 0
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode0_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode0_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode0_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode0_hic033/'),
    
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode2_encode0/'), # HiCNN Encode 2
    os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode2_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode2_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode2_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-encode2_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-hic026_encode0/'), # HiCNN hic026
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-hic026_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-hic026_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-hic026_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-hic026_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-synthetic-ensemble_encode0/'), # HiCNN Synthetic Ensemble
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-synthetic-ensemble_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-synthetic-ensemble_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-synthetic-ensemble_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-synthetic-ensemble_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-real-ensemble_encode0/'), # HiCNN Real Ensemble
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-real-ensemble_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-real-ensemble_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-real-ensemble_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-real-ensemble_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-random-noise_encode0/'), # HiCNN Random Noise
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-random-noise_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-random-noise_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-random-noise_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-random-noise_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-gaussian-noise_encode0/'), # HiCNN Gaussian Noise
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-gaussian-noise_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-gaussian-noise_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-gaussian-noise_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-gaussian-noise_hic033/'),

    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-uniform-noise_encode0/'), # HiCNN Uniform Noise
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-uniform-noise_encode1/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-uniform-noise_encode2/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-uniform-noise_hic026/'),
    # os.path.join(DATA_DIRECTORY, 'GM12878_hicnn-uniform-noise_hic033/'),

]

upscaled_experiments_cutoffs = [globals.dataset_cutoff_values['deeplearning_upscaled']]*len(upscaled_experiments)


upscaled_experiment_names = [
    # 'hicnn-encode0:encode0',
    # 'hicnn-encode0:encode1',
    # 'hicnn-encode0:encode2',
    # 'hicnn-encode0:hic026',
    # 'hicnn-encode0:hic033',

    # 'hicnn-encode2:encode0',
    'hicnn-encode2:encode1',
    # 'hicnn-encode2:encode2',
    # 'hicnn-encode2:hic026',
    # 'hicnn-encode2:hic033',

    # 'hicnn-hic026:encode0',
    # 'hicnn-hic026:encode1',
    # 'hicnn-hic026:encode2',
    # 'hicnn-hic026:hic026',
    # 'hicnn-hic026:hic033',

    # 'hicnn-synthetic-ensemble:encode0',
    # 'hicnn-synthetic-ensemble:encode1',
    # 'hicnn-synthetic-ensemble:encode2',
    # 'hicnn-synthetic-ensemble:hic026',
    # 'hicnn-synthetic-ensemble:hic033',

    # 'hicnn-real-ensemble:encode0',
    # 'hicnn-real-ensemble:encode1',
    # 'hicnn-real-ensemble:encode2',
    # 'hicnn-real-ensemble:hic026',
    # 'hicnn-real-ensemble:hic033',


    # 'hicnn-random-noise:encode0',
    # 'hicnn-random-noise:encode1',
    # 'hicnn-random-noise:encode2',
    # 'hicnn-random-noise:hic026',
    # 'hicnn-random-noise:hic033',

    # 'hicnn-gaussian-noise:encode0',
    # 'hicnn-gaussian-noise:encode1',
    # 'hicnn-gaussian-noise:encode2',
    # 'hicnn-gaussian-noise:hic026',
    # 'hicnn-gaussian-noise:hic033',

    # 'hicnn-uniform-noise:encode0',
    # 'hicnn-uniform-noise:encode1',
    # 'hicnn-uniform-noise:encode2',
    # 'hicnn-uniform-noise:hic026',
    # 'hicnn-uniform-noise:hic033',

]


common.evaluate('gm12878_real-world-dataset_results', 
        os.path.join(DATA_DIRECTORY,'GM12878_rao_et_al/'),
        upscaled_experiments,
        target_cutoff=globals.dataset_cutoff_values['GM12878_rao_et_al'],
        upscaled_cutoffs=upscaled_experiments_cutoffs,
        experiment_names=upscaled_experiment_names,
        steps=steps,
        full_results=True      
)

upscaled_experiments = [
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-encode0_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-encode2_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-hic026_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-synthetic-ensemble_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-real-ensemble_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-random-noise_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-gaussian-noise_hic057/'),
    os.path.join(DATA_DIRECTORY, 'IMR90_hicnn-uniform-noise_hic057/'),

]
upscaled_experiments_cutoffs = [globals.dataset_cutoff_values['deeplearning_upscaled']]*len(upscaled_experiments)

upscaled_experiment_names = [
    'hicnn-encode0:hic057',
    'hicnn-encode2:hic057',
    'hicnn-hic026:hic057',
    'hicnn-synthetic-ensemble:hic057',
    'hicnn-real-ensemble:hic057',
    'hicnn-random-noise:hic057',
    'hicnn-gaussian-noise:hic057',
    'hicnn-uniform-noise:hic057',
]

# common.evaluate('imr90_real-world-dataset_results', 
#         os.path.join(DATA_DIRECTORY,'IMR90_rao_et_al/'),
#         upscaled_experiments,
#         target_cutoff=globals.dataset_cutoff_values['IMR90_rao_et_al'],
#         upscaled_cutoffs=upscaled_experiments_cutoffs,
#         experiment_names=upscaled_experiment_names,
#         steps=steps,
#         full_results=True      
# )



upscaled_experiments = [
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-encode0_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-encode2_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-hic026_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-synthetic-ensemble_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-real-ensemble_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-random-noise_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-gaussian-noise_hic073/'),
    os.path.join(DATA_DIRECTORY, 'K562_hicnn-uniform-noise_hic073/'),

]
upscaled_experiments_cutoffs = [globals.dataset_cutoff_values['deeplearning_upscaled']]*len(upscaled_experiments)

upscaled_experiment_names = [
    'hicnn-encode0:hic073',
    'hicnn-encode2:hic073',
    'hicnn-hic026:hic073',
    'hicnn-synthetic-ensemble:hic073',
    'hicnn-real-ensemble:hic073',
    'hicnn-random-noise:hic073',
    'hicnn-gaussian-noise:hic073',
    'hicnn-uniform-noise:hic073',
]

# common.evaluate('k562_real-world-dataset_results', 
#         os.path.join(DATA_DIRECTORY,'K562_rao_et_al/'),
#         upscaled_experiments,
#         target_cutoff=globals.dataset_cutoff_values['K562_rao_et_al'],
#         upscaled_cutoffs=upscaled_experiments_cutoffs,
#         experiment_names=upscaled_experiment_names,
#         steps=steps,
#         full_results=True      
# )
