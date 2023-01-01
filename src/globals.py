"""
    Path to the data directory that contains all the .hic files and will contain all the generated files
"""
DATA_DIRECTORY = '/media/murtaza/ubuntu/hic_data/'
TEMP_DATA_DIRECTORY = '/home/murtaza/Documents/hic_upscaling_evaluation/temp/'
QC_PARAMETERS_FILE = '/media/murtaza/ubuntu/updated_hic_data/data/3dqc_config/parameters.txt'
"""
    This dictionary defines the dataset partitions for hg19 assembly HiC datasets
"""
dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
    'valid': [8, 9, 10, 11],
    'test': [19, 20, 21, 22],
    'all': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}


"""
    Supported HiC dataset file formats
"""
supported_formats = [
    'hic',
    'cool',
    'npz'
]

'''
    Cutoff percentage used for normalizing the hic matrices
'''
CUTOFF_PERCENTILE = 99.95


'''
    Pre-Computed cutoff values for datasets
'''
dataset_cutoff_values = {
    'GM12878_rao_et_al': 255,
    'GM12878_rao_et_al_replicate': 255,
    'IMR90_rao_et_al': 255,
    'K562_rao_et_al': 255,
    'downsampled_8': 140,
    'downsampled_16': 100,
    'downsampled_25': 80,
    'downsampled_50': 50,
    'downsampled_100': 25,
    'GM12878_encode0': 90,
    'GM12878_encode1': 87,
    'GM12878_encode2': 96,
    'GM12878_hic026': 143,  
    'GM12878_hic033': 29, 
    'IMR90_hic057': 68,   
    'K562_hic073': 28,
    'deeplearning_upscaled': 255,
    'GM12878_synthetic16': 100,
    'GM12878_synthetic25': 80,
    'GM12878_synthetic44': 50,
    'GM12878_synthetic50': 50,
    'GM12878_synthetic100': 20,
    'IMR90_synthetic16': 100,
    'IMR90_synthetic25': 80,
    'IMR90_synthetic44': 50,
    'IMR90_synthetic50': 50,
    'IMR90_synthetic100': 20,
    'K562_synthetic16': 100,
    'K562_synthetic25': 80,
    'K562_synthetic44': 50,
    'K562_synthetic50': 50,
    'K562_synthetic100': 20
}


'''
    DNA file path
'''
#na_motif_file_paths = 


'''
    Model configurations
'''
configurations = {
    'smoothing' : {
        'model_name': 'gaussian_smoothing',
        'configurations': {
            'kernel_size': (17,17),
            'sigma': 7
        }

    },
    'hicplus' : {},
    'hicnn16' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_16'
        }
    },
    'hicnn25' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_25'
        }
    },
    'hicnn50' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_50'
        }
    },
    'hicnn100' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_100'
        }
    },
    'hicnn-gaussian' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_gaussian'
        }
    },
    'hicnn-random' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_random'
        }
    },
    'hicnn-uniform' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_uniform'
        }
    },
    'hicnn-lrc' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_lrc'
        }
    },
    'hicnn-lrc-ensemble' : {
        'model_name': 'hicnn',
        'configurations': {
            'weights': 'weights/HiCNN/hicnn_lrc-ensemble'
        }
    },
    'deephic16': {
        'model_name': 'deephic',
        'configurations': {
            'weights': 'weights/DeepHiC/deephic16'
        }
    },
    'deephic25': {
        'model_name': 'deephic',
        'configurations': {
            'weights': 'weights/DeepHiC/deephic25'
        }
    },
    'deephic50': {
        'model_name': 'deephic',
        'configurations': {
            'weights': 'weights/DeepHiC/deephic50'
        }
    },
    'deephic100': {
        'model_name': 'deephic',
        'configurations': {
            'weights': 'weights/DeepHiC/deephic100'
        }
    },
    'vehicle': {}
}


'''
    3D Reconstruction parameters
'''

PIECE_SIZE = 200
JAR_LOCATION = "other_tools/3DMax/examples/3DMax.jar"




'''
    Loop Analysis parameters
'''
DNA_MOTIF_FILE_PATH = '/media/murtaza/ubuntu/updated_hic_data/data/motifs/CTCF.bed'
CHIA_PET_FILE = '/media/murtaza/ubuntu/updated_hic_data/data/chia_pet_datasets/K562_CTCF_mediated_chromatin_interactions.bed'


'''
    Conserved Feature Analysis
'''
FEATURES_FOLDER_PATH = '/media/murtaza/ubuntu/updated_hic_data/data/shared_features/'