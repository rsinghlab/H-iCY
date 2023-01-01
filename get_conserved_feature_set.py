import os, math
import numpy as np

from src.globals import DATA_DIRECTORY, dataset_partitions, FEATURES_FOLDER_PATH
from src.evaluation.biological_analysis import check_if_cooler_file_exists
from src.evaluation.downstream_analysis import get_cell_shared_feature_set


HIC_EXTRACTED_DATA_PATH = '/media/murtaza/ubuntu/hic_data/chromosome_files/synthetic'

high_res_files = list(filter(lambda x: 'rao_et_al' in x ,os.listdir(HIC_EXTRACTED_DATA_PATH)))
high_res_file_full_paths = list(map(lambda x: os.path.join(HIC_EXTRACTED_DATA_PATH, x), high_res_files))
get_cell_shared_feature_set(high_res_file_full_paths, 'loops', rp=3)
get_cell_shared_feature_set(high_res_file_full_paths, 'borders', rp=5)
get_cell_shared_feature_set(high_res_file_full_paths, 'hairpins', rp=2)












































