import numpy as np
import os
from src import globals
from src.matrix_ops.ops import normalize
import sys
from src.evaluation.correlation_analysis import pearsons
import matplotlib.pyplot as plt



def get_diagonals(file_path, cut_off, compact=[], num_diagonals=200, verbose=False):
    if cut_off == -1:
        cut_off = sys.maxsize
    matrix = normalize(file_path, cut_off, compact)
    diagonals = [matrix.diagonal(diagonal) for diagonal in range(num_diagonals)]

    return diagonals





def compute_distribution_analysis_on_experiment_directory(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    dataset='test',
    full_results=False,
    verbose=False
):
    compiled_correlations = []

    for chromosome_id in globals.dataset_partitions[dataset]:
        base_file = os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id))
        target_file = os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id))
        if not (os.path.exists(base_file) and os.path.exists(target_file)):
            print("Missing Chromosome files")
            continue

        
        comapct_indices_base =  np.load(base_file, allow_pickle=True)['compact']
        compact_indices_target = np.load(target_file, allow_pickle=True)['compact']

        compact_indexes = list(set.intersection(set(comapct_indices_base), set(compact_indices_target)))
        
        if verbose: print("Base file: {}\nTarget File: {}".format(base_file, target_file))
        
       
        
        base_diagonals = get_diagonals(base_file, base_cutoff, compact_indexes)
        target_diagonals = get_diagonals(target_file, target_cutoff, compact_indexes)
        

        compiled_correlations.append([pearsons(base_diagonals[idx], target_diagonals[idx]) for idx in range(len(base_diagonals))])

    compiled_correlations = np.array(compiled_correlations)

    
    
    compiled_correlations = np.mean(compiled_correlations, axis=0)

    return compiled_correlations
    



