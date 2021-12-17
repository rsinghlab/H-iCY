from src import globals, utils
import os
import numpy as np
from src.matrix_ops import ops


def add_gaussian_noise(data, mu=0, sigma=0.05):
    data = data + np.random.normal(mu, sigma, [data.shape[0], data.shape[1]]) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data

def add_uniform_noise(data, low=0, high=0.07):
    data = data + np.random.uniform(low, high, [data.shape[0], data.shape[1]]) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data


def add_random_noise(data, max=0.7):
    data = data + max*np.random.rand((data.shape[0], data.shape[1])) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data

def no_noise(data):
    return data

noise_sources = {
    'gaussian': add_gaussian_noise,
    'uniform': add_uniform_noise,
    'random': add_random_noise,
    'none':  no_noise
}

def dataset_divider(chromosome_id, 
                    base_file, 
                    target_file,
                    base_cutoff=255,
                    target_cutoff=100, 
                    chunk=40, 
                    stride=40, 
                    bound=201, 
                    compact_type='intersection',
                    verbose=False,
                    noise_type=None
                    ):

    # Loading the datafiles
    base_data = np.load(base_file)
    target_data = np.load(target_file)
    
    # For compact indices, there are three different ways this can go. 
    # 1) Completely ignore compact indices (Models might not work correctly with missing values in matrices)
    # 2) Take an intersection of compact indices and use them as default (default, seems to have smallest impact on comparisons down the line)
    # 3) Only use the target dataset's indices (len(target_compact_idx) > len(base_compact_idx) so in most cases we are just discarding some values in the base set)

    base_compact_idx = base_data['compact']
    target_compact_idx = target_data['compact']

    base_data = base_data['hic']
    target_data = target_data['hic']
    

    compact_indexes = None
    base_condensed_matrix = None
    target_condensed_matrix = None

    full_size = None
    # Ensure full sizes are the same
    if base_data.shape[0] == target_data.shape[0]:
        full_size = base_data.shape[0]
    else:
        print("Shapes of the datasets is not same, {} vs {}".format(base_data.shape, target_data.shape ))
        exit(1)

    if verbose: print("Compacting type: {}".format(compact_type))
    
    if compact_type == 'ignore':
        compact_indexes = []
        base_condensed_matrix = base_data
        target_condensed_matrix = target_data
            
    elif compact_type == 'intersection':
        compact_indexes = list(set.intersection(set(base_compact_idx), set(target_compact_idx)))
        base_condensed_matrix = ops.compactM(base_data, compact_indexes)
        target_condensed_matrix = ops.compactM(target_data, compact_indexes)

    elif compact_type == 'target':
        compact_indexes = target_compact_idx
        base_condensed_matrix = ops.compactM(base_data, compact_indexes)
        target_condensed_matrix = ops.compactM(target_data, compact_indexes)
    else:
        print("Invalid value for variable compact_type, please choose one of [ignore, intersection, target].")
        exit(1)
    
    
    if verbose: print("Working with matrix with sizes, Base: {} Target: {}".format(base_condensed_matrix.shape, target_condensed_matrix.shape))
    
    
    if base_condensed_matrix.shape == (0,0):
        return -1
    
    # Clamping step (this ensures that the matrix values are between 0 and cutoff value)
    base_condensed_matrix = np.minimum(base_cutoff, base_condensed_matrix)
    base_condensed_matrix = np.maximum(base_condensed_matrix, 0)
    
    target_condensed_matrix = np.minimum(target_cutoff, target_condensed_matrix)
    target_condensed_matrix = np.maximum(target_condensed_matrix, 0)
    
    # Rescaling
    base_condensed_matrix = base_condensed_matrix / base_cutoff
    target_condensed_matrix = target_condensed_matrix / target_cutoff
    
    if not noise_type:
        base_condensed_matrix = noise_sources[noise_type](base_condensed_matrix)

    div_target, div_target_inds = ops.divide(target_condensed_matrix, chromosome_id, chunk, stride, bound)
    
    div_target = ops.pooling(div_target, 1, pool_type='max', verbose=False).numpy()
    
    div_base, _ = ops.divide(base_condensed_matrix, chromosome_id, chunk, stride, bound, verbose=False)

    return chromosome_id, div_base, div_target, div_target_inds, compact_indexes, full_size








def create_dataset_file( base_files_path, 
                         target_files_path, 
                         output_directory,
                         dataset_name,
                         base_cutoff, 
                         target_cutoff,
                         dataset='test',  
                         chunk=40, stride=40,
                         bound=201, 
                         compact_type='intersection',
                         noise_type='none',
                         verbose=False):
    """
        This function takes in a base file and target chromosome files path directory and creates a dataset file that can be used
        by deep learning methods to upscale the base chunks
        @params: base_files_path <string> path to base chromosome files
        @params: target_files_path <string> path to target chromosome files
        @params: output_directory <string> path to save the generated dataset
        @params: dataset_name <string> name of the dataset (goes as the first part of the dataset name string)
        @params: base_cutoff <int> Cutoff value for base chromosome files
        @params: target_cutoff <int> Cutoff value for target chromosome files
        @params: dataset <string> Dataset type, (test, valid or train)
        @params: chunk <int> chunk size of the dataset
        @params: stride <int> stride for the division function
        @params: bound <int> Max bounds around the diagonal 
        @params: compact_type <string> Compaction strategy (intersection, ignore, target)
        @params: noise_type <string> can take value Gaussian, Uniform or Random,  deafults to None. 
        @params: verbose <boolean> Debugging print statements
        @returns: None
    """
    utils.create_entire_path_directory(output_directory)
    
    results = []
    if verbose: print("Creating a {} dataset with chunk size {} stride {} and bound value {}. With cutoff values for base {} and target {} and compaction strategy {}".format(
        dataset,
        chunk,
        stride,
        bound,
        base_cutoff,
        target_cutoff,
        compact_type
    ))
    
    for chromosome_id in globals.dataset_partitions[dataset]:
        base_file = os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id))
        target_file = os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id))
        if verbose: print("Base file: {}\nTarget File: {}".format(base_file, target_file))
        
        res = dataset_divider(
            chromosome_id, base_file, target_file,
            base_cutoff=base_cutoff, target_cutoff=target_cutoff,
            chunk=chunk, stride=stride, 
            bound=bound, compact_type=compact_type,
            verbose=verbose,
            noise_type=noise_type
        )
        if res == -1:
            print("No valid data, ignoring the file")
            continue
        
        results.append(res)
    
    data = np.concatenate([r[1] for r in results])
    target = np.concatenate([r[2] for r in results])
    inds = np.concatenate([r[3] for r in results])
    compacts = {r[0]: r[4] for r in results}
    sizes = {r[0]: r[5] for r in results}
    
    filename = '{}_c{}_s{}_b{}_{}.npz'.format(
        dataset_name, chunk, stride, bound, dataset
    )
    output_path = os.path.join(output_directory, filename)
    print('Saving file:', output_path)
    np.savez_compressed(output_path, data=data, target=target, inds=inds, compacts=compacts, sizes=sizes)


