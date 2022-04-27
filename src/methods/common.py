import numpy as np
import os
from tqdm import tqdm

from src.utils import data_info, dataloader, create_entire_path_directory
from src import globals
from src.methods import smoothing, HiCNN, DeepHiC
from src.matrix_ops import ops



def create_upscaling_function(conf):
    '''
        Takes in a configuration dictionary and generates the upscaling function
        @params: conf <dict>, configurations dictionary
        @returns: python function that upscales
    '''

    if conf['model_name'] == 'gaussian_smoothing':
        return smoothing.upscale(conf)
    elif conf['model_name'] == 'hicnn':
        return HiCNN.upscale(conf)
    elif conf['model_name'] == 'deephic':
        return DeepHiC.upscale(conf)
    else:
        print("Wrong model specifed, and there is no configuration file for it in the globals.py")
        exit(1)

def save_data(smoothed, compact, size, file):
    hic = ops.spreadM(smoothed, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, compact=compact)
    print('Saving file:', file)


def upscale(input_dataset, model, output_path):
    '''
        A wrapper function that takes in model name and upscales the provided dataset file
        @params: input_dataset <string> path to the input dataset
        @params: model <string> Name of the model
        @params: output_path <string> path of the output directory
    '''

    create_entire_path_directory(output_path)

    # Step 1: Load the dataset directory and convert into a dataloader format
    input_data = np.load(os.path.join(input_dataset), allow_pickle=True)

    # We need indices, compacts for ensuring we are not doing wasteful computations and have a method to reconstruct out chromosomes
    indices, compacts, sizes = data_info(input_data)
    # Generated the input loader
    input_loader = dataloader(input_data, batch_size=64)

    # Step 2: using the configurations create the upscaling function    
    configurations = globals.configurations[model]

    upscaling_func = create_upscaling_function(configurations)


    # Step 3: Upscale the data using the upscaling function provided
    result_data = []
    result_inds = []
    
    description_string = 'Upscaling using {}: '.format(configurations['model_name'])
    

    for batch in tqdm(input_loader, desc=description_string):
        data, inds = batch
        result = upscaling_func(data.numpy())
        result_data.append(result)
        result_inds.append(inds.numpy())

    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    
    upscaled = ops.together(result_data, result_inds, tag='Reconstructing: ')


    
    # Step 4: Store as chromosomal matrices
    def save_data_n(key):
        file = os.path.join(output_path, f'chr{key}.npz')
        save_data(upscaled[key], compacts[key], sizes[key], file)

    
    for key in compacts.keys():
        save_data_n(key)
































