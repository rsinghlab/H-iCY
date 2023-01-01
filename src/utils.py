import os
import numpy as np
import torch
import shutil
import gzip
from torch.utils.data import TensorDataset, DataLoader
from src import globals
from src.matrix_ops import ops


def create_entire_path_directory(path):
    """
        Given a path, creates all the missing directories on the path and 
        ensures that the given path is always a valid path
        @params: <string> path, full absolute path, this function can act wonkily if given a relative path
        @returns: None
    """
    
    path = path.split('/')
    curr_path = '/'
    for dir in path:
        curr_path = os.path.join(curr_path, dir)
        if os.path.exists(curr_path):
            continue
        else:
            os.mkdir(curr_path)


def compute_cutoff_value(hic_directory_path, percentile=99.95):
    avg_percentile = 0
    count = 0
    
    for chromo in range(1, 23):
        hic_file = os.path.join(hic_directory_path, 'chr{}.npz'.format(chromo))
        try:
            data = np.load(hic_file)
        except:
            print("Chromosome file {} missing, ignoring...".format(hic_file))
            
        condensed_data = ops.compactM(data['hic'], data['compact'])
        condensed_data = np.sort(condensed_data.flatten())
        condensed_data = condensed_data[np.where(condensed_data != 0)]
        
        
        if len(condensed_data > 0):
            avg_percentile += np.percentile(condensed_data, percentile)
            count += 1
    
    return avg_percentile/count
        

def downsample_chromsosome_file(input_path, output_path, downsampling_ratio, dataset='test'):
   
    for chromo in globals.dataset_partitions[dataset]:
        input_chrom_path = os.path.join(input_path, 'chr{}.npz'.format(chromo))
        output_chrom_path = os.path.join(output_path, 'chr{}.npz'.format(chromo))
        
        try:
            data = np.load(input_chrom_path)
            compact = data['compact']
            data = data['hic']

        except:
            print("Chromosome file {} missing, ignoring...".format(input_chrom_path))

        downsampled_data = ops.downsampling(data, downsampling_ratio)



        np.savez_compressed(output_chrom_path, hic=downsampled_data, compact=compact)

        




def data_info(data):
    '''
        Takes in a .npz read object and parse out all the individual objects from it
        @params: data <np.object>, numpy object 
        @returns indices, compacts and sizes
    '''
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes



def dataloader(data, batch_size=64):
    '''
        Converts the dataset into a dataloader object
        @params: data <np.array> 
        @params: batch_size <int>, batch size of the data loader object
        @params: shuffle <bool>, shuffle the dataset or no
        @returns: A data loader object
    '''
    inputs = torch.tensor(data['data'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def compress_file(file_path, clean=False):
    '''
        This function compresses the file found on the provided path
        @params: file_path <string>, path to the file that needs to be compressed
        @params: clean <bool>, to keep or remove the original file
        @returns: return the path of the generated .gzip file
    '''
    compressed_file_path = file_path+'.gz'
    with open(file_path, 'rb') as f_in, gzip.open(compressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if clean:
        if os.path.exists(file_path):
            os.remove(file_path)
    return compressed_file_path


def clean_up_folder(directory_path):
    for files in os.listdir(directory_path):
        path = os.path.join(directory_path, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)