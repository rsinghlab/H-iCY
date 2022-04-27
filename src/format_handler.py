from src.hic_format_reader import hic_reader
from src import globals, utils
import os
import numpy as np
import math
import pandas as pd
import cooler
import gzip
from src.matrix_ops import ops

###################################### .npz file format reader ####################################################################
def read_npz_file(numpy_file_path):
    # loading the file, we always assume correct full input file path
    dense_hic_file = np.load(numpy_file_path, allow_pickle=True)

    if 'hic' in list(dense_hic_file.keys()):
        data_key = 'hic'
    else:
        data_key = 'deephic'



    h, w = dense_hic_file[data_key].shape

    if h != w:
        print("The input file doesnt contain a matrix that has width equals to its height")
        exit(1)
    # Process the dense data
    dense_data = dense_hic_file[data_key]

    return dense_data, data_key

def normalize_dense_matrix(dense_data, cutoff, upscale):
    # clamp 0 - cutoff
    if cutoff != -1:
        dense_data = np.minimum(cutoff, dense_data)
    
    dense_data = np.maximum(dense_data, 0)

    # Rescale
    dense_data = dense_data / np.max(dense_data)
    dense_data = dense_data*upscale

    dense_data = dense_data.astype(int)

    return dense_data

####################################### .hic to .npz conversion functions ############################################################

def get_hic_file_header(path_to_hic_file):
    '''
        Reads the header of the .hic file and returns it.
        @params: path_to_hic_file <string>, path to the .hic file
        @returns <dict> a dictionary that contains the statistics contained in the HiC file header
    '''
    return hic_reader.read_hic_header(path_to_hic_file)['Attributes']['statistics']



def extract_npz_chromosomal_files_from_hic_file(path_to_hic_file, output_path, 
                                                resolution=10000, normalization='KR',
                                                chromosomes='all', verbose=False):
    """
        Read .hic file and returns a list of parsed chromosome files
    """
    if verbose:
        print("Parsing out {} chromosomes at resolution {} from file at {} with {} normalization!".format(chromosomes, resolution, path_to_hic_file, normalization))

    utils.create_entire_path_directory(output_path)
    
    # Extract all chromosomes
    for chr_num in globals.dataset_partitions[chromosomes]:
        try:
            mat, compact_idx = hic_reader.matrix_extract(chr_num, chr_num, 
                                    resolution, path_to_hic_file, 
                                    normalization)
        except:
           print("Faulty Chromosome data, skipping!")
           continue
        
        file_path = os.path.join(output_path, 
                              'chr{}.npz'.format(chr_num))
    
        np.savez_compressed(file_path, hic=mat, compact=compact_idx)
        print('Saving file:', file_path, ' of shape {}'.format(mat.shape))
    


####################################### .npz to .cool conversion functions ###########################################################

def create_genomic_bins(
    chromosome_name,
    resolution,
    size,
    output_type='bed',
    max_size_type ='bin'):
    
    """
    The only currently supported type is 'bed' format which is chromosome_id, start, end
    So the function requires input of 'chromosome_name' chromosome name and 'resolution' resolution of of the file. 
    This function also requires size of the chromosome to estimate the maximum number of bins
    """

    chr_names = []
    starts = []
    ends = []

    if max_size_type == 'bin':
        number_of_bins = size
    elif max_size_type == 'size':
        number_of_bins = math.ceil(size/resolution)
    else:
        print("Invalid max_size_type value provided. This parameter can only take 'bin' or 'size'")
        exit(1)
    
    start = 0
    end = resolution
    
    for _ in range(number_of_bins):
        chr_names.append(chromosome_name)
        starts.append(start)
        ends.append(end)
        start = end
        end += resolution

    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }

    bins = pd.DataFrame(data=bins)

    return bins



def create_genomic_pixels(
    dense_matrix, 
    output_type='bed',):
    """
        Converts a dense matrix into a .bed style sparse matrix file
        @params: dense_matrix <np.array>, input dense matrix
        @params: output_type <string>, output type, currently only supported style is bed style
    """
    width, height = dense_matrix.shape

    bin_ones = []
    bin_twos = []
    counts = []

    for i in range(width):
        for j in range(height):
            if dense_matrix[i][j] == 0:
                continue
            if i > j:
                continue
            if dense_matrix[i][j] < 0:
                count = 0
            else:
                count =  dense_matrix[i][j]


            bin_ones.append(i)
            bin_twos.append(j)
            counts.append(dense_matrix[i][j])
    
    pixels = {
        'bin1_id': bin_ones,
        'bin2_id': bin_twos,
        'count': counts
    }

    pixels = pd.DataFrame(data=pixels)

    return pixels

def balance_cooler_file(cooler_file_path):
    """
        This function uses cooler utility to balance a .cool matrix
        
        @params: cooler_file_path <string>, path to the cooler file to balance
    """
    cmd = 'cooler balance {}'.format(cooler_file_path)
    
    os.system(cmd)



def create_cool_file_from_numpy(
    numpy_file_path,
    output_file_path,
    chromosome_name,
    resolution = 10000,
    upscale=255,
    cutoff=-1
    ):

    '''
        @params: <string> numpy_file_path file path of the numpy array that needs to be converted to cooler file
        @params: <string> output_file_path full path to the output folder that contains the output file
        @chromosome_name: <string> name of the chromosome for example: chr21 
        @resolution: <int> resolution at which we have sampled the input dense array
    '''
    # loading the file, we always assume correct full input file path
    
    dense_data, data_key = read_npz_file(numpy_file_path)

    dense_data = normalize_dense_matrix(dense_data, cutoff, upscale)
    

    h, w = dense_data.shape

    dense_hic_file_genomic_bins = create_genomic_bins(chromosome_name, resolution, h, max_size_type='bin')
    dense_hic_file_pixels_in_bins = create_genomic_pixels(dense_data)

    # This generates a cooler file in the provided output file path
    cooler.create_cooler(output_file_path, dense_hic_file_genomic_bins, dense_hic_file_pixels_in_bins,
                        dtypes={"count":"int"}, 
                        assembly="hg19")

    balance_cooler_file(output_file_path)
    



    
def create_contact_map(matrix, output_path, number_of_bins, resolution, chr_name):
    with open(output_path, 'w') as f:
        for i in range(number_of_bins):
            for j in range(number_of_bins):
                contact_value = matrix[i, j]
                ith_chr_index = i*resolution
                jth_chr_index = j*resolution
                if contact_value != 0:
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(
                        chr_name, 
                        ith_chr_index,
                        chr_name,
                        jth_chr_index,
                        contact_value
                    ))
                
    output_path = utils.compress_file(output_path, clean=True)
    print("Contact File generated at {}".format(output_path))
    


def create_sparse_matrix_files_from_numpy(
    numpy_file_path,
    output_directory_path,
    chromosome_name,
    resolution = 10000,
    upscale=255,
    cutoff=-1,
    chunk=200,
    stride=200,
    bound=190,
    compact_indexes=[]):
    '''
        @params: <string> numpy_file_path file path of the numpy array that needs to be converted to cooler file
        @params: <string> output_directory_path full path to the output folder that will contain all the generated files
        @params: chromosome_name: <string> name of the chromosome for example: chr21 
        @params: resolution: <int> resolution at which we have sampled the input dense array,
        @params: upscale <int>, matrix division parameter
        @params: cutoff <int>, matrix division parameter
        @params: chunk <int>, matrix division parameter
        @params: stride <int>, matrix division parameter
        @params: bound <int>, matrix division parameter
        @params: compact_indexes <list> matrix division parameter
    '''
    # loading the file, we always assume correct full input file path
    dense_data, _ = read_npz_file(numpy_file_path)
    dense_data = normalize_dense_matrix(dense_data, cutoff, upscale)

    dense_data = ops.compactM(dense_data, compact_indexes)

    
    utils.create_entire_path_directory(output_directory_path)
    utils.clean_up_folder(output_directory_path)

    
    divided_data, indices = ops.divide(dense_data, chromosome_name, chunk_size=chunk, stride=stride, bound=bound)
    
    # Create a separate folder for 
    for sample, index in zip(divided_data, indices):
        contact_file_path = os.path.join(output_directory_path, '{}_{}.contact.map'.format(index[2], index[3]))
        sample = sample[0, :, :]
        
        create_contact_map(sample, contact_file_path,
                        chunk,
                        resolution,
                        chromosome_name 
                    )
        




####################################### .npz to .fithic conversion functions ###########################################################

def create_fithic_files(numpy_file_path, 
    output_directory_path, 
    chr_name,
    resolution = 10000,
    cutoff=-1,
    upscale=255,
    compact_indexes=[]):

    dense_data, _ = read_npz_file(numpy_file_path)
    dense_data = normalize_dense_matrix(dense_data, cutoff, upscale)

    dense_data = ops.compactM(dense_data, compact_indexes)

    utils.create_entire_path_directory(output_directory_path)

    #interaction_counts_file = open(os.path.join(output_directory_path, '{}_interactions_count.fithic'.format(chr_name)), 'w')
    
    fragments = {}

    # Initialize all fragments 
    for i in range(dense_data.shape[0]):
        mid = int((i*resolution) + (resolution/2))

        fragments[i] = {
            'chr_num': chr_name,
            'start': i*resolution,
            'mid': mid,
            'tcc': 0
        }

    with open(os.path.join(output_directory_path, 'interactions_count.fithic'), 'w') as f: 
        for i in range(dense_data.shape[0]):
            for j in range(dense_data.shape[1]): 
                
                # Zero values and lower traingular matrix are ignored
                if dense_data[i][j] == 0 and i > j: 
                    continue
                
                # i_start = i*resolution
                # i_end = i_start + resolution
                # i_mid = int((i_start+i_end)/2)

                # j_start = j*resolution
                # j_end = j_start + resolution
                # j_mid = int((j_start+j_end)/2)
                
                # f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                #     chr_name,
                #     i_mid,
                #     chr_name,
                #     j_mid,
                #     dense_data[i][j]
                # ))

                # update fragment dictionary
                fragments[i]['tcc'] += int(dense_data[i][j])
                fragments[j]['tcc'] += int(dense_data[i][j])

                f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    fragments[i]['chr_num'],
                    fragments[i]['mid'],
                    fragments[j]['chr_num'],
                    fragments[j]['mid'],
                    int(dense_data[i][j])
                ))

    utils.compress_file(os.path.join(output_directory_path, 'interactions_count.fithic'), True)

    with open(os.path.join(output_directory_path, 'fragment_mappability.fithic'.format(chr_name)), 'w') as f:
        for i in range(dense_data.shape[0]):
            fragment = fragments[i]
            # total_count = 0
            # i_start = i*resolution
            # i_end = i_start + resolution
            # i_mid = int((i_start+i_end)/2)

            # for j in range(dense_data.shape[1]):
            #     total_count += dense_data[i][j]

            # print(chr_name, i_start, i_mid, total_count, fragment)
            
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                fragment['chr_num'],
                fragment['start'],
                fragment['mid'],
                fragment['tcc'],
                1
            ))
    
    utils.compress_file(os.path.join(output_directory_path, 'fragment_mappability.fithic'), True)

