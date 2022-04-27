from math import trunc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F





def compactM(matrix, compact_idx, verbose=False):
    """
        Compacting matrix according to the index list.
        @params: matrix <np.array> Full sized matrix, that needs to be compressed
        @params: compact_idx <list> Indexes of rows that contain data
        @params: verbose <boolean> Debugging print statements
        @returns: <np.array> Condesed matrix with zero arrays pruned 
    """
    compact_size = len(compact_idx)
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    
    return result

def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """
        Spreading matrix according to the index list (a reversed operation to compactM).
        @params: c_mat <np.array> condensed matrix
        @params: compact_idx <list> positions of compact indexes
        @params: full_size <int> full size of the matrix
        @params: convert_int <boolean>, convert all the values in the matrix to ints
        @params: verbose <boolean>, Debugging print statements
    """
    result = np.zeros((full_size, full_size)).astype(c_mat.dtype)
    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!' )
    for i, s_idx in enumerate(compact_idx):
        result[s_idx, compact_idx] = c_mat[i]
    return result


# dividing method
def divide(mat, chr_num, chunk_size=40, stride=28, bound=201, padding=True, species='hsa', verbose=False):
    """
        @params: mat <np.array> HiC matrix that needs to be chunked up
        @params: chr_num <string> Chromosome number of the input HiC matrix
        @params: chunk_size <int> Size of the chunks we divide the main matrix into
        @params: stride <int> Strides we make when making chunks
        @params: bound <int> Maximum distance we go from the diagonal to create chunks
        @params: padding <boolean> Pad to fix the issue that arises because of strides
        @params: verbose <boolean> Debugging print statements
        @returns: list<np.array>, list first return is chunked up matrices in a list and second 
                return contains the positions 
    """
    chr_str = str(chr_num)
    result = []
    index = []
    size = mat.shape[0]
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len,pad_len), (pad_len,pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j)<=bound and i+chunk_size<height and j+chunk_size<width:
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(f'[Chr{chr_str}] Dividing HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


def together(matlist, 
             indices, 
             corp=0,  
             tag='HiC',
             verbose=False):
    """
        @params: matlist <list<np.array>
        @params: indices <list>
        @params: corp <int>
        @params: tag <string>
        @params: verbose <boolean>
        @returns: 
    """
    chr_nums = sorted(list(np.unique(indices[:,0])))
    # convert last element to str 'X'
    
    if verbose: print(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        loci = np.where(indices[:,0] == n)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0,1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[i:i+h, j:j+w] = sub
        results[n] = full_mat
    return results

def pooling(mat, 
            scale, 
            pool_type='max', 
            return_array=False, 
            verbose=True):
    """
        @params:
        @params:
        @params:
        @params:
        @params:
    """
    mat = torch.tensor(mat).float()
    if len(mat.shape) == 2:
        mat.unsqueeze_(0) # need to add channel dimension
    if pool_type == 'avg':
        out = F.avg_pool2d(mat, scale)
    elif pool_type == 'max':
        out = F.max_pool2d(mat, scale)
    if return_array:
        out = out.squeeze().numpy()
    if verbose:
        print('({}, {}) sized matrix is {} pooled to ({}, {}) size, with {}x{} down scale.'.format(*mat.shape[-2:], pool_type, *out.shape[-2:], scale, scale))
    return out




def matrix_division(chr_num, matrix_file_path, cut_off,
                    chunk=40, stride=40, bound=201, verbose=True, 
                    compact=True):
    
    """
        Matrix Division, divides a NxN matrix into smaller matrices with defined coverage around them
        @params: chr_num <string> or <int> string if it is refering to chromosome X else its an int
        @params: matrix_file_path <string> path to the stored matrix
        @params: cut_off <int> Preprocessing of matrix 
        @params: chunk <int> dimension of the chunk (both dimensions are the same) :: Defaults to 40
        @params: stride <int> stride when picking out chunks :: Defaults to 40 (means no part of the data is skipped)
        @params: bound <int> megabases of data to include in the chunk :: Defaults to 201
    """

    
    matrix_data = np.load(matrix_file_path, allow_pickle=True)
    
    compaction_indices = matrix_data['compact']
    hic_data_key = list(filter(lambda x: 'hic' in x ,list(matrix_data.keys())))[0]

    full_size = matrix_data[hic_data_key].shape
    
    if verbose: print("Read a matrix of size: {}".format(full_size))
    if verbose: print("Number of compact indices: {}".format(len(compaction_indices)))
    
    # Compaction
    matrix_data = matrix_data[hic_data_key]
    
    if compact:
        matrix_data = compactM(matrix_data, compaction_indices)

    if verbose: print("Matrix has max: {} and min: {}".format(np.max(matrix_data), np.min(matrix_data)))

    if verbose: 
        print("Clamping and rescaling with cutoff value: {}".format(cut_off))
    
    #Clamping
    matrix_data = np.minimum(cut_off, matrix_data)
    matrix_data = np.maximum(matrix_data, 0)

    #Rescaling
    matrix_data = matrix_data / np.max(matrix_data)
    
    # Divide the matrix as specified in the arguments
    divided_matrices, indexes = divide(matrix_data, chr_num, chunk, stride, bound, verbose=verbose)

    return divided_matrices, indexes




def normalize(matrix_file_path, cut_off, compact_idxs=[], compact=True, verbose=False):
    """
        Matrix Division, divides a NxN matrix into smaller matrices with defined coverage around them
        @params: matrix_file_path <string> path to the stored matrix
        @params: cut_off <int> Preprocessing of matrix 
        @params: compact <bool> To remove empty rows or not
        @params: verbose <boo> Show print statements
        @returns <np.array> A normalized matrix 
    """

    
    matrix_data = np.load(matrix_file_path, allow_pickle=True)
    
    if compact_idxs == []:
        compact_idxs = matrix_data['compact']
    
    hic_data_key = list(filter(lambda x: 'hic' in x ,list(matrix_data.keys())))[0]

    full_size = matrix_data[hic_data_key].shape
    
    if verbose: print("Read a matrix of size: {}".format(full_size))
    if verbose: print("Number of compact indices: {}".format(len(compact_idxs)))
    
    # Compaction
    matrix_data = matrix_data[hic_data_key]
    
    if compact:
        matrix_data = compactM(matrix_data, compact_idxs)

    if verbose: print("Matrix has max: {} and min: {}".format(np.max(matrix_data), np.min(matrix_data)))

    if verbose: 
        print("Clamping and rescaling with cutoff value: {}".format(cut_off))
    
    #Clamping
    matrix_data = np.minimum(cut_off, matrix_data)
    matrix_data = np.maximum(matrix_data, 0)

    #Rescaling
    matrix_data = matrix_data / np.max(matrix_data)

    return matrix_data