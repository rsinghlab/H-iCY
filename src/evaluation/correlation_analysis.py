import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.special import kl_div
from scipy.stats import entropy as ent

from src.matrix_ops import ops
from src import globals
import json


def invalid_comparisons(x, y):
    if (x.flatten() == y.flatten()).all():
        return True
    if np.all(x == x[0]) or np.all(y == y[0]):
        return True
    
    return False

def mse(x, y):
    #print("MSE")
    if invalid_comparisons(x, y):
        return -100
    return mean_squared_error(x, y)

def mae(x, y):
    #print("MAE")
    if invalid_comparisons(x, y):
        return -100
    return mean_absolute_error(x, y)

def psnr(x, y):
    #print("PSNR")
    if invalid_comparisons(x, y):
        return -100
    
    data_range = np.max(x) - np.min(x)
    err = mean_squared_error(x, y)
    if err == 0:
        return 100
    else:
        return 10*np.log10((data_range**2)/err)

def ssim(x, y):
    #print("SSIM")
    if invalid_comparisons(x, y):
        return -100
    data_range = y.max()-y.min()
    if (x.flatten() == y.flatten()).all():
        return -100
    return structural_similarity(x, y, data_range=data_range)

def pearsons(x, y):
    #print("PCC")
    if invalid_comparisons(x, y):
        return -100
    
    r, p_value = pearsonr(x.flatten(), y.flatten())
    return r

def spearmans(x, y):
    #print("SCC")
    if invalid_comparisons(x, y):
        return -100
    
    r, p_value = spearmanr(x.flatten(), y.flatten())
    return r

def same(x, y):
    if (x.flatten() == y.flatten()).all():
        return 1
    return 0


def kl_divergence(x, y, epsilon=0.00001, upscale=255):
    if invalid_comparisons(x, y):
        return -100

    x = x + epsilon
    y = y + epsilon

    x = x*upscale
    y = y*upscale

    return kl_div(x, y)


def entropy(x, y, upscale=255):
    if invalid_comparisons(x, y):
        return -100
    x = x*upscale
    y = y*upscale
    
    x = x.astype(int)
    y = y.astype(int)

    return ent(x.flatten(), y.flatten())


def mutual_information(x, y, upscale=255):
    if invalid_comparisons(x, y):
        return -100
    x = x*upscale
    y = y*upscale
    
    x = x.astype(int)
    y = y.astype(int)

    return normalized_mutual_info_score(x.flatten(), y.flatten())


def ks_test(x, y, upscale=255):
    if invalid_comparisons(x, y):
        return -100
    
    x = np.sort(x.flatten())
    y = np.sort(x.flatten())
    x = x*upscale
    y = y*upscale
    
    return ks_2samp(x, y)[0]


list_of_eval_funcs = {
    'MSE': mse,
    'MAE': mae,
    'PSNR': psnr,
    'SSIM': ssim,
    'PCC': pearsons,
    'SCC': spearmans,
    'MI': mutual_information,
}

def evaluate(y, y_bar, func):
    """
        A wrapper function on list of all chunks and we apply the func that can be 
            [mse, mae, psnr, ssim, pearsons, spearmans or same]
            
        @params: y <np.array>, Base Chunks
        @params: y_bar <np.array> Predicted Chunks
        @returns <np.array>, list : all errors and statistics of the errors.
    """
    errors = list(map(func, y[:,0,:,:], y_bar[:,0,:,:]))
    return {
        'errors': errors, 
        'stats': [np.mean(errors), np.std(errors), np.median(errors)]
    }




def compute_correlation_metrics_on_experiment_directory(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    dataset='test',
    full_results=False,
    verbose=False
):
    
    compiled_results = {}

    for chromosome_id in globals.dataset_partitions[dataset]:
        base_file = os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id))
        target_file = os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id))
        if verbose: print("Base file: {}\nTarget File: {}".format(base_file, target_file))
        
        print(chromosome_id, base_file, base_cutoff)


        y, _ = ops.matrix_division(chromosome_id, base_file, base_cutoff, 200, 200, 190, verbose=False)
        y_bar, _ = ops.matrix_division(chromosome_id, target_file, target_cutoff, 200, 200, 190, verbose=False)
        
        for key, eval_func in list_of_eval_funcs.items():
            mean_value = evaluate(y_bar, y, eval_func)['stats'][0]
            if key not in compiled_results.keys():
                compiled_results[key] = []
            
            compiled_results[key].append(mean_value)
   
    averaged_results = {}
    for key in compiled_results.keys():
        if full_results:
            averaged_results[key] = compiled_results[key]
        else:
            averaged_results[key] = np.mean(compiled_results[key])
    
    
    return '{}:{}'.format(experiment_name, averaged_results)
    
    
    
