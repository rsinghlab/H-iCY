import gzip
import numpy as np
import os
from PIL import Image
from pandas.core.indexes import base
from src.evaluation.biological_analysis import check_if_cooler_file_exists
from src.utils import create_entire_path_directory, clean_up_folder
from src import globals
from src.format_handler import read_npz_file, normalize_dense_matrix, create_fithic_files
from src.matrix_ops import ops
import glob
import subprocess

import torch
import torch.nn.functional as F
import tmscoring
import math 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


LOOP_RELAXATION_PARAMETER = 3
BORDER_RELAXATION_PARAMETER = 5
HAIRPIN_RELAXATION_PARAMETER = 2

# 3D reconstruction and structure comparison
def create_constrains_file_from_cooler_file(npz_file, output_directory,
    chromosome,
    upscale,
    cutoff,
    chunk,
    stride,
    bound,
    compact_indexes=[]
):
    '''
        @params: npz_file <string> path to npz file
        @params: base_directory <string> base directory where we dump the constraints 
        @params: upscale <int> used for division of matrix
        @params: chunk <int> used for division of matrix
        @params: stride <int> used for division of matrix
        @params: bound <int> used for division of matrix
        @params: compact_index <list> used for division of matrix
    '''



    dense_data, _ = read_npz_file(npz_file)
    dense_data = normalize_dense_matrix(dense_data, cutoff, upscale)
    
    dense_data = ops.compactM(dense_data, compact_indexes)

    divided_data, indices = ops.divide(dense_data, chromosome, chunk_size=chunk, stride=stride, bound=bound)

    # Create a separate folder for 
    for sample, index in zip(divided_data, indices):
        constraint_file_path = os.path.join(output_directory, '{}_{}_constraints.txt'.format(index[2], index[3]))
        sample = sample[0, :, :]
        with open(constraint_file_path, 'w') as f:
            for x in range(0, chunk):
                for y in range(x, chunk):
                    f.write(str(x) + str('\t') + str(y) + str('\t') + str(sample[x, y]) + str('\n'))


def create_parameters(constraints_path, parameters_output_path, output_path):
    '''
        @params: constraints_path <string>, path to folder that currently houses all the constraints
        @params: parameters_output_path <int>, path where to store the parameter files
        @params: output_path <string>, path to the folder to store the generated 3D models
    '''
    all_constraint_files = glob.glob(
        os.path.join(constraints_path, '*')
    )

    for constraint in all_constraint_files:
        suffix = constraint.split('/')[-1]
        

        stri = """NUM = 1
OUTPUT_FOLDER = {}
INPUT_FILE = {}
CONVERT_FACTOR = 0.6
VERBOSE = false
LEARNING_RATE = 1
MAX_ITERATION = 10000""".format(
            output_path,
            constraint
        )

        with open(os.path.join(parameters_output_path, suffix), 'w') as f:
            f.write(stri)

def create_3d_models(parameters_file_path):
    '''
        @params: parameters_file_path <string>, path to folder that contains all the parameters
    '''
    params = glob.glob(
        os.path.join(parameters_file_path, '*')
    )
    for param in params:
        subprocess.run("java -Xmx5000m -jar "+globals.JAR_LOCATION+" "+param, shell=True) 



def check_if_3d_reconstruction_folder_exists(base_directory, 
    chromosome,
    upscale,
    cutoff,
    chunk,
    stride,
    bound,
    compact_indexes,
    verbose=False
):
    '''
        This function checks if the folder that contains the 3D models exists, if it exists then it returns and if it doesnt it
        makes all the 3D models 
        @params: base_directory <string>, path to the base directory that contains all the 3D model files
        @params: chromosome <int>, the chromosome id
        @params: cutoff <int>, cutoff value to normalize the HiC matrix

    '''
    # if os.path.exists(base_directory):
    #     clean_up_folder(base_directory)
    
    # First of all we need to check if we have a cooler file in the folder or not, if not then make sure it exists
    npz_file = os.path.join('/'.join(base_directory.split('/')[:-1]), 'chr{}.npz'.format(chromosome))
    
    if not os.path.exists(npz_file):
        print("Npz file {} doesnt exist, can not proceede, exiting")
        exit(1)

    # Create all the folders
    create_entire_path_directory(os.path.join(base_directory, 'constraints'))
    create_entire_path_directory(os.path.join(base_directory, 'outputs'))
    create_entire_path_directory(os.path.join(base_directory, 'parameters'))
    clean_up_folder(os.path.join(base_directory, 'constraints'))
    clean_up_folder(os.path.join(base_directory, 'outputs'))
    clean_up_folder(os.path.join(base_directory, 'parameters'))
    
    
    # create constraints, parameter files and 3D models
    create_constrains_file_from_cooler_file(npz_file, os.path.join(base_directory, 'constraints'),
        chromosome,
        upscale,
        cutoff, 
        chunk,
        stride,
        bound,
        compact_indexes
    )
    # create the parameters for 3DMax and then run them
    create_parameters(os.path.join(base_directory, 'constraints'),
        os.path.join(base_directory, 'parameters'),
        os.path.join(base_directory, 'outputs')
    )

    create_3d_models(os.path.join(base_directory, 'parameters'))


def reconstruction_score(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    chunk,
    stride,
    bound,
    upscale,
    dataset='test',
    verbose=False,
    full_results=False
):
    compiled_results = {'3d_reconstruction_tmscore': []}
    
    for chromosome_id in globals.dataset_partitions[dataset]:
        base_3d_recon_files_path = os.path.join(base_files_path, 'chr{}_c{}_3dmod'.format(chromosome_id, chunk))
        target_3d_recon_files_path = os.path.join(target_files_path, 'chr{}_c{}_3dmod'.format(chromosome_id, chunk))
        if verbose: print("Working with base directory, {} and target directory {}".format(
            base_files_path,
            target_files_path
        ))

        # compact indexes are needed for the matrix compression in the later stage of the sparse matrix construction
        comapct_indices_base =  np.load(os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']
        compact_indices_target = np.load(os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']

        compact_indexes = list(set.intersection(set(comapct_indices_base), set(compact_indices_target)))


        check_if_3d_reconstruction_folder_exists(base_3d_recon_files_path, 
            chromosome_id,
            upscale,
            base_cutoff,
            chunk, 
            stride,
            bound, 
            compact_indexes,
            verbose 
        )
        check_if_3d_reconstruction_folder_exists(target_3d_recon_files_path, 
            chromosome_id,
            upscale,
            target_cutoff,
            chunk, 
            stride,
            bound, 
            compact_indexes,
            verbose 
        )

        base_3d_files = glob.glob(os.path.join(base_3d_recon_files_path, 'outputs/*.pdb'))
        
        target_3d_files = glob.glob(os.path.join(target_3d_recon_files_path, 'outputs/*.pdb'))
        
        scores = []
        for base_3d_file_path in base_3d_files:
            base_3d_file_name = base_3d_file_path.split('/')[-1]
            base_3d_file_identifier = '_'.join(base_3d_file_name.split('_')[:3])
            for target_3d_file_path in target_3d_files:
                if base_3d_file_identifier in target_3d_file_path:
                    break
            
            alignment = tmscoring.TMscoring(base_3d_file_path, target_3d_file_path)
            alignment.optimise() 

            score = alignment.tmscore(**alignment.get_current_values())
            scores.append(score)
        
        chr_score = np.mean(list(filter(lambda x: x!=np.nan, scores)))

    compiled_results['3d_reconstruction_tmscore'].append(chr_score)

    averaged_results = {}
    for key in compiled_results.keys():
        if full_results:
            averaged_results[key] = compiled_results[key]
        else: 
            averaged_results[key] = np.mean(compiled_results[key])
    
    return '{}:{}'.format(experiment_name, averaged_results)




def check_if_fithic_folder_exists(
    base_directory, 
    chromosome,
    upscale,
    cutoff,
    compact_indexes,
    verbose=False
):
    '''
        @params: base_directory <string>
        @params: chromosome <int> chromosome id
        @params: upscale <int> Normalization parameter
        @params: cutoff <int> Normalization parameters
        @params: compact_indexes <List> list of indexes that are informative 
        @params: verbose <bool> verbosity of this function
    '''
    # Npz file from parent directory to create th fithic file from
    npz_file = os.path.join('/'.join(base_directory.split('/')[:-1]), 'chr{}.npz'.format(chromosome))
    #print(npz_file)

    # if this file doesnt exists, then there is something seriously wrong with the file structure
    if not os.path.exists(npz_file):
        print("Npz file {} doesnt exist, can not proceede, exiting".format(npz_file))
        exit(1)
    

    create_entire_path_directory(base_directory)
    if verbose: print("creating FitHiC files in folder {}".format(base_directory))

    create_fithic_files(npz_file, base_directory, chromosome, cutoff=cutoff, upscale=upscale, compact_indexes=compact_indexes)
    fithic_analysis_command = 'fithic -i {} -f {} -r 10000 --upperbound {} -o {}'.format(
        os.path.join(base_directory, 'interactions_count.fithic.gz'),
        os.path.join(base_directory, 'fragment_mappability.fithic.gz'),
        2000000,
        base_directory
    )

    # Create the significant interaction files
    os.system(fithic_analysis_command)


def read_significant_interactions_fithic_file(file_path):
    file_data = gzip.open(file_path, 'r').read().decode('ascii').split('\n')
    file_data = list(map(lambda x: x.split('\t'), file_data))[1:-1]

    
    file_data = list(map(lambda x: [int(x[0]), int(x[1]), # chr 1, midpos 1 
                                    int(x[2]), int(x[3]), # chr 2, midpos 2
                                    int(x[4]), # interaction count 
                                    float(x[5]), float(x[6]), #p-values and q-values 
                                ],
                        file_data
                        )
                    )

    return file_data

def jaccard_index(x, y):
    set_x = set(map(lambda a: str(a[1])+':'+str(a[3]), x))
    set_y = set(map(lambda a: str(a[1])+':'+str(a[3]), y))
    union = list(set_x.union(set_y))
    interesection = list(set_x.intersection(set_y))

    return float(len(interesection))/len(union)

def significant_interaction_recovery(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    upscale,
    dataset='test',
    verbose=False,
    full_results=False
): 
    compiled_results = {
        'p-values': {
            1: [],
            0.1: [],
            0.01: [],
            0.001: [],
            0.0001: [],
            0.00001: [],
            0.000001: [],
            0.0000001: [],
        },
        'q-values': {
            1: [],
            0.1: [],
            0.01: [],
            0.001: [],
            0.0001: [],
            0.00001: [],
            0.000001: [],
            0.0000001: [],      
        },
    }

    for chromosome_id in globals.dataset_partitions[dataset]:
        base_fithic_files_path = os.path.join(base_files_path, 'chr{}_fithic'.format(chromosome_id))
        target_fithic_files_path = os.path.join(target_files_path, 'chr{}_fithic'.format(chromosome_id))
        if verbose: print("Working with base directory, {} and target directory {}".format(
            base_files_path,
            target_files_path
        ))

        # compact indexes are needed for the matrix compression in the later stage of the sparse matrix construction
        comapct_indices_base =  np.load(os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']
        compact_indices_target = np.load(os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']

        compact_indexes = list(set.intersection(set(comapct_indices_base), set(compact_indices_target)))
        
        check_if_fithic_folder_exists(base_fithic_files_path, chromosome_id, upscale, 
                cutoff=base_cutoff, compact_indexes=compact_indexes, verbose=verbose)
        
        check_if_fithic_folder_exists(target_fithic_files_path, chromosome_id, upscale, 
                cutoff=target_cutoff, compact_indexes=compact_indexes, verbose=verbose)
        

        base_significant_interactions = read_significant_interactions_fithic_file(os.path.join(base_fithic_files_path, 'FitHiC.spline_pass1.res10000.significances.txt.gz'))
        target_significant_interactions = read_significant_interactions_fithic_file(os.path.join(target_fithic_files_path, 'FitHiC.spline_pass1.res10000.significances.txt.gz'))

        for score_type in compiled_results.keys():
            for value in compiled_results[score_type].keys():
                score_index = 5 if score_type == 'p-values' else 6

                
                filtered_base_intereactions = list(filter(lambda x: x[score_index] <= value, base_significant_interactions))
                filtered_target_intereactions = list(filter(lambda x: x[score_index] <= value, target_significant_interactions))
            
                compiled_results[score_type][value].append(jaccard_index(filtered_base_intereactions, filtered_target_intereactions))


                
    averaged_results = {
        'p-values': {},
        'q-values': {},
    }

    for key in compiled_results.keys():
        for value in compiled_results[key].keys():
            if full_results:
                averaged_results[key][value] = compiled_results[key][value]
            else:
                averaged_results[key][value] = np.mean(compiled_results[key][value])

    return '{}:{}'.format(experiment_name, averaged_results)
 






# Loop recall analysis
def check_if_loop_file_exists(loop_file, chromosome, dump_directory='/home/murtaza/Documents/hic_upscaling_evaluation/temp/loop_dump/', resolution='10kb', pt=0.05):
    '''
        @params: <string> full abosolute path of the cooler file
        @params: <string> dump_directory, where to store the temporary files
        @params: <string> resolution, but in string format 
        @params: <float> pt value, P threshold for identifying significant loops
        @returns: <List<List<Float>>> coordinates in a list of list format
    '''
    
    if os.path.exists(loop_file):
        return
    # use cooler file to create a new loop file
    cooler_file = os.path.join('/'.join(loop_file.split('/')[:-1]), 'chr{}.cool'.format(chromosome))
    
    # Ensure that dump directory exists
    create_entire_path_directory(dump_directory)    

    # Creating mustache run command
    cmd = 'mustache -f {} -r {} -pt {} -o {}dump.tsv'.format(cooler_file, resolution, pt, dump_directory)
    print(cmd)

    os.system(cmd)

    loop_coordinates = open('{}/dump.tsv'.format(dump_directory)).read().split('\n')
    

    # Ignoring first and last element in the list, header and empty element
    loop_coordinates = loop_coordinates[1:-1]
    loop_coordinates = list(map(lambda x: x.split('\t'), loop_coordinates))

    
    # Store the loop ranges and discard other data
    loop_coordinate_ranges = list(map(
        lambda x: [x[0], int(x[1]), int(x[2]), x[3], int(x[4]), int(x[5]), float(x[6])],
        loop_coordinates
    ))
    
    #Clean up the dump directory 
    os.remove('{}/dump.tsv'.format(dump_directory))
    
    with open(loop_file, 'w') as f:
        for loop_coordinate in loop_coordinate_ranges:
            f.write(str(loop_coordinate[0]) + '\t' +
                str(loop_coordinate[1]) + '\t' + 
                str(loop_coordinate[2]) + '\t' +
                str(loop_coordinate[3]) + '\t' +
                str(loop_coordinate[4]) + '\t' +
                str(loop_coordinate[5]) + '\t' +
                str(loop_coordinate[6]) + '\t' +
                '\n'
            )
    


def read_dna_motif_file(dna_motif_file, chromosome):
    '''
        This file filters out the relevant motif locations based on the chromosome. Done to 
        reduce the work in the later bruteforcing steps

        @params: dna_motif_file <string>, Absolute file path to the dna motif file
        @paramn: chromosome <string>, Chromosome name in format ''chr{}''.
        @returns: <List>, a list that contains the motif locations
    '''
    dna_motif_file_data = open(dna_motif_file).read().split('\n')[:-1]
    # Split up the data to ensure that its in the correct list of list format
    dna_motif_file_data = list(
        map(
            lambda x: x.split('\t'),
            dna_motif_file_data
        )
    )
    
    # Fix the the numbers to be in float format rather than a string
    dna_motif_file_data = list(
        map(
            lambda x: [x[0],float(x[1]),float(x[2])],
            dna_motif_file_data
        )
    )

    
    # Final step, filter out only the required chromosome files
    dna_motif_file_data = list(
        filter(
            lambda x: x[0] == chromosome,
            dna_motif_file_data
        )
    )

    return dna_motif_file_data


def assign_DNA_motif_to_loop_anchor(dna_motif_file, loop_file, chromosome, slack=5000):
    '''
        @params: dna_motif_file <string>, absolute path of the file that contains the location of motifs on the DNA
        @params: loop_coordinate_ranges <List<List>>, List of list that contains the positions of loop anchors
        @params: chromosome <string>, required input for the dna_motif_file to filter out relevant motif positions
        @params: slack <int>,  Acceptable error range between the motif location and the loop anchors, 
                               defaults to 5000 which equals to half of the resolution
        @returns: <List<List>>, returns a list of list in the loop coordinate_ranges format, the contents are filtered 
                                to only contain loops that have motifs on their anchor positions
    '''
    dna_motifs = read_dna_motif_file(dna_motif_file, chromosome)
    
    loop_coordinates_ranges = open(loop_file).read().split('\n')[1:-1]
    
    loop_coordinates_ranges = list(map(lambda x: x.split('\t'), loop_coordinates_ranges))
    

    motif_mediated_loops = []

    for loop_coordinate in loop_coordinates_ranges:
        bin_1_start, bin_1_end = int(loop_coordinate[1]), int(loop_coordinate[2])
        bin_2_start, bin_2_end = int(loop_coordinate[4]), int(loop_coordinate[5])
        
        
        positive_anchor = 0
        negative_anchor = 0
        for dna_motif in dna_motifs:
            motif_x = dna_motif[1]
            motif_y = dna_motif[2]

            # Check if that motif is found in the +ve anchor
            if motif_x >= (bin_1_start - slack) and motif_x <= (bin_1_end + slack):
                positive_anchor =+ 1
            elif motif_y >= (bin_1_start - slack) and motif_y <= (bin_1_end + slack):
                positive_anchor =+ 1
            
            # Check if that motif is found in the -ve anchor
            if motif_x >= (bin_2_start - slack) and motif_x <= (bin_2_end + slack):
                negative_anchor =+ 1
            elif motif_y >= (bin_2_start - slack) and motif_y <= (bin_2_end + slack):
                negative_anchor =+ 1
            
            
        # CTCF Mediated loop, (HARD-CODED FOR CTCF)
        if positive_anchor >= 1 and negative_anchor >= 1:
            motif_based_loop = [
                    loop_coordinate[0], bin_1_start, bin_1_end,
                    loop_coordinate[3], bin_2_start, bin_2_end,
                    loop_coordinate[6]
                ]
            motif_mediated_loops.append(motif_based_loop)
    
    return motif_mediated_loops



def read_chia_pet_loop_position_file(chia_pet_file, chromosome):
    '''
        This function filters only intrachromosomal loops for a single chromosome
        @params: chia_pet_file <string>, Absolute file path to the chia pet file
        @params:  chromosome <string>, Chromosome name in format ''chr{}''
        @returns <List> a list that contains the loop coordinates
    '''
    chia_pet_file_data = open(chia_pet_file).read().split('\n')[:-1]
    # Split up the data to ensure that its in the correct list of list format
    chia_pet_file_data = list(
        map(
            lambda x: x.split('\t'),
            chia_pet_file_data
        )
    )
    
    chia_pet_file_data = list(
        map(
            lambda x: x[3].split(',')[0].split('-'),
            chia_pet_file_data
        )
    )
    
    chia_pet_file_data = list(
        map(
            lambda x: [x[0].split(':'), x[1].split(':')] ,
            chia_pet_file_data
        )
    )
   
    chia_pet_file_data = list(
        filter(
            lambda x: x[0][0] == x[1][0] and x[0][0] == chromosome,
            chia_pet_file_data
        )
    )
    
    chia_pet_file_data = list(
        map(
            lambda x: [x[0][1].split('..') + x[1][1].split('..')] ,
            chia_pet_file_data
        )
    )
    

    chia_pet_file_data = list(
        map(
            lambda x: [float(x[0][0]), float(x[0][1]), float(x[0][2]), float(x[0][3])] ,
            chia_pet_file_data
        )
    )

    return chia_pet_file_data



def identify_true_positive_loops(CTCF_PET_clusters, loop_coordinate_ranges):
    '''
        This function naively uses the CTCF cluster sites and tries to find biologically relevant loops 
        in the upscaled HiC file
        @params: <List<List>> CTCF_PET_cluster, this input contains the coordinates of the CTCF clusters
        @params:  <List<List>> loop_coordinate_ranges, this input contains the loop coordinates and their resolution range
    '''
    CTCF_based_loops = []

    for loop_coordinate_range in loop_coordinate_ranges:
        for ctcf_pet_cluster in CTCF_PET_clusters:
            bin_1_start, bin_1_end = loop_coordinate_range[1], loop_coordinate_range[2]
            bin_2_start, bin_2_end = loop_coordinate_range[4], loop_coordinate_range[5]

            ctcf_cluster_bin_1_start, ctcf_cluster_bin_1_end = ctcf_pet_cluster[0], ctcf_pet_cluster[1] 
            ctcf_cluster_bin_2_start, ctcf_cluster_bin_2_end = ctcf_pet_cluster[2], ctcf_pet_cluster[3]

            overlap_bin_1 = False
            overlap_bin_2 = False

            # Bin 1 start in the ctcf bin 1 range
            if bin_1_start >= (ctcf_cluster_bin_1_start - 10000) and bin_1_start <= (ctcf_cluster_bin_1_end + 10000):
                overlap_bin_1 = True
            
            # Bin 1 end in the ctcf bin 1 range
            if bin_1_end >= (ctcf_cluster_bin_1_start - 10000) and bin_1_end <= (ctcf_cluster_bin_1_end + 10000):
                overlap_bin_1 = True


            # Bin 2 start in the ctcf bin 2 range
            if bin_2_start >= (ctcf_cluster_bin_2_start - 10000) and bin_2_start <= (ctcf_cluster_bin_2_end + 10000):
                overlap_bin_2 = True
            
            # Bin 2 end in the ctcf bin 2 range
            if bin_2_end >= (ctcf_cluster_bin_2_start - 10000) and bin_2_end <= (ctcf_cluster_bin_2_end + 10000):
                overlap_bin_2 = True

            # Both bins must be in the ctcf cluster range
            if overlap_bin_1 and overlap_bin_2:
                #print("Loop bin 1 range",bin_1_start, bin_1_end)
                #print("Cluster bin 1 range",ctcf_cluster_bin_1_start, ctcf_cluster_bin_1_end)
                
                #print("Loop bin 2 range",bin_2_start, bin_2_end)
                #print("Cluster bin 2 range",ctcf_cluster_bin_2_start, ctcf_cluster_bin_2_end)
                
                ctcf_based_loop = [
                    loop_coordinate_range[0], bin_1_start, bin_1_end,
                    loop_coordinate_range[3], bin_2_start, bin_2_end,
                    loop_coordinate_range[6]
                ]
                CTCF_based_loops.append(ctcf_based_loop)
                break
    
    return CTCF_based_loops  




def identify_loops_overlapping_with_baseline(baseline_loops, true_positive_loops):
    count = 0
    for loop_coordinate_range in true_positive_loops:
        for ctcf_pet_cluster in baseline_loops:
            bin_1_start, bin_1_end = loop_coordinate_range[1], loop_coordinate_range[2]
            bin_2_start, bin_2_end = loop_coordinate_range[4], loop_coordinate_range[5]

            ctcf_cluster_bin_1_start, ctcf_cluster_bin_1_end = ctcf_pet_cluster[1], ctcf_pet_cluster[2] 
            ctcf_cluster_bin_2_start, ctcf_cluster_bin_2_end = ctcf_pet_cluster[4], ctcf_pet_cluster[5]

            overlap_bin_1 = False
            overlap_bin_2 = False

            # Bin 1 start in the ctcf bin 1 range
            if bin_1_start >= (ctcf_cluster_bin_1_start - 10000) and bin_1_start <= (ctcf_cluster_bin_1_end + 10000):
                overlap_bin_1 = True
            
            # Bin 1 end in the ctcf bin 1 range
            if bin_1_end >= (ctcf_cluster_bin_1_start - 10000) and bin_1_end <= (ctcf_cluster_bin_1_end + 10000):
                overlap_bin_1 = True


            # Bin 2 start in the ctcf bin 2 range
            if bin_2_start >= (ctcf_cluster_bin_2_start - 10000) and bin_2_start <= (ctcf_cluster_bin_2_end + 10000):
                overlap_bin_2 = True
            
            # Bin 2 end in the ctcf bin 2 range
            if bin_2_end >= (ctcf_cluster_bin_2_start - 10000) and bin_2_end <= (ctcf_cluster_bin_2_end + 10000):
                overlap_bin_2 = True

            # Both bins must be in the ctcf cluster range
            if overlap_bin_1 and overlap_bin_2:
                #print("Loop bin 1 range",bin_1_start, bin_1_end)
                #print("Cluster bin 1 range",ctcf_cluster_bin_1_start, ctcf_cluster_bin_1_end)
                
                #print("Loop bin 2 range",bin_2_start, bin_2_end)
                #print("Cluster bin 2 range",ctcf_cluster_bin_2_start, ctcf_cluster_bin_2_end)
                
                count+=1 
                break
    
    return count  





def perform_loop_recall_analysis(
    files_path,
    experiment_name,
    dataset='test',
    baseline_loops=[]
):
    '''
        @params: file_path <string>
    '''
    compiled_results = {
        'loop_recall_true_positive_rate': [],
        'loop_recall_overlap_with_baseline': [],
    }

    for chromosome_id in globals.dataset_partitions[dataset]:
        loop_file = os.path.join(files_path, 'chr{}_loop_coordinates.tsv'.format(chromosome_id))
        check_if_loop_file_exists(loop_file, chromosome_id)

        ctcf_mediated_loop_coordinates = assign_DNA_motif_to_loop_anchor(globals.DNA_MOTIF_FILE_PATH, loop_file, 'chr{}'.format(chromosome_id))

        chia_pet_loops = read_chia_pet_loop_position_file(globals.CHIA_PET_FILE, 'chr{}'.format(chromosome_id))

        true_positive_loops = identify_true_positive_loops(chia_pet_loops, ctcf_mediated_loop_coordinates)
        true_positives = len(true_positive_loops)
        

        if baseline_loops == []:
            print("Baseline Loops are NULL")
            overlap_with_baseline = true_positives
        else:
            overlap_with_baseline = identify_loops_overlapping_with_baseline(baseline_loops, true_positive_loops )
        
        compiled_results['loop_recall_true_positive_rate'].append(true_positives)
        compiled_results['loop_recall_overlap_with_baseline'].append(overlap_with_baseline)


    
    summed_results = {}
    for key in compiled_results.keys():
        summed_results[key] = np.sum(compiled_results[key])
    
    
    return ('{}:{}'.format(experiment_name, summed_results), true_positive_loops)






class InsulationLoss(torch.nn.Module):
    def __init__(self, window_radius=20, deriv_size=20):
        super(InsulationLoss, self).__init__()
        self.deriv_size     = deriv_size
        self.window_radius  = window_radius
        self.di_pool        = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1)
        self.top_pool       = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.mse = torch.nn.MSELoss()

    def indivInsulation(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        loss  = self.mse(tar_dv, out_dv).detach().numpy()
        return loss
        



def compute_insulation_score_on_experiment_directory(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    upscale,
    dataset='test',
    full_results=False,
    verbose=False
):
    
    compiled_results = {
        'insulation_score': []
    }
    insulation_score = InsulationLoss()
    
    for chromosome_id in globals.dataset_partitions[dataset]:
        base_file = os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id))
        target_file = os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id))
        if verbose: print("Base file: {}\nTarget File: {}".format(base_file, target_file))
        
        if not (os.path.exists(base_file) and os.path.exists(target_file)):
            print("Missing Chromosome files")
            continue

        
        comapct_indices_base =  np.load(base_file, allow_pickle=True)['compact']
        compact_indices_target = np.load(target_file, allow_pickle=True)['compact']

        compact_indexes = list(set.intersection(set(comapct_indices_base), set(compact_indices_target)))

        base_data = ops.normalize(base_file, base_cutoff, compact_indexes, verbose=verbose)
        base_data, _ = ops.divide(base_data, str(chromosome_id), 200, 200, 190, verbose=verbose)

        target_data = ops.normalize(target_file, target_cutoff, compact_indexes, verbose=verbose)
        target_data, _ = ops.divide(target_data, str(chromosome_id), 200, 200, 190, verbose=verbose)

        
        
        
        scores = float(insulation_score(torch.from_numpy(base_data), torch.from_numpy(target_data)))

        compiled_results['insulation_score'].append(scores)
    
    averaged_results = {}
    for key in compiled_results.keys():
        if full_results:
            averaged_results[key] = compiled_results[key]
        else:
            averaged_results[key] = np.mean(compiled_results[key])
    
    print(averaged_results)

    return '{}:{}'.format(experiment_name, averaged_results)


def read_bed_file(path, resolution=10000):
    # Split up the data to ensure that its in the correct list of list format
    data = open(path).read().split('\n')[:-1]
    
    data = list(
        map(
            lambda x: x.split('\t'),
            data
        )
    )[1:]

    # Fix the the numbers to be in float format rather than a string
    data = list(
        map(
            lambda x: [int(x[1])//resolution,int(x[2])//resolution],
            data
        )
    )
    return np.array(data)



def read_conserved_features_file(features_path, chromosome):
    '''
        This function reads and filter out the conserved loops and borders from the provided path to the files

        @params: dna_motif_file <string>, Absolute path to the folder containing the position bed file
        @paramn: chromosome <string>, Chromosome name in format ''chr{}''.
        @returns: <List>, <List>, Two list that contains borders and loop positions
    '''
    loop_file_path = os.path.join(features_path, 'loops', '{}.bed'.format(chromosome)) 
    border_file_path = os.path.join(features_path, 'borders', '{}.bed'.format(chromosome))
    hairpin_file_path = os.path.join(features_path, 'hairpins', '{}.bed'.format(chromosome))

    return read_bed_file(loop_file_path), read_bed_file(border_file_path), read_bed_file(hairpin_file_path)

def run_chromosight(cooler_file, chromosome):
    folder = '/'.join(cooler_file.split('/')[:-1])
    loops_output_path = os.path.join(
        folder,
        '{}_loops'.format(chromosome)
    )
    borders_output_path = os.path.join(
        folder,
        '{}_borders'.format(chromosome)
    )
    hairpins_output_path = os.path.join(
        folder,
        '{}_hairpins'.format(chromosome)
    )
    
    if not os.path.exists('{}.tsv'.format(borders_output_path)):
        cmd_path = 'chromosight detect --pattern=borders --pearson=0.3 --threads 8 {} {};'.format(
            cooler_file,
            borders_output_path
        )
        os.system(cmd_path)

    if not os.path.exists('{}.tsv'.format(hairpins_output_path)):
        cmd_path = 'chromosight detect --pattern=hairpins --pearson=0.4 --threads 8 {} {};'.format(
            cooler_file,
            hairpins_output_path
        )
        os.system(cmd_path)

    if not os.path.exists('{}.tsv'.format(loops_output_path)):
        cmd_path = 'chromosight detect --pattern=loops --threads 8 --min-dist 2000 --max-dist 200000 {} {};'.format(
            cooler_file, 
            loops_output_path,
        )
        os.system(cmd_path)
        
    return loops_output_path+'.tsv', borders_output_path+'.tsv', hairpins_output_path+'.tsv'


def read_chromosight_tsv_file(file_path):
    if os.path.exists(file_path):
        data = open(file_path).read().split('\n')[1:-1]
        data = np.array(list(map(lambda x: [x.split('\t')[i] for i in [6, 7]], data))).astype(np.int64)
        return data
    else: 
        return []
    
def distance(c1, c2):
    return math.sqrt((float(c1[0]) - float(c2[0]))**2 + (float(c1[1]) - float(c2[1]))**2)


def is_overlapping(coordinate, target_coordinates, rp=0):
    relaxation_parameter = math.sqrt(2*(rp**2))
    closest = sorted(list(map(lambda x: (x, distance(coordinate, x)), target_coordinates)), key = lambda y: y[1])[0]
    coor, d = closest

    if d <= relaxation_parameter:
       return np.array2string(coor, separator=',')
    else:
        return ''


def overlap_analysis(base, target, rp):
    multi_map = {}
    for coordinate in base:
        coordinate = is_overlapping(coordinate, target, rp)
        if coordinate not in multi_map.keys(): 
            multi_map[coordinate] = 0
            
        multi_map[coordinate] += 1

    fp = multi_map[''] if '' in multi_map.keys() else 0  # When we couldnt find any mapping in target
    fn = len(target)
    tp = 0.000000001
    mm = -1

    for key in multi_map.keys():
        if multi_map[key] >= 1 and key != '':
            tp += 1
            fn -= 1
        if multi_map[key] >= 2:
            mm += 1


    precision = tp/(tp+fp)
    recall =  tp/(tp+fn)
    
    f1 = (2*precision*recall)/(precision + recall)

    accuracy = tp/(tp+fn+fp)

    return precision, recall, f1, accuracy


def bisect_into_specific_and_shared(base, target, rp):
    specific_features = set()
    shared_features = set()
    for coordinate in base:
        if is_overlapping(coordinate, target, rp) == '':
            specific_features.add(np.array2string(coordinate, separator=','))
        else:
            shared_features.add(np.array2string(coordinate, separator=','))
    
    specific_features = list(map(lambda x: x.strip('[]').split(','),list(specific_features)))
    specific_features = list(map(lambda x: np.array([int(x[0]), int(x[1])]), specific_features))
    shared_features = list(map(lambda x: x.strip('[]').split(','),list(shared_features)))
    shared_features = list(map(lambda x: np.array([int(x[0]), int(x[1])]), shared_features))
    

    return np.array(specific_features), np.array(shared_features)
        






def filter_coordinates(coordinates, coordinate_range):
    filtered_coordinates = list(filter(
        lambda x: (x[0] >= coordinate_range[0] and x[0] <= coordinate_range[1] and x[1] >= coordinate_range[2] and x[1] <= coordinate_range[3]),
        coordinates
    ))
    filtered_coordinates = list(map(
        lambda x: np.array([x[0] - coordinate_range[0], x[1] - coordinate_range[2]]),
        filtered_coordinates
    ))
    return filtered_coordinates


def visualize_hic_matrix(hic_path, output_path, cutoff, submat, size=100, loop_coordinates=[], border_coordinates=[], hairpin_coordinates=[]):
    REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
    hic_matrix, _ = read_npz_file(hic_path)
    
    hic_matrix = hic_matrix[submat:submat+size, submat:submat+size]
    cutoff_value = np.percentile(hic_matrix, 98.0)

    hic_matrix = np.minimum(cutoff_value, hic_matrix)
    hic_matrix = np.maximum(hic_matrix, 0)


    loop_coordinates = filter_coordinates(loop_coordinates, [submat, submat+size, submat, submat+size])
    border_coordinates = filter_coordinates(border_coordinates, [submat, submat+size, submat, submat+size])
    hairpin_coordinates = filter_coordinates(hairpin_coordinates, [submat, submat+size, submat, submat+size])


    plt.scatter([x[0] for x in border_coordinates], [x[1] for x in border_coordinates], c='blue', s=50, marker='s')
    plt.scatter([x[0] for x in loop_coordinates], [x[1] for x in loop_coordinates], c='green', s=50, marker='P')
    plt.scatter([x[0] for x in hairpin_coordinates], [x[1] for x in hairpin_coordinates], c='black', s=50, marker='|')
    plt.imshow(hic_matrix, cmap=REDMAP)
    plt.axis('off')
    print(output_path)

    plt.savefig(os.path.join(output_path), bbox_inches='tight', dpi=1200)
    plt.close()


def save_conserved_features(path, data, chromosome, resolution=10000):
    with open(path, 'w') as f:
        f.write('chromosome\tstart\tend\n')
        print('Dataset {} has {} shared features'.format(path, data.shape[0]))
        for idx in range(data.shape[0]):
            f.write('{}\t{}\t{}\n'.format(
                'chr{}'.format(chromosome), int(data[idx][0])*resolution, int(data[idx][1])*resolution
        ))


def get_cell_shared_feature_set(files, feature_type='loops', resolution=10000, rp=0):
    all_pair_files = [(a, b) for idx, a in enumerate(list(files)) for b in list(files)[idx + 1:]]
    shared_features = {}

    for chr_id in globals.dataset_partitions['test']:
        shared_features = {}
        print(len(all_pair_files))

        for pair in all_pair_files:
            base_cooler_file = os.path.join(pair[0], 'chr{}.cool'.format(chr_id))
            target_cooler_file = os.path.join(pair[1], 'chr{}.cool'.format(chr_id))
            
            check_if_cooler_file_exists(base_cooler_file, chr_id, 255)
            check_if_cooler_file_exists(target_cooler_file, chr_id, 255)
            
            run_chromosight(base_cooler_file, 'chr{}'.format(chr_id))
            run_chromosight(target_cooler_file, 'chr{}'.format(chr_id))
            
            p1_features = read_chromosight_tsv_file(os.path.join(pair[0], 'chr{}_{}.tsv'.format(chr_id, feature_type)))
            p2_features = read_chromosight_tsv_file(os.path.join(pair[1], 'chr{}_{}.tsv'.format(chr_id, feature_type)))

            for feature in p1_features:
                coordinate = is_overlapping(feature, p2_features, rp)
                if coordinate not in shared_features.keys(): 
                    shared_features[coordinate] = 0
                    
                shared_features[coordinate] += 1

        with open(os.path.join(globals.FEATURES_FOLDER_PATH, feature_type, 'chr{}.bed'.format(chr_id)), 'w') as f:
            count = 0
            for key in shared_features.keys():
                if shared_features[key] > 21 or shared_features[key] == 0:
                    continue
                else:
                    count+=1
                    x, y = key.strip('[]').split(',')
                    f.write('{}\t{}\t{}\n'.format('chr{}'.format(chr_id), int(x)*resolution, int(y)*resolution))
            print('Chr {} has {} shared features'.format(chr_id, count))
        
        print(count, p1_features.shape, p2_features.shape)






def conserved_feature_analysis_on_experiment_directory( 
        target_files_path,
        base_files_path,
        experiment_name,
        base_cutoff,
        target_cutoff,
        upscale,
        dataset='all',
        full_results=False,
        verbose=False
    ):
    results ={
        'Aggregate': {
            'Loops': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Borders': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Hairpins': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
        },
        'Shared': {
            'Loops': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Borders': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Hairpins': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
        },
        'Specific': {
            'Loops': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Borders': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
            'Hairpins': {
                'precision': [],
                'recall': [],
                'f1': [],
                'acc': []
            },
        }
    }
    

    for chromosome_id in globals.dataset_partitions['test']:
        print(base_files_path)
        
        base_cooler_file = os.path.join(base_files_path, 'chr{}.cool'.format(chromosome_id))
        target_cooler_file = os.path.join(target_files_path, 'chr{}.cool'.format(chromosome_id))
        check_if_cooler_file_exists(base_cooler_file, chromosome_id, base_cutoff)
        check_if_cooler_file_exists(target_cooler_file, chromosome_id, target_cutoff)
        if verbose: print("Base file: {}\nTarget File: {}".format(base_cooler_file, target_cooler_file))

        if verbose: print('Running Chromosight to extract features')
        run_chromosight(base_cooler_file, 'chr{}'.format(chromosome_id))
        run_chromosight(target_cooler_file, 'chr{}'.format(chromosome_id))
        
        if verbose: print('Reading the chromosight generated files')
        base_loops = read_chromosight_tsv_file(os.path.join(base_files_path, 'chr{}_loops.tsv'.format(chromosome_id)))
        target_loops = read_chromosight_tsv_file(os.path.join(target_files_path, 'chr{}_loops.tsv'.format(chromosome_id)))

        base_borders = read_chromosight_tsv_file(os.path.join(base_files_path, 'chr{}_borders.tsv'.format(chromosome_id)))
        target_borders = read_chromosight_tsv_file(os.path.join(target_files_path, 'chr{}_borders.tsv'.format(chromosome_id)))
        
        base_hairpins = read_chromosight_tsv_file(os.path.join(base_files_path, 'chr{}_hairpins.tsv'.format(chromosome_id)))
        target_hairpins = read_chromosight_tsv_file(os.path.join(target_files_path, 'chr{}_hairpins.tsv'.format(chromosome_id)))


        if verbose: print('Reading conserved features file')
        conserved_loops, conserved_borders, conserved_hairpins = read_conserved_features_file(globals.FEATURES_FOLDER_PATH, 'chr{}'.format(chromosome_id))
        
        target_specific_loops, target_shared_loops = bisect_into_specific_and_shared(target_loops, conserved_loops, LOOP_RELAXATION_PARAMETER)
        target_specific_borders, target_shared_borders = bisect_into_specific_and_shared(target_borders, conserved_borders, BORDER_RELAXATION_PARAMETER)
        target_specific_hairpins, target_shared_hairpins = bisect_into_specific_and_shared(target_hairpins, conserved_hairpins, HAIRPIN_RELAXATION_PARAMETER)
        
        print(target_loops.shape, target_specific_loops.shape, target_shared_loops.shape)
        print(target_borders.shape, target_specific_borders.shape, target_shared_borders.shape)
        print(target_hairpins.shape, target_specific_hairpins.shape, target_shared_hairpins.shape)
        

        # Aggregate analysis
        precision, recall, f1, acc = overlap_analysis(base_loops, target_loops, LOOP_RELAXATION_PARAMETER)
        results['Aggregate']['Loops']['precision'].append(precision)
        results['Aggregate']['Loops']['recall'].append(recall)
        results['Aggregate']['Loops']['f1'].append(f1)
        results['Aggregate']['Loops']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_borders, target_borders, BORDER_RELAXATION_PARAMETER)
        results['Aggregate']['Borders']['precision'].append(precision)
        results['Aggregate']['Borders']['recall'].append(recall)
        results['Aggregate']['Borders']['f1'].append(f1)
        results['Aggregate']['Borders']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_hairpins, target_hairpins, HAIRPIN_RELAXATION_PARAMETER)
        results['Aggregate']['Hairpins']['precision'].append(precision)
        results['Aggregate']['Hairpins']['recall'].append(recall)
        results['Aggregate']['Hairpins']['f1'].append(f1)
        results['Aggregate']['Hairpins']['acc'].append(acc)


        # Shared analysis
        precision, recall, f1, acc = overlap_analysis(base_loops, target_shared_loops, LOOP_RELAXATION_PARAMETER)
        results['Shared']['Loops']['precision'].append(precision)
        results['Shared']['Loops']['recall'].append(recall)
        results['Shared']['Loops']['f1'].append(f1)
        results['Shared']['Loops']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_borders, target_shared_borders, BORDER_RELAXATION_PARAMETER)
        results['Shared']['Borders']['precision'].append(precision)
        results['Shared']['Borders']['recall'].append(recall)
        results['Shared']['Borders']['f1'].append(f1)
        results['Shared']['Borders']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_hairpins, target_shared_hairpins, HAIRPIN_RELAXATION_PARAMETER)
        results['Shared']['Hairpins']['precision'].append(precision)
        results['Shared']['Hairpins']['recall'].append(recall)
        results['Shared']['Hairpins']['f1'].append(f1)
        results['Shared']['Hairpins']['acc'].append(acc)


        # Specific analysis
        precision, recall, f1, acc = overlap_analysis(base_loops, target_specific_loops, LOOP_RELAXATION_PARAMETER)
        results['Specific']['Loops']['precision'].append(precision)
        results['Specific']['Loops']['recall'].append(recall)
        results['Specific']['Loops']['f1'].append(f1)
        results['Specific']['Loops']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_borders, target_specific_borders, BORDER_RELAXATION_PARAMETER)
        results['Specific']['Borders']['precision'].append(precision)
        results['Specific']['Borders']['recall'].append(recall)
        results['Specific']['Borders']['f1'].append(f1)
        results['Specific']['Borders']['acc'].append(acc)

        precision, recall, f1, acc = overlap_analysis(base_hairpins, target_specific_hairpins, HAIRPIN_RELAXATION_PARAMETER)
        results['Specific']['Hairpins']['precision'].append(precision)
        results['Specific']['Hairpins']['recall'].append(recall)
        results['Specific']['Hairpins']['f1'].append(f1)
        results['Specific']['Hairpins']['acc'].append(acc)
        

        #if True:
        visualize_hic_matrix(
            os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id)), 
            os.path.join(base_files_path, 'chr{}.png'.format(chromosome_id)),
            base_cutoff,
            4500,
            200,
            base_loops,
            base_borders,
            base_hairpins
        )

        visualize_hic_matrix(
            os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id)), 
            os.path.join(target_files_path, 'chr{}.png'.format(chromosome_id)),
            target_cutoff,
            4500,
            100,
            target_loops,
            target_borders,
            target_hairpins
        )
        
        # for rp in range(30):
        #     print('Computing for borders')
        #     precision, recall, f1, accuracy = overlap_analysis(base_hairpins, target_hairpins, rp)

        #     if rp not in rp_results.keys():
        #         rp_results[rp] = {
        #             'precision': [],
        #             'recall': [],
        #             'f1': [],
        #             'accuracy': []
        #         }
            
        #     rp_results[rp]['precision'].append(precision)
        #     rp_results[rp]['recall'].append(recall)
        #     rp_results[rp]['f1'].append(f1)
        #     rp_results[rp]['accuracy'].append(accuracy)
            
            
            




       

        # target_overlapping_loops, target_specific_loops = get_overlapping_features(target_loops, conserved_loops)
        # base_overlapping_loops, base_specific_loops = get_overlapping_features(base_loops, conserved_loops)

        # target_overlapping_borders, target_specific_borders = get_overlapping_features(target_borders, conserved_borders)
        # base_overlapping_borders, base_specific_borders = get_overlapping_features(base_borders, conserved_borders)
        
        # base_target_overlap_loops, _ = get_overlapping_features(base_loops, target_loops)
        # base_target_overlap_borders, _ = get_overlapping_features(base_borders, target_borders)

        
        # results_string += 'chr:{},base_total_loops:{},base_specific_loops:{},base_shared_loops:{},target_total_loops:{},target_specific_loops:{},target_shared_loops:{},base-target_overlap_loops:{}----'.format(
        #     chromosome_id,
        #     base_loops.shape[0],
        #     base_specific_loops.shape[0],
        #     base_overlapping_loops.shape[0],
        #     target_loops.shape[0],
        #     target_specific_loops.shape[0],
        #     target_overlapping_loops.shape[0],
        #     base_target_overlap_loops.shape[0]
        # )
        
        # results_string += 'chr:{},base_total_borders:{},base_specific_borders:{},base_shared_borders:{},target_total_borders:{},target_specific_borders:{},target_shared_borders:{},base-target_overlap_borders:{}----'.format(
        #     chromosome_id,
        #     base_borders.shape[0],
        #     base_specific_borders.shape[0],
        #     base_overlapping_borders.shape[0],
        #     target_borders.shape[0],
        #     target_specific_borders.shape[0],
        #     target_overlapping_borders.shape[0],
        #     base_target_overlap_borders.shape[0]
        # )
    
    # results_string = 'agg_precision:{},specific_precision:{},shared_precision:{},agg_recall:{},specific_recall:{},shared_recall:{},agg_f1:{},specific_f1:{},shared_f1:{},agg_acc:{},specific_acc:{},shared_acc:{}'.format(
    #     np.mean(rp_agg_results['precision']), np.mean(rp_specific_results['precision']), np.mean(rp_shared_results['precision']),
    #     np.mean(rp_agg_results['recall']), np.mean(rp_specific_results['recall']), np.mean(rp_shared_results['recall']),
    #     np.mean(rp_agg_results['f1']), np.mean(rp_specific_results['f1']), np.mean(rp_shared_results['f1']),
    #     np.mean(rp_agg_results['acc']), np.mean(rp_specific_results['precision']), np.mean(rp_shared_results['acc']),
    # )
    compiled_results = {}
    
    for analysis_type in results.keys():
        if analysis_type not in compiled_results.keys():
            compiled_results[analysis_type] = {}
        for feature_type in results[analysis_type].keys():
            if feature_type not in compiled_results[analysis_type].keys():
                compiled_results[analysis_type][feature_type] = {}
            for metric in results[analysis_type][feature_type].keys():
                if metric not in compiled_results[analysis_type][feature_type].keys():
                    compiled_results[analysis_type][feature_type][metric] = 0
                compiled_results[analysis_type][feature_type][metric] = np.mean(results[analysis_type][feature_type][metric])




    # compiled_results = {}
    # for rp in rp_results.keys():
    #     print('{}\t{}\t{}\t{}\t{}'.format(
    #         rp,
    #         np.mean(rp_results[rp]['precision']),
    #         np.mean(rp_results[rp]['recall']),
    #         np.mean(rp_results[rp]['f1']),
    #         np.mean(rp_results[rp]['accuracy'])
    #     ))
        
        # compiled_results[rp] = {
        #     'recall': np.mean(rp_results[rp]['recall']),
        #     'precision': np.mean(rp_results[rp]['precision']),
        #     'f1': np.mean(rp_results[rp]['f1']),
        #     'accuracy': np.mean(rp_results[rp]['accuracy'])
        # }
    


    return '{}:{}'.format(experiment_name, compiled_results)