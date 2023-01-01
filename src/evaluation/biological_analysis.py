import os
from numpy.core.numeric import full
from numpy.lib.function_base import average

from pandas.core.indexes import base
from src import globals
from src import format_handler
import numpy as np
from src import utils
import hicreppy
from importlib.machinery import SourceFileLoader
import cooler as clr            

# Fixing hicreppy importing error
hicrep_file_path = os.path.join(hicreppy.__path__[0], 'hicrep.py')

hicrep = SourceFileLoader("hicrep", hicrep_file_path).load_module()


# HiCRep
def check_if_cooler_file_exists(cooler_file_path, chromosome, cutoff):
    """
        This function checks if the cooler file exits, if it doesnt then it finds a .npz file in the directory and if it exists,
        it creates a cooler file from that .npz file.
        @params: cooler_file_path <string>,  path to the cooler file
        @params: chromosome <string>, chromosome id
        @params: cutoff value to use
        @returns: None
    """
    if not os.path.exists(cooler_file_path):
        npz_file_path = cooler_file_path.replace('.cool', '.npz')
        if not os.path.exists(npz_file_path):
            print("Chromosome file missing, aborting program execution")
            exit(1)
        format_handler.create_cool_file_from_numpy(npz_file_path, cooler_file_path, chromosome, cutoff=cutoff)


# HiCRep wrapper script
def get_genome_scc_on_cooler_files(cooler_file_one_path, cooler_file_two_path, max_dist=200000, h=1):
    '''
        @params: <string> cooler_file_one_path absolute path to the cooler file one
        @params: <string> cooler_file_two_path absolute path to the cooler file two
        @params: <int> max_dist max distance at which we consider our stratum , defualts to 2MBs
        @params: <int> h , smoothing parameter required for hicrep, defaults to 1, minimal smoothing
        @returns: <float> scc, hicrep similarity score    
    '''
    cooler_one = clr.Cooler(cooler_file_one_path)
    cooler_two = clr.Cooler(cooler_file_two_path)

    return hicrep.genome_scc(cooler_one, cooler_two, max_dist, h)


def compute_hicrep_scores_on_experiment_directory(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    dataset='test',
    full_results=False,
    verbose=False
):
    
    compiled_results = {'hicrep': []}

    for chromosome_id in globals.dataset_partitions[dataset]:
        base_cooler_file = os.path.join(base_files_path, 'chr{}.cool'.format(chromosome_id))
        target_cooler_file = os.path.join(target_files_path, 'chr{}.cool'.format(chromosome_id))
        check_if_cooler_file_exists(base_cooler_file, chromosome_id, base_cutoff)
        check_if_cooler_file_exists(target_cooler_file, chromosome_id, target_cutoff)
        
       
        if verbose: print("Base file: {}\nTarget File: {}".format(base_cooler_file, target_cooler_file))
        
       
        mean_value = get_genome_scc_on_cooler_files(base_cooler_file, target_cooler_file)

        compiled_results['hicrep'].append(mean_value)

    averaged_results = {}
    
    for key in compiled_results.keys():
        if full_results:
            averaged_results[key] = compiled_results[key]
        else:
            averaged_results[key] = np.mean(compiled_results[key])
    
    return '{}:{}'.format(experiment_name, averaged_results)


# HiC-Spector, GenomeDisco and QuASAR-Rep

def check_if_chromosome_bed_file_exists(output_path, chr_name, resolution=10000, number_of_bins=200):
    '''
        @params: output_path
        @params: chr_name
        @params: resolution
        @params: number_of_bins
    '''
    # Check if the bed file already exists, if yes return and use the bed file
    # if os.path.exists(output_path):
    #     return output_path
    #output_path = '/'.join(output_path.split('/')[:-1])

    # Generate the .bed file for resolution bins, it has format chr_name, start_bin end_bin value?
    with open(output_path, 'w') as f:
        # will we support arbitrary starting locations?
        curr_starting = 0
        curr_ending = curr_starting + resolution
        for _ in range(number_of_bins):
            f.write('{}\t{}\t{}\t{}\n'.format(chr_name, curr_starting, curr_ending, curr_starting))
            curr_starting = curr_ending
            curr_ending += resolution
    
    # Compress .bed file
    output_path = utils.compress_file(output_path, clean=True)
    return output_path




def check_if_sparse_contact_matrix_directory_exists(sparse_contact_matrix_file_path, chromosome, cutoff, chunk, stride, bound, upscale, compact_indexes):
    """
        This function checks if the sparse matrix directory exists that is required for the 3Dchromatin replicate analysis scripts, 
        if the directory doesnt exist, it creates the directory and populates it by converting the dense matrix located in the same directory.
        @params: sparse_contact_matrix_file_path <string> path to the sparse contact matrix file
        @params: chromosome <string>, chromosome id
        @params: cutoff <int> cutoff value to use
        @params: chunk <int>, matrix division parameter
        @params: stride <int> matrix division parameter
        @params: bound <int> matrix division parameter
        @params: upscale <int> matrix division parameter
        @params: compact_indexes <list> matrix division parameter
        @returns: None
    """
    #if not os.path.exists(sparse_contact_matrix_file_path):
    chromosome_file = os.path.join('/'.join(sparse_contact_matrix_file_path.split('/')[:-1]), 'chr{}.npz'.format(chromosome))
    
    if not os.path.exists(chromosome_file):
        print('Chromosome file not found, exiting the program')
        exit(1)

    format_handler.create_sparse_matrix_files_from_numpy(chromosome_file, sparse_contact_matrix_file_path, 
                chromosome, cutoff=cutoff, chunk=chunk,
                stride=stride, bound=bound,
                upscale=upscale, compact_indexes=compact_indexes)
    


def create_metadata_samples_file(base_sparse_matrix_directory, 
        target_sparase_matrix_directory, output_directory):
    '''
        @params: base_sparse_matrix_directory <string>, path to base sparse matrix directory
        @params: target_sparse_matrix_directory <string>, path to target sparse matrix directory
        @params: output_directory <string>, temporary folder to store the generated files
    '''
    utils.create_entire_path_directory(output_directory)
    
    base_sparse_matrix_directory_files = list(map(lambda x: os.path.join(base_sparse_matrix_directory, x),
        os.listdir(base_sparse_matrix_directory)))
    target_sparase_matrix_directory_files = list(map(lambda x: os.path.join(target_sparase_matrix_directory, x),
        os.listdir(target_sparase_matrix_directory)))
    
    base_ids = []
    target_ids = []

    # Clean up the exisiting file
    if os.path.exists(os.path.join(output_directory, 'metadata.samples')):
        os.remove(os.path.join(output_directory, 'metadata.samples'))
        

    for sparse_matrix_file in base_sparse_matrix_directory_files:
        filename = sparse_matrix_file.split('/')[-1]
        parent_dir = sparse_matrix_file.split('/')[-2]
        base_id = 'base_{}_{}'.format(parent_dir, filename.split('.')[0])

        with open(os.path.join(output_directory, 'metadata.samples'), 'a+') as f:
            f.write('{}\t{}\n'.format(base_id, sparse_matrix_file))
        
        base_ids.append(base_id)
    
    for sparse_matrix_file in target_sparase_matrix_directory_files:
        filename = sparse_matrix_file.split('/')[-1]
        parent_dir = sparse_matrix_file.split('/')[-2]
        target_id = 'target_{}_{}'.format(parent_dir, filename.split('.')[0])

        with open(os.path.join(output_directory, 'metadata.samples'), 'a+') as f:
            f.write('{}\t{}\n'.format(target_id, sparse_matrix_file))
        
        target_ids.append(target_id)
    
    assert(len(base_ids) == len(target_ids))

    return base_ids, target_ids, os.path.join(output_directory, 'metadata.samples')

def create_metadata_pairs_file(base_ids, target_ids, output_directory):
        '''
            Based on the ids generated by metadata_samples function we create a pairs file required by the 3dqc util
            @params: base_ids <list>, list of the generated base ids
            @params: target_ids <list> , list of the generated target ids
            @params: output_directory <string>, path to the temporary directory that holds the generated file
        '''

        utils.create_entire_path_directory(output_directory)
        assert(len(base_ids) == len(target_ids))
        
        if os.path.exists(os.path.join(output_directory, 'metadata.pairs')):
            os.remove(os.path.join(output_directory, 'metadata.pairs'))
    
        for base_id in base_ids:
            base_id_identifier = '_'.join(base_id.split('_')[1:])
            for target_id in target_ids:
                if base_id_identifier in target_id:
                    break
            
            with open(os.path.join(output_directory, 'metadata.pairs'), 'a+') as f:
                f.write('{}\t{}\n'.format(base_id, target_id))

        return os.path.join(output_directory, 'metadata.pairs')


def create_3dqc_command_string(samples_path, pairs_path, bed_path, output_path):
    '''
        Takes in all the paths and compiles a command string that can be executed later to collect the results
    '''
    output_path = os.path.join(output_path, 'results')
    
    if os.path.exists(output_path):
        os.system('rm -rf {}'.format(output_path))
    
    command = '''3DChromatin_ReplicateQC run_all --metadata_samples {} --metadata_pairs {} --bins {} --outdir {}  --concise_analysis --parameters_file {}'''.format(
        samples_path, 
        pairs_path,
        bed_path,
        output_path,
        globals.QC_PARAMETERS_FILE
    )
    
    return command, output_path


def compile_results(results_path):
    '''
        This function reads the results directory generated by the 3dqc utility and parses out the relevant results
        @params: results_path
        @returns <list> of results, where 0th index is gdisco, 1st index is hic-spector and 2nd index is qrep
    '''
    results_data = open(os.path.join(results_path, 'scores/reproducibility.genomewide.txt')).read().split('\n')
    results_data = list(map(lambda x: x.split('\t'), results_data))

    results_data = results_data[1:-1]
    gdisco = []
    hspector = []
    qrep = []

    for data in results_data:
        gdisco.append(float(data[2]))
        hspector.append(float(data[3]))
        qrep.append(float(data[4]))
    
    return np.mean(gdisco), np.mean(hspector), np.mean(qrep)

def compute_hicspector_gdisco_qrep_scores_on_experiment_directory(
    base_files_path,
    target_files_path,
    experiment_name,
    base_cutoff,
    target_cutoff,
    chunk=200,
    stride=200,
    bound=190,
    upscale=255,
    dataset='test',
    verbose=True,
    full_results=False,
    temp_dir=os.path.join(globals.TEMP_DATA_DIRECTORY, '3dqc_temp/')
):
    '''
    
    '''
    compiled_results = {'hic-spector': [], 'genomedisco': [], 'quasar-rep': []}

    for chromosome_id in globals.dataset_partitions[dataset]:
        base_sparse_contact_matrix_directory = os.path.join(base_files_path, 'chr{}_c{}_s{}_b{}_u{}'.format(
            chromosome_id,
            chunk,
            stride,
            bound,
            upscale
        ))
        target_sparse_contact_matrix_directory = os.path.join(target_files_path, 'chr{}_c{}_s{}_b{}_u{}'.format(
            chromosome_id,
            chunk,
            stride,
            bound,
            upscale
        ))
        
        # compact indexes are needed for the matrix compression in the later stage of the sparse matrix construction
        comapct_indices_base =  np.load(os.path.join(base_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']
        compact_indices_target = np.load(os.path.join(target_files_path, 'chr{}.npz'.format(chromosome_id)), allow_pickle=True)['compact']

        compact_indexes = list(set.intersection(set(comapct_indices_base), set(compact_indices_target)))


        check_if_sparse_contact_matrix_directory_exists(base_sparse_contact_matrix_directory, 
                chromosome_id, base_cutoff, chunk, stride, bound, upscale, compact_indexes)
        check_if_sparse_contact_matrix_directory_exists(target_sparse_contact_matrix_directory, 
                chromosome_id, target_cutoff, chunk, stride, bound, upscale, compact_indexes)

        base_ids , target_ids, metadata_samples_file_path = create_metadata_samples_file(base_sparse_contact_matrix_directory, 
                        target_sparse_contact_matrix_directory, output_directory=temp_dir)

        metadata_pairs_file_path = create_metadata_pairs_file(base_ids, target_ids, output_directory=temp_dir)

        bed_file_path = os.path.join(globals.DATA_DIRECTORY, 'chr_bed_files/chr{}_{}.bed.gz'.format(chromosome_id, chunk))
        bed_file_path = check_if_chromosome_bed_file_exists(bed_file_path, 'chr{}'.format(chromosome_id), number_of_bins=chunk)
        
        command, results_path = create_3dqc_command_string(metadata_samples_file_path, metadata_pairs_file_path, bed_file_path, output_path=temp_dir)
        
        os.system(command)
        gdisco, hspector, qrep = compile_results(results_path)
        compiled_results['genomedisco'].append(gdisco)
        compiled_results['hic-spector'].append(hspector)
        compiled_results['quasar-rep'].append(qrep)
        
    
    averaged_results = {}
    for key in compiled_results.keys():
        if full_results:
            averaged_results[key] = compiled_results[key]
        else: 
            averaged_results[key] = np.mean(compiled_results[key])
    
    
    return '{}:{}'.format(experiment_name, averaged_results)





























