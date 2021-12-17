import argparse
import os

from src import dataset_creator
from src import globals


parser = argparse.ArgumentParser(description='Takes path to the chromosome directories and generates datasets that are compatible with the deeplearing \
        based methods. It works in two steps, first step if a cutoff value is not provided computes the cutoff value and then in the second step it  \
        divides the chromosomal matrices with provided chunk size values and strides to construct a dataset that contains high quality and low quality pairs.')

parser.add_argument('-b', '--base-directory', help='Path to the low quality chromosome matrices directory', required=True)
parser.add_argument('-t', '--target-directory', help='Path to the high quality chromosome matrices directory', required=True)
parser.add_argument('-o', '--output-directory', help='Path to the directory where the generated dataset is stored', required=True)
parser.add_argument('-n', '--name-of-dataset-file', help='Name of the generated dataset file', required=True)
parser.add_argument('-v', '--verbose', help='Increases the information that is displayed on the terminal',
                    action='store_true')

parser.add_argument('--chunk', help='Size of the chunks cropped out from the chromosomal matrices', type=int, default=40)
parser.add_argument('--stride', help='Stride length after each chunk, smaller than chunk size means that chunks have overlapping data',
                     type=int, default=40)
parser.add_argument('--bound', help='Maximum distance in number of bins to consider from the center, defaults to 201, beacause most of the genomic are in that range',
                     type=int, default=201)

parser.add_argument('--cutoff-base', help='Cutoff value to use for the base chromosome files (These cutoffs are specified in globals.py)', type=int, required=True)
parser.add_argument('--cutoff-target', help='Cutoff value to use for the target chromosome files (These cutoffs are specified in globals.py)', type=int, required=True)

parser.add_argument('-d', '--dataset', help='Dataset division, choices [train, valid, test, all]. all generates all three datasets', required=True)

parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')

args = parser.parse_args()

base_directory = args.base_directory
target_directory = args.target_directory
output_directory = args.output_directory

# If absolute argument is not supplied update the paths with the base path
if not args.absolute:
    base_directory = os.path.join(globals.DATA_DIRECTORY, base_directory)
    target_directory = os.path.join(globals.DATA_DIRECTORY, target_directory)
    output_directory = os.path.join(globals.DATA_DIRECTORY, output_directory)
    
    if args.verbose:
        print('Updated base directory path to: {}'.format(base_directory))
        print('Updated target directory path to: {}'.format(target_directory))
        print('Updated output directory path to: {}'.format(output_directory))
        
if args.dataset != 'all':
    datasets = [args.dataset]
else:
    datasets = ['train', 'valid', 'test']

for dtst in datasets:
    dataset_creator.create_dataset_file(
        base_directory,
        target_directory,
        output_directory,
        args.name_of_dataset_file,
        args.cutoff_base,
        args.cutoff_target,
        dataset=dtst,
        chunk=args.chunk,
        stride=args.stride,
        bound=args.bound,
        compact_type='intersection',
        verbose=args.verbose
    )