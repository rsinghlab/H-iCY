import argparse
import os
from src import globals
from src.utils import downsample_chromsosome_file, create_entire_path_directory







parser = argparse.ArgumentParser(description='This script downsamples all the chromosome files in a directory by a uniform ratio')
parser.add_argument('-b', '--base-directory', help='Path to the chromosome matrices directory', required=True)
parser.add_argument('-o', '--output-directory', help='Path to the output directory where to store the downsampled chromosomes.', required=True)
parser.add_argument('--downsampling-ratio', help='Downsampling ratio, for example: a ratio of 16 means all the read counts are scaled to their 1/16th value.', type=int, required=True)
parser.add_argument('-v', '--verbose', help='Increases the information that is displayed on the terminal',
                    action='store_true')
parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')


args = parser.parse_args()

path_to_chromosomes = args.base_directory
output_directory = args.output_directory

create_entire_path_directory(output_directory)

dowsampling_ratio = args.downsampling_ratio

if not args.absolute:
    path_to_chromosomes = os.path.join(globals.DATA_DIRECTORY, path_to_chromosomes)
    if args.verbose:
        print('Updated directory path to: {}'.format(path_to_chromosomes))


downsample_chromsosome_file(path_to_chromosomes, output_directory, dowsampling_ratio)
