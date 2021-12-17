import argparse
import os
from src import globals
from src.utils import compute_cutoff_value

parser = argparse.ArgumentParser(description='This script finds cutoff value for all chromosome files contained in a directory. This cutoff value is used to normalize HiC contact matrices')
parser.add_argument('-b', '--base-directory', help='Path to the chromosome matrices directory', required=True)
parser.add_argument('--percentile', help='Percentile value, defaults to 99.95th percentile', type=float, default=99.95)
parser.add_argument('-v', '--verbose', help='Increases the information that is displayed on the terminal',
                    action='store_true')
parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')


args = parser.parse_args()

path_to_chromosomes = args.base_directory
percentile = args.percentile

if not args.absolute:
    path_to_chromosomes = os.path.join(globals.DATA_DIRECTORY, path_to_chromosomes)
    if args.verbose:
        print('Updated directory path to: {}'.format(path_to_chromosomes))



cutoff_value = compute_cutoff_value(path_to_chromosomes, percentile)

print("Cutoff value for {} is {}".format(path_to_chromosomes, cutoff_value))