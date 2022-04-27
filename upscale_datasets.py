import argparse
import os
from src import globals
from src.utils import create_entire_path_directory
from src.methods import common




parser = argparse.ArgumentParser(description='This script provides an interface to upscale the HiC matrices through a set of deep learning based techniques.')
parser.add_argument('-d', '--path-to-dataset', help='Path to the dataset', required=True)
parser.add_argument('-o', '--output-directory', help='Path to the output directory where to store the upscaled chromosomes.', required=True)
parser.add_argument('--model', help='Name of the model to use for upscaling the HiC matrices. Currently, we have smoothing, hicplus16, hicplus25, hicplus50, hicplus100, hicnn16, hicnn25, hicnn50, hicnn100, deephic16, deephic25, deephic50, deephic100, vehicle, hicnn-gaussian, hicnn-random, hicnn-uniform, hicnn-synthetic-ensemble, hicnn-lrc and hicnn-lrc-ensemble', required=True)
parser.add_argument('-v', '--verbose', help='Increases the information that is displayed on the terminal',
                    action='store_true')
parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')


args = parser.parse_args()

dataset = args.path_to_dataset
output_directory = args.output_directory


if not args.absolute:
    dataset = os.path.join(globals.DATA_DIRECTORY, dataset)
    output_directory = os.path.join(globals.DATA_DIRECTORY, output_directory)
    print(dataset)

    if args.verbose:
        print('Updated directory path to: {}'.format(dataset))


model = args.model

create_entire_path_directory(output_directory)

common.upscale(dataset, model, output_directory)
