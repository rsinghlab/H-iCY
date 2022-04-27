import argparse
import os
from src import globals
from src.utils import create_entire_path_directory
from src.evaluation import common



parser = argparse.ArgumentParser(description='This script runs all the evaluation metrics and store the results in a file')
parser.add_argument('-t', '--path-to-target-chromosomes', help='Path to the High Quality, High Res or High Read count chromosomes.', required=True)
parser.add_argument('-b', '--path-to-base-chromosomes', help='Path to the either upscaled or baseline chromsome directory', required=True)
parser.add_argument('-c1', '--target-cutoff-value', help='Cutoff value for the target chromosomes', required=True, type=int)
parser.add_argument('-c2', '--base-cutoff-value', help='Cutoff value for the base chromosomes', required=True, type=int)
parser.add_argument('-e', '--name-of-the-experiment', help='Results file would contain entry with this name', required=True)
parser.add_argument('--correlation-analysis', help='Run Correlation analysis', action='store_true')
parser.add_argument('--hic-analysis', help='Run HiC similarity analysis', action='store_true')
parser.add_argument('--downstream-analysis', help='Run downstream analysis', action='store_true')
parser.add_argument('-r','--results-file-name', help='Name of the results file that would be generated in results/', required=True)
parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')

args = parser.parse_args()
target_dataset_path = args.path_to_target_chromosomes
base_dataset_path = args.path_to_base_chromosomes
target_dataset_cutoff = args.target_cutoff_value
base_dataset_cutoff = args.base_cutoff_value
experiment_name = args.name_of_the_experiment
results_file_name = args.results_file_name


if not args.absolute:
    target_dataset_path = os.path.join(globals.DATA_DIRECTORY, target_dataset_path)
    base_dataset_path = os.path.join(globals.DATA_DIRECTORY, base_dataset_path)
    
steps = []
if args.correlation_analysis:
    steps.append('correlation_metrics')

if args.hic_analysis:
    steps.append('hicrep')
    steps.append('biological_metrics')

if args.downstream_analysis:
    steps.append('3d_similarity')
    steps.append('significant_interaction')


common.evaluate(
    results_file_name,
    target_dataset_path,
    [base_dataset_path],
    target_dataset_cutoff,
    [base_dataset_cutoff],
    [experiment_name],
    full_results=True,
    steps=steps
)

