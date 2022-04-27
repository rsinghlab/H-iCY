import argparse
import os
from src import globals
from src import format_handler



parser = argparse.ArgumentParser(description='Read a HiC dataset file and extract all the intra chromosomal matrices. \
    Currently only supports .hic format. (Will add support for .cool and .mcool eventually)')



parser.add_argument('-f', '--input-file', help='Path to the input file', required=True)
parser.add_argument('-o', '--output-path', help='Path to the output folder that will host the generated chromosome files', required=True)
parser.add_argument('-v', '--verbose', help='Increases the information that is displayed on the terminal',
                    action='store_true')
parser.add_argument('-r', '--resolution', help='Resolution to sample the output matrices at, defaults to 10000', default=10000)
parser.add_argument('-n', '--normalization',  help='Normalization method to use, currently only supported method is KR so it defaults to that', default='KR')
parser.add_argument('-a', '--absolute', help='Is the provided path an absolute path, if this flag is set to false then parser \
appends the DATA_DIRECTORY path (its defined in the src/globals.py file)', action='store_true')

parser.add_argument('-d', '--dataset', help='Which set of chromosomes to extract. There are four divisions: 1) test [1-8, 13-18] \
    2) valid [9-12] 3) test 19-22 and 4) all [1-22]. Defaults to extracting all chromosomes', default='all' 
)


args = parser.parse_args()

input_file_path = args.input_file
output_directory_path = args.output_path

print(input_file_path)

# If absolute argument is not supplied update the paths with the base path
if not args.absolute:
    input_file_path = os.path.join(globals.DATA_DIRECTORY, input_file_path)
    output_directory_path = os.path.join(globals.DATA_DIRECTORY, output_directory_path)
    if args.verbose:
        print('Updated input file path to: {}'.format(input_file_path))
        print('Updated output directory path to: {}'.format(output_directory_path))


format_handler.extract_npz_chromosomal_files_from_hic_file(input_file_path, output_directory_path, 
    resolution=int(args.resolution), normalization=args.normalization, 
    chromosomes=args.dataset, verbose=args.verbose)