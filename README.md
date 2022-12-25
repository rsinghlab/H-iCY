# H-iCY


## Abstract
HiC is a widely used technique to study the 3D organization of the genome. Unfortunately, due to its high sequencing cost, most of the generated datasets are of coarse quality, consequently limiting the quality of the downstream analyses. Recently, many computational methods have been proposed that improve the quality of these matrices primarily using deep-learning-based methods. Unfortunately, most of these methods make simplifying assumptions during the generation of training datasets and while evaluating these methods, which makes the applicability in real-world scenarios questionable. Our results show that none of the existing techniques are adapted to upscale real-world HiC matrices, with HiCNN among them offering the best generalizability off-the-shelf. More importantly, we show that retraining these existing methods with real-world datasets improves the performance and generalizability on upscaling real-world datasets. Thus this observation provides solid evidence in favor of training the upcoming methods with real-world datasets to enhance the applicability of these methods.

![Evaluation Pipeline](figures/HicEvalFlowChart.drawio.png?raw=true "Pipeline")

## Datasets
In this evaluation setup we use nine HiC datasets that we collect from various sources. The table below summarizes the information of the datasets we have used in our evaluation setup. We recommend that you construct a data/hic_datasets/CELL_LINE where we store all the *.hic* files for that CELL_LINE. This would make it easier to manage paths easier later in the processing pipeline. 

```bash
    # Optional: These commands will help you create the aforementioned directory structure.
    mkdir data/
    mkdir data/hic_datasets/
    mkdir data/hic_datasets/GM12878/
    mkdir data/hic_datasets/IMR90/
    mkdir data/hic_datasets/K562/
    # Run these commands to create the data directory structure
```

| Cell Line | Source | Label | Read Counts | Link |
| --- | --- | --- | --- | --- |
| GM12878 | GEO    | GM12878-HRC   | 1,844,107,778 | [Link](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary%2Breplicate_combined_30.hic)  | 
| GM12878 | ENCODE | GM12878-LRC-1 | 42,453,795    | [Link](https://www.encodeproject.org/experiments/ENCSR382RFU/) |
| GM12878 | ENCODE | GM12878-LRC-2 | 37,079,587    | [Link](https://www.encodeproject.org/experiments/ENCSR382RFU/) | 
| GM12878 | ENCODE | GM12878-LRC-3 | 70,138,184    | [Link](https://www.encodeproject.org/experiments/ENCSR968KAY/) | 
| GM12878 | GEO    | GM12878-LRC-4 | 18,696,952    | [Link](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551582/suppl/GSM1551582_HIC033_30.hic) | 
| IMR90   | GEO    | IMR90-HRC     | 735,043,093   | [Link](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_IMR90_combined_30.hic) | 
| IMR90   | GEO    | IMR90-LRC-1   | 75,193,876    | [Link](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551606/suppl/GSM1551606_HIC057_30.hic) | 
| K562    | GEO    | K562-HRC      | 641,402,880   | [Link](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined_30.hic) | 
| K562    | GEO    | K562-LRC-1    | 44,882,605    | [Link](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551622/suppl/GSM1551622_HIC073_30.hic) | 

Note: All of these files are already filtered to have contact counts with MAPQ>= 30.

## Python Virtual Environment
First of all we create a python virtual environment and we progressively add packages to it to ensure we only download the required packages.

```bash
    python3 -m venv hic_eval # Create the python environment
    source hic_eval/bin/activate # To Activate the environment
```

## HiC File Pre-Processing
For this step, the first thing that we need to create is a directory that would house all the intermediate intra-chromosomal files that we will extract from the .hic file at 10Kb resolution. 

``` bash
    mkdir data/chromosome_files/
```

We then install all the required package for this module of our pipeline:
```bash 
    pip install numpy
    pip install hic-straw
    pip install scipy
    pip install torch
    pip install pandas
    pip install cooler
```

Third step we update the value of the DATA_DIRECTORY variable in the src/globals.py to the absolute path of the data directory. 

Fourth step, we need to make a minor modification in the straw.py file located in the directory hic_eval/lib64/python3.8/site-packages/straw/straw.py. We update the return signature of the function 'straw' from return [xActual, yActual, counts] to [xActual, yActual, counts] [c1Norm, c2Norm]. We need the position of the normalization vectors to condsense out the contact matrices by removing all the rows and columns that have no information on them. We have adopted this technique from the DeepHiC paper. However make sure that you install the correct version of hic-straw which is hic-straw==0.0.6. 

Lastly, we run the the preprocessing script 'data_parser.py' that has the function signautre: 
``` bash
usage: data_parser.py [-h] -f INPUT_FILE -o OUTPUT_PATH [-v] [-r RESOLUTION] [-n NORMALIZATION] [-a] [-d DATASET]

Read a HiC dataset file and extract all the intra chromosomal matrices. Currently only supports .hic format. (Will add support for .cool and .mcool eventually)

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input-file INPUT_FILE
                        Path to the input file
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to the output folder that will host the generated chromosome files
  -v, --verbose         Increases the information that is displayed on the terminal
  -r RESOLUTION, --resolution RESOLUTION
                        Resolution to sample the output matrices at, defaults to 10000
  -n NORMALIZATION, --normalization NORMALIZATION
                        Normalization method to use, currently only supported method is KR so it defaults to that
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)
  -d DATASET, --dataset DATASET
                        Which set of chromosomes to extract. There are four divisions: 1) test [1-8, 13-18] 2) valid [9-12] 3) test 19-22 and 4) all [1-22]. Defaults to extracting all chromosomes
```

Warning: This script eats up a lot of system memory and crashes when it runs out of it for larger chromosomes. So make, for a single threaded operation, that you have at least 32 GBs of system memory available. 


This shows a sample command to extract all chromosomes from the file GSM1551738_HIC193_30.hic at resolution 10Kb in folder data/chromosome_files/GM12878_hic193/. 

```bash
python data_parser.py -f data/hic_datasets/GM12878/GSM1551738_HIC193_30.hic -o data/chromosome_files/GM12878_hic193/ -v -r 10000 -d all

```


## Dataset creator
The dataset creator script 'dataset_creator.py' follows the following function protocol: 
```bash
usage: dataset_creator.py [-h] -b BASE_DIRECTORY -t TARGET_DIRECTORY -o OUTPUT_DIRECTORY -n NAME_OF_DATASET_FILE [-v] [--chunk CHUNK] [--stride STRIDE] [--bound BOUND] --cutoff-base CUTOFF_BASE
                          --cutoff-target CUTOFF_TARGET -d DATASET [-a]

Takes path to the chromosome directories and generates datasets that are compatible with the deeplearing based methods. It works in two steps, first step if a cutoff value is not provided computes the
cutoff value and then in the second step it divides the chromosomal matrices with provided chunk size values and strides to construct a dataset that contains high quality and low quality pairs.

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_DIRECTORY, --base-directory BASE_DIRECTORY
                        Path to the low quality chromosome matrices directory
  -t TARGET_DIRECTORY, --target-directory TARGET_DIRECTORY
                        Path to the high quality chromosome matrices directory
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Path to the directory where the generated dataset is stored
  -n NAME_OF_DATASET_FILE, --name-of-dataset-file NAME_OF_DATASET_FILE
                        Name of the generated dataset file
  -v, --verbose         Increases the information that is displayed on the terminal
  --chunk CHUNK         Size of the chunks cropped out from the chromosomal matrices
  --stride STRIDE       Stride length after each chunk, smaller than chunk size means that chunks have overlapping data
  --bound BOUND         Maximum distance in number of bins to consider from the center, defaults to 201, beacause most of the genomic are in that range
  --cutoff-base CUTOFF_BASE
                        Cutoff value to use for the base chromosome files (These cutoffs are specified in globals.py)
  --cutoff-target CUTOFF_TARGET
                        Cutoff value to use for the target chromosome files (These cutoffs are specified in globals.py)
  -d DATASET, --dataset DATASET
                        Dataset division, choices [train, valid, test, all]. all generates all three datasets
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)
```
The path to the directory of the low and high quality chromosome depends on the output path provided to the previous script to store the extracted chromosomes. Output directory can be any place, but in our example we have made another directory 'data/datasets' to store all the generated datasets. You can make the following directory by running command:
```bash
mkdir data/datasets
```

We followed a particular naming convention for the generated datasets files to ensure we had appropriately labeled files describing all the features of the dataset it contains. We followed the naming scheme '{noise}-{cell-line}-{base-dataset}-{target}-{chunksize}-{stride}-{bound}-{dataset-type}.npz'. Most of the features are self-explanatory, except: 

1. **Noise**: Noise explains the artificial noise injected on the base chromosome sub-matrices to force the model generalize better. We currently have None, Gaussian, Uniform and Random noise as possible place holders for this attribute. 
2. **Chunk Size**: Chunk size defines the size of the submatrices we crop out from the entire intrachromosomal matrix. 
3. **Stride**: Stride controls the overlap between each submatrix. Stride size equal to chunk has no overlap. 
 
4. **Bound**: Upper bound of genomic distance, e.g. 201 means 200 x 10kb. This ensures that all the chunks are maximum 2MB distal. 
5. **Dataset Type**: Dataset type, is the dataset a test, train or valid (validation) dataset. 

Last important parameters that goes into this script are the cutoff values. You can find all the required cutoff values for the datasets defined above in the table below. 

| HiC Base File | Label | Cutoff |
| --- | --- | --- |
| GSE63525_GM12878_insitu_primary_30.hic | GM12878_HRC-1 | 255 |
| ENCFF799QGA.hic | GM12878_LRC-1 | 90 | 
| ENCFF473CAA.hic | GM12878_LRC-2 | 87 |
| ENCFF227XJZ.hic | GM12878_LRC-3 | 96 |
| GSM1551582_HIC033_30.hic | GM12878-LRC-4 | 29 |
| GSE63525_IMR90_combined_30.hic | IMR90-HRC-1 | 255 |
| GSM1551606_HIC057_30.hic | IMR90-LRC-1 | 68 |
| GSE63525_K562_combined_30.hic | K562-HRC-1 | 255 |
| GSM1551622_HIC073_30.hic | K562-LRC-1 | 28 |



### Finding Cutoff values (Required only if adding new datasets)
If you plan to add your .hic file to this evaluation pipeline you might want to calculate the cutoff value for the datasets that would be needed for the dataset_creator.py script. We have added a small and simple utility that works on the directory created by the data_parser.py script and iterates over all the chromosome files and find the cutoff value for the dataset. We use 99.95th percentile on the values left after filtering out all the zero values from the contact matrix. The input signature for the script is: 

```bash
This script finds cutoff value for all chromosome files contained in a directory. This cutoff value is used to normalize HiC contact matrices

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_DIRECTORY, --base-directory BASE_DIRECTORY
                        Path to the chromosome matrices directory
  --percentile PERCENTILE
                        Percentile value, defaults to 99.95th percentile
  -v, --verbose         Increases the information that is displayed on the terminal
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)
```

### Creating Synthetic (Downsampled) Datasets (Optional)
So far in our pipeline we have been using and working with the real-world low read count (LRC) datasets. We have provided a script that is able to generate Downsampled datasets from the High Read Count datasets. The generate_synthetic_datasets.py has the following function signature: 
```bash
usage: generate_synthetic_datasets.py [-h] -b BASE_DIRECTORY -o OUTPUT_DIRECTORY --downsampling-ratio DOWNSAMPLING_RATIO [-v] [-a]

This script downsamples all the chromosome files in a directory by a uniform ratio

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_DIRECTORY, --base-directory BASE_DIRECTORY
                        Path to the chromosome matrices directory
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Path to the output directory where to store the downsampled chromosomes.
  --downsampling-ratio DOWNSAMPLING_RATIO
                        Downsampling ratio, for example: a ratio of 16 means all the read counts are scaled to their 1/16th value.
  -v, --verbose         Increases the information that is displayed on the terminal
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)

```

## Upscaling Pipeline
By this point we have created all the dataset files, now we can finally dive into upscaling them with different HiC resolution upcscaling methods. In this repository we provide five different methods. Gaussian Smoothing and DeepHiC works with datasets that are created with chunk size of 40 and stride of 40, HiCNN and HiCPlus work with datasets that are generated with chunnk size of 40 but with stride of 28. And lastly, VeHiCle works with datasets that have chunk size of 269 and stride of 257. In the upscaling scripts we have done no validation on the specifications of the datasets, consequently there is a chance if you input the dataset with wrong specifications the pipeline will crash. 

Our program has the following signature:
```bash 
usage: upscale_datasets.py [-h] -d PATH_TO_DATASET -o OUTPUT_DIRECTORY --model MODEL [-v] [-a]

This script provides an interface to upscale the HiC matrices through a set of deep learning based techniques.

optional arguments:
  -h, --help            show this help message and exit
  -d PATH_TO_DATASET, --path-to-dataset PATH_TO_DATASET
                        Path to the dataset
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Path to the output directory where to store the upscaled chromosomes.
  --model MODEL         Name of the model to use for upscaling the HiC matrices. Currently, we have smoothing, hicplus16, hicplus25, hicplus50, hicplus100, hicnn16, hicnn25, hicnn50, hicnn100, deephic16,
                        deephic25, deephic50, deephic100, vehicle, hicnn-gaussian, hicnn-random, hicnn-uniform, hicnn-synthetic-ensemble, hicnn-lrc and hicnn-lrc-ensemble
  -v, --verbose         Increases the information that is displayed on the terminal
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)

```

If you plan to use DeepHiC, then you would have to install the torchvision utility because DeepHiC uses VGG16 to compute perceptual loss. Use the command below to install that. 
```bash
pip install torchvision # This updates the pytorch version as well
```

### Gaussian Smoothing
To run with gaussian smoothing just pass on gaussian-smoothing as the input to the --model paramter. 
### HiCPlus
We are still in the process of integrating the HiCPlus script with our pipeline. Stay tuned! 
### HiCNN 
HiCNN has various versions available as specified in the function signature help above. Pick any and pass it to the --model paramter. 
### DeepHiC 
DeepHiC has only four versions available that are trained on the downsampled inputs. Pickup any and pass it to the --model paramter. 
### VeHiCle
We are still in the process of integrating the VeHiCLe script with our pipeline. Stay tuned! 

Note: The output generated by original HiCPlus and VeHiCle code bases can be repurposed to be used for our evaluation pipeline explained and described in the next step. 
Warning: Make sure the input datasets is compatible with the model you are using to upscale it, otherwise you can get wrong and faulty predictions (upscale). 

## Evaluation Pipeline
Finally to collect results, we run the evaluation pipeline on the chromosome files we have generated. The purpose of this script is to provide a unified interface to run all the evaluation metrics in a single automated manner. We run six correlation based metrics, four HiC similarity metrics and two downstream evaluation metrics. 

### Correlation Based Evaluations 
We have six correlation based metrics. 
1. MSE: Mean squared error, we use the sklearn library to import mean_square_error function.
2. MAE: Mean Average Error, we similar to MSE use sklearn library to import mean_average_error function. 
3. PSNR: Peak Signal to Noise Ratio, we use mean_square_error function to write our own version of PSNR that is bounded to 100. 
4. SSIM: Structural Similarity Index Metric, we skimage library to import the structural_similarity to compute SSIM score. 
5. Pearsons': Pearsons Correlation, we use scipy.stats library to import the pearsonsr function.
6. Spearmans': Spearmans Correlation, we use scipy.stats library to import the spearmansr function.


### HiC Specific Evaluations 
For HiC specific similarity metrics we setup the https://github.com/kundajelab/3DChromatin_ReplicateQC repository locally and installed all the required utilities. We however eventually plan to port all the code bases to Python to improve the integration to the pipeline and overall runtime. 


### Downstream Evaluations 
We setup two downstream analysis scripts. 
1. 3D Chromatin Reproducibility: We downloaded the https://github.com/BDM-Lab/3DMax repository and placed the repository in folder other_tools/3DMax/. If you change the directory you install 3DMax in, make sure you update its path in the globals.py script. 
2. Significant Interactions: We downloaded the FitHiC2 tool from https://github.com/ay-lab/fithic repository. We installed it using the pip and that provided us a CLI access. 

To run all of these evaluations, we run the script 
``` bash
usage: run_evaluations.py [-h] -t PATH_TO_TARGET_CHROMOSOMES -b PATH_TO_BASE_CHROMOSOMES -c1 TARGET_CUTOFF_VALUE -c2 BASE_CUTOFF_VALUE -e NAME_OF_THE_EXPERIMENT [--correlation-analysis] [--hic-analysis]
                          [--downstream-analysis] -r [-a]

This script runs all the evaluation metrics and store the results in a file

optional arguments:
  -h, --help            show this help message and exit
  -t PATH_TO_TARGET_CHROMOSOMES, --path-to-target-chromosomes PATH_TO_TARGET_CHROMOSOMES
                        Path to the High Quality, High Res or High Read count chromosomes.
  -b PATH_TO_BASE_CHROMOSOMES, --path-to-base-chromosomes PATH_TO_BASE_CHROMOSOMES
                        Path to the either upscaled or baseline chromsome directory
  -c1 TARGET_CUTOFF_VALUE, --target-cutoff-value TARGET_CUTOFF_VALUE
                        Cutoff value for the target chromosomes
  -c2 BASE_CUTOFF_VALUE, --base-cutoff-value BASE_CUTOFF_VALUE
                        Cutoff value for the base chromosomes
  -e NAME_OF_THE_EXPERIMENT, --name-of-the-experiment NAME_OF_THE_EXPERIMENT
                        Results file would contain entry with this name
  --correlation-analysis
                        Run Correlation analysis
  --hic-analysis        Run HiC similarity analysis
  --downstream-analysis
                        Run downstream analysis
  -r, --results-file-name
                        Name of the results file that would be generated in results/
  -a, --absolute        Is the provided path an absolute path, if this flag is set to false then parser appends the DATA_DIRECTORY path (its defined in the src/globals.py file)
```


For the upscaled HiC matrices, use cutoff value of 255 we do not modify (or clamp) the distribution to fairly evaluate the models. 













