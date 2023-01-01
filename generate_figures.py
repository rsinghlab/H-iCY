# imports 
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from scipy.stats import ttest_ind

from src.format_handler import read_npz_file
from src.utils import create_entire_path_directory

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

cell_lines = ['GM12878', 'IMR90', 'K562']

generated_results_output_path = '/home/murtaza/Documents/hic_upscaling_evaluation/results/figures/new_figures/'

visualization_regions = {
    19: 2950,
    20: 3170,
    21: 1900,
    22: 4100,
}

methods_dictionary = {
    'smoothing': {
        'Name': 'Smoothing',
        'Color': '#D81B60',
        'Symbol': 'o',
    },
    'hicplus': {
        'Name': 'HiCPlus',
        'Color': '#1E88E5',
        'Symbol': 's',  
    },
    'hicnn': {
        'Name': 'HiCNN',
        'Color': '#FFC107',
        'Symbol': '*',
    },
    'hicnn2':{
        'Name': 'HiCNN2',
        'Color': '#004D40',
        'Symbol': '*',
    },
    'deephic': {
        'Name': 'DeepHiC',
        'Color': '#8E5DF7',
        'Symbol': '^',
    },
    'vehicle': {
        'Name': 'VeHiCLe',
        'Color': '#229F59',
        'Symbol': 'd',
    },
}
metric_replicate_upperbound = {
    'MSE': 0.0025,
    'MAE': 0.0269,
    'PSNR': 26.8285,
    'SSIM': 0.7403,
    'PCC': 0.9697,
    'SCC': 0.8584, 
    'hicrep': 0.9822,
    'hic-spector': 0.7030,
    'genomedisco': 0.9538,
    'quasar-rep': 0.9386,
    '3d_reconstruction_tmscore': 0.8024,
    'Loops': 0.8745,
    'Borders': 0.8601,
    'Hairpins': 0.8449
}

lrc_datasets_to_sparsity = {
    'encode0': 44,
    'encode1': 50,
    'encode2': 25, 
    'hic026': 9,
    'hic033': 100,
    'hic057': 10,
    'hic073': 14,
}


def parse_line(line):
    method, dataset  = line.split(':')[:2]
    results = ':'.join(line.split(':')[2:])
    if 'Aggregate' not in results:
        results = ast.literal_eval(results)
        return (method, dataset, results) 
    else:
        results = ast.literal_eval(results)
        results = results['Aggregate']
        feature_results = {
            'Loops': results['Loops']['f1'],
            'Borders': results['Borders']['f1'],
            'Hairpins': results['Hairpins']['f1'],
        }
        return (method, dataset, feature_results)



def visualize_hic_data(hic_matrix, output_path, region, percentile=98.0, range=200):
    hic_matrix = hic_matrix[region: region+range, region: region+range]
    cutoff_value = np.percentile(hic_matrix, percentile)

    hic_matrix = np.minimum(cutoff_value, hic_matrix)
    hic_matrix = np.maximum(hic_matrix, 0)

    plt.matshow(hic_matrix, cmap=REDMAP)
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'i-{}_j-{}.png'.format(region, region)), bbox_inches='tight')
    plt.close()

############################################### FIGURE 2 #################################################################
# Sub figure B: This figure contains a line plot showing how the SSIM scores vary as we increase downsampling ratio for three cell lines 

def create_line_graph_downsampled_datasets(results_path, metric):
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))

    cell_line = results_path.split('/')[1].split('_')[0]

    # We hard code the x-values and ignore all other results
    x_values = ['16', '25', '50', '100']
    y_min = 9999999999
    y_max = -999999999

    

    for method in methods_dictionary.keys():
        values = []
        filtered_data = list(filter(lambda x: method == x[0].split('-')[0], data))
        for _, _, result  in filtered_data:
            values.append(np.mean(result[metric]))
        if min(values) <= y_min:
            y_min = min(values)
        if max(values) >= y_max:
            y_max = max(values)
        
        plt.plot(
            x_values, values, 
            color=methods_dictionary[method]['Color'], 
            linewidth=3.0, 
            marker=methods_dictionary[method]['Symbol'], 
            markersize=10,
            label=methods_dictionary[method]['Name'], 
        )
    upper_bound_line = [metric_replicate_upperbound[metric]]*len(x_values)
    plt.plot(x_values, upper_bound_line, linewidth=3.0, linestyle='--', color='black')


    y_custom_ticks = np.linspace(y_min, y_max, 3, dtype=float)
    y_custom_ticks = np.round(y_custom_ticks, 2)

    plt.xticks(fontsize= 15)
    plt.yticks(y_custom_ticks, y_custom_ticks, fontsize= 15)
    plt.xlabel('Downsampling Ratio', fontsize= 20)
    plt.ylabel(metric, fontsize= 20)
    plt.legend(fontsize = 15)

    plt.tight_layout()
    
    plt.savefig('results/figures/new_figures/downsampled-methods_downsampled_datasets/{}_{}.png'.format(metric, cell_line))
    plt.cla()
    plt.clf()
    plt.close()

result_paths =[
    'results/gm12878_downsampled-dataset_results.txt',
    'results/imr90_downsampled-dataset_results.txt',
    'results/k562_downsampled-dataset_results.txt'
]

metrics = ['MSE', 'MAE', 'SSIM', 'PSNR', 'PCC', 'SCC', 'hicrep', 'hic-spector', 'genomedisco', 'quasar-rep']
    
for result_path in result_paths:
    for metric in metrics:
        create_line_graph_downsampled_datasets(result_path, metric)



# Scripts for plotting visualization of Hi-C contact matrices 
chromosome_files_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/synthetic/'

def plot_hic_matrices_for_downsampled_datasets(chromosome_files_path, downsampling_ratio=50, cell_line='GM12878', chr=22):
    base_chrom_path = os.path.join(chromosome_files_path, '{}_synthetic{}'.format(cell_line, downsampling_ratio), 'chr{}.npz'.format(chr))
    target_chrom_path = os.path.join(chromosome_files_path, '{}_rao_et_al'.format(cell_line), 'chr{}.npz'.format(chr) )
    
    # Plot target chromosomes
    data, _ = read_npz_file(target_chrom_path)
    output_path = os.path.join(generated_results_output_path, 'downsampled-methods_downsampled_datasets/visualizations/target/{}/chr{}'.format(cell_line, chr))
    create_entire_path_directory(output_path)
    visualize_hic_data(data, output_path, visualization_regions[chr])
    
    # Plot base chromosomes
    data, _ = read_npz_file(base_chrom_path)
    output_path = os.path.join(generated_results_output_path, 'downsampled-methods_downsampled_datasets/visualizations/synthetic{}/{}/chr{}'.format(downsampling_ratio, cell_line, chr))
    create_entire_path_directory(output_path)
    visualize_hic_data(data, output_path, visualization_regions[chr])
    
    for method in methods_dictionary:
        method_name = ''
        if method == 'smoothing':
            method_name = 'smoothing-gaussian'
        elif method == 'vehicle':
            method_name = method
        else:
            method_name = method + '-' + str(downsampling_ratio)

        chrom_path = os.path.join(chromosome_files_path, '{}_{}_synthetic{}'.format(cell_line, method_name, downsampling_ratio), 'chr{}.npz'.format(chr))
        data, _ = read_npz_file(chrom_path)
        output_path = os.path.join(generated_results_output_path, 'downsampled-methods_downsampled_datasets/visualizations/{}/{}/chr{}'.format(method, cell_line, chr))
        create_entire_path_directory(output_path)
        visualize_hic_data(data, output_path, visualization_regions[chr])


# for cell_line in cell_lines:
#     for chr in visualization_regions:
#         for downsampling_ratio in [16, 25, 50, 100]:
#             plot_hic_matrices_for_downsampled_datasets(chromosome_files_path, downsampling_ratio, cell_line, chr)


############################################### FIGURE 3 #################################################################
# Scripts show the decay of score in downsampled datasets vs the LRC datasets make it easier to understand that 
chromosome_files_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/'
# Difference of two hic-matrices on a log-scale 
def plot_hic_matrices_difference_for_downsampled_datasets(chromosome_files_path, downsampling_ratio=50, lrc_dataset='encode1', cell_line='GM12878', chr=22):
    synthetic_chrom_path =  os.path.join(chromosome_files_path, 'synthetic', '{}_synthetic{}'.format(cell_line, downsampling_ratio), 'chr{}.npz'.format(chr))
    lrc_chrom_path =         os.path.join(chromosome_files_path, 'real',     '{}_{}'.format(cell_line, lrc_dataset), 'chr{}.npz'.format(chr))


    output_path = '/home/murtaza/Documents/hic_upscaling_evaluation/results/figures/new_figures/downsampled-data_vs_lrc-data/{}/{}/chr{}'.format('synthetic', cell_line, chr)
    create_entire_path_directory(output_path)
    synthetic, _ = read_npz_file(synthetic_chrom_path)
    visualize_hic_data(synthetic, output_path, visualization_regions[chr])

    output_path = '/home/murtaza/Documents/hic_upscaling_evaluation/results/figures/new_figures/downsampled-data_vs_lrc-data/{}/{}/chr{}'.format('real', cell_line, chr)
    create_entire_path_directory(output_path)
    lrc, _ = read_npz_file(lrc_chrom_path)
    visualize_hic_data(lrc, output_path, visualization_regions[chr])


    synthetic = np.log10(synthetic[visualization_regions[chr]: visualization_regions[chr]+200, visualization_regions[chr]: visualization_regions[chr]+200] + 0.0000000001)
    lrc = np.log10(lrc[visualization_regions[chr]: visualization_regions[chr]+200, visualization_regions[chr]: visualization_regions[chr]+200] + 0.0000000001)

    difference = np.abs(synthetic - lrc)

    
    difference = difference
    heatmap = plt.pcolor(difference)
    plt.colorbar(heatmap)
    plt.axis('off')
    plt.savefig(os.path.join('results/figures/new_figures/downsampled-data_vs_lrc-data/{}_{}_{}_chr{}.png'.format(cell_line, downsampling_ratio, lrc_dataset, chr)), bbox_inches='tight')
    plt.close()



def plot_score_trends_comparison_graph(synthetic_results_file, lrc_results_file, metric):
    ##dcd8a7 lrc dwnspled #c10600
    x_values = [8, 16, 25, 50, 100]
    synthetic_datasets_y_values = []
    cell_line = synthetic_results_file.split('/')[1].split('_')[0]

    data = open(synthetic_results_file).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))
    data = list(filter(lambda x: 'baseline' == x[0].split('-')[0], data))
    for _, _, result  in data:
        synthetic_datasets_y_values.append(np.mean(result[metric]))
    
    print(metric, x_values, synthetic_datasets_y_values)


    lrc_datasets_values = []
    data = open(lrc_results_file).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))
    data = list(filter(lambda x: 'baseline' == x[0].split('-')[0], data))
    for _, dataset, result  in data:
        if dataset != 'replicate':
            lrc_datasets_values.append((lrc_datasets_to_sparsity[dataset], np.mean(result[metric])))
    
    lrc_datasets_values = sorted(lrc_datasets_values, key = lambda x: x[0])
    
    plt.plot(np.array(x_values), np.array(synthetic_datasets_y_values), c='#dcd8a7',  linewidth=4.0, markersize=20)
    
    if len(lrc_datasets_values) > 1:
        plt.plot(list(map(lambda x: x[0], lrc_datasets_values)), list(map(lambda x: x[1], lrc_datasets_values)), c='#c10600',  linewidth=4.0, markersize=20)
    else:
        plt.scatter(list(map(lambda x: x[0], lrc_datasets_values)), list(map(lambda x: x[1], lrc_datasets_values)), c='#c10600', s=200)

    plt.xlabel('Read Count Ratio', size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(metric, size=16)


    plt.savefig(os.path.join('results/figures/new_figures/downsampled-data_vs_lrc-data/{}_{}.png'.format(cell_line, metric)), bbox_inches='tight')
    plt.close()


def plot_bio_feature_trends_comparison_graph(synthetic_results_file, lrc_results_file, feature):
    x_values = [8, 16, 25, 50, 100]
    synthetic_datasets_y_values = []
    cell_line = synthetic_results_file.split('/')[1].split('_')[0]

    data = open(synthetic_results_file).read().split('\n')[:-1]
    data = list(filter(lambda x: feature in x, data))
    data = list(map(parse_line, data))
    data = list(filter(lambda x: 'baseline' == x[0].split('-')[0], data))
    for _, dataset, result  in data:
        if dataset == 'replicate':
            continue
        synthetic_datasets_y_values.append(np.mean(result[feature]))

    lrc_datasets_values = []
    data = open(lrc_results_file).read().split('\n')[:-1]
    data = list(filter(lambda x: feature in x, data))
    data = list(map(parse_line, data))
    data = list(filter(lambda x: 'baseline' == x[0].split('-')[0], data))
    for _, dataset, result  in data:
        if dataset != 'replicate':
            lrc_datasets_values.append((lrc_datasets_to_sparsity[dataset], np.mean(result[feature])))
        print(dataset, result)


    lrc_datasets_values = sorted(lrc_datasets_values, key = lambda x: x[0])
    
    plt.plot(np.array(x_values), np.array(synthetic_datasets_y_values), c='#dcd8a7', linewidth=4.0, markersize=20)
    
    if len(lrc_datasets_values) > 1:
        plt.plot(list(map(lambda x: x[0], lrc_datasets_values)), list(map(lambda x: x[1], lrc_datasets_values)), c='#c10600', linewidth=4.0, markersize=20)
    else:
        plt.scatter(list(map(lambda x: x[0], lrc_datasets_values)), list(map(lambda x: x[1], lrc_datasets_values)), c='#c10600', s=200)

    plt.xlabel('Read Count Ratio', size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('{} F1'.format(feature), size=16)

    plt.savefig(os.path.join('results/figures/new_figures/downsampled-data_vs_lrc-data/{}_{}.png'.format(cell_line, feature)), bbox_inches='tight')
    plt.close()

# for chr in [19, 20, 21, 22]:
#     plot_hic_matrices_difference_for_downsampled_datasets(chromosome_files_path, chr=chr)
# metrics = ['MSE', 'MAE', 'SSIM', 'PSNR', 'PCC', 'SCC', 'hicrep', 'hic-spector', 'genomedisco', 'quasar-rep']
# for metric in metrics:
#     plot_score_trends_comparison_graph('results/gm12878_downsampled-dataset_results.txt', 'results/gm12878_real-world-dataset_results.txt', metric)
#     plot_score_trends_comparison_graph('results/imr90_downsampled-dataset_results.txt', 'results/imr90_real-world-dataset_results.txt', metric)
#     plot_score_trends_comparison_graph('results/k562_downsampled-dataset_results.txt', 'results/k562_real-world-dataset_results.txt', metric)

# features = ['Loops', 'Borders', 'Hairpins']
# for feature in features:
#     plot_bio_feature_trends_comparison_graph('results/gm12878_downsampled-dataset_results.txt', 'results/gm12878_real-world-dataset_results.txt', feature)
#     plot_bio_feature_trends_comparison_graph('results/imr90_downsampled-dataset_results.txt', 'results/imr90_real-world-dataset_results.txt', feature)
#     plot_bio_feature_trends_comparison_graph('results/k562_downsampled-dataset_results.txt', 'results/k562_real-world-dataset_results.txt', feature)















################################################### Figure 4 ################################################################3
# With this figure we argue that the models trained with downsampled datasets tend to not perform well on the LRC datasets. Or at least not as well as they did on Downsampled datasets
gm12878_results_path = 'results/gm12878_real-world-dataset_results.txt'
imr90_results_path = 'results/imr90_real-world-dataset_results.txt'
k562_results_path = 'results/k562_real-world-dataset_results.txt'

def percentage_change(a, b):
    a = np.mean(a)
    b = np.mean(b)
    return ((b - a)/b) * 100


def create_line_graph_downsampled_models_lrc_dataset(results_path, metric, downsampling_ratio=100):
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))

    cell_line = results_path.split('/')[1].split('_')[0]

    # We hard code the x-values and ignore all other results
    y_min = 9999999999
    y_max = -999999999

    for method in methods_dictionary.keys():
        compiled_results = {}
        method_name = ''
        if method == 'vehicle' or method == 'smoothing' or method == 'hicnn2':
            method_name = method
        else:
            method_name = method + '-' + str(downsampling_ratio)
        
        filtered_data = list(filter(lambda x: method_name == x[0], data))
        
        for mtd, dataset, result  in filtered_data:
            if dataset not in compiled_results.keys():
                compiled_results[dataset] = []
            compiled_results[dataset].append(np.mean(result[metric]))
        
            
        for key in compiled_results.keys():
            compiled_results[key] = np.mean(compiled_results[key])
        

        x_values = list(map(lambda x: lrc_datasets_to_sparsity[x], compiled_results.keys()))
        y_values = list(compiled_results.values())
        
        values =  sorted(zip(x_values, y_values), key = lambda x: x[0])
        print(values, method)

        plt.plot(
            list(map(lambda x: str(x[0]), values)), 
            list(map(lambda x: x[1], values)), 
            color=methods_dictionary[method]['Color'], 
            linewidth=3.0, 
            marker=methods_dictionary[method]['Symbol'], 
            markersize=10,
            label=methods_dictionary[method]['Name'],
        )

        if min(y_values) <= y_min:
            y_min = min(y_values)
        if max(y_values) >= y_max:
            y_max = max(y_values)
    

    
    upper_bound_line = [metric_replicate_upperbound[metric]]*len(x_values)
    if max(upper_bound_line) >= y_max:
        y_max = max(upper_bound_line)
    
    plt.plot(list(map(lambda x: str(x[0]), values)), upper_bound_line, linewidth=3.0, linestyle='--', color='black', label='replicate')

    y_custom_ticks = np.linspace(y_min, y_max, 3, dtype=float)
    y_custom_ticks = np.round(y_custom_ticks, 2)

    plt.xticks(fontsize= 20)
    plt.yticks(y_custom_ticks, y_custom_ticks, fontsize= 20)
    plt.xlabel('Sparsity', fontsize= 20)
    plt.ylabel(metric, fontsize= 20)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('results/figures/new_figures/downsampled-methods_lrc_datasets/{}_{}.png'.format(metric, cell_line))
    plt.cla()
    plt.clf()
    plt.close()


def create_barchart_downsampled_models_lrc_dataset(results_path, hic_file, downsampled_results_path, metric, downsampling_ratio=100, xlabel=True):
    # Get HiCNN upper bound 
    data = open(downsampled_results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))
    filtered_data = list(filter(lambda x: 'hicnn-{}'.format(downsampling_ratio) == x[0], data))
    hicnn_upper_bound = filtered_data[0][-1][metric]
    replicate_upper_bound = [metric_replicate_upperbound[metric]]

    # Get results on real-world LRC dataset
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))

    cell_line = results_path.split('/')[1].split('_')[0]

    # We hard code the x-values and ignore all other results
    y_min = 9999999999
    y_max = -999999999
    all_y_values = []
    all_methods = []
    all_colors = []
    all_pvalues = []
    all_precent_decrease = []

    for method in methods_dictionary.keys():
        compiled_results = {}
        method_name = ''
        if method == 'vehicle' or method == 'smoothing' or method =='hicnn2':
            method_name = method
        else:
            method_name = method + '-' + str(downsampling_ratio)
        
        filtered_data = list(filter(lambda x: method_name == x[0], data))
        for mtd, dataset, result  in filtered_data:
            if dataset != hic_file:
                continue
                
            if dataset not in compiled_results.keys():
                compiled_results[dataset] = []
            compiled_results[dataset].append(result[metric])
            
            
        for key in compiled_results.keys():
            t_test = ttest_ind(hicnn_upper_bound, compiled_results[key][0])
            all_precent_decrease.append(percentage_change(compiled_results[key][0], hicnn_upper_bound))

            if t_test.pvalue <= 0.005:
                all_pvalues.append('*')
            else:
                all_pvalues.append(' ')

            compiled_results[key] = np.mean(compiled_results[key][0])
        
        all_y_values.append(list(compiled_results.values())[0])
        all_methods.append(method_name.split('-')[0])
        all_colors.append(methods_dictionary[method]['Color'])

    
    
    
    print(metric ,all_methods, all_precent_decrease, all_pvalues)

    
    x_pos = np.arange(len(all_y_values))
    
    hicnn_upper_bound_line = [np.mean(hicnn_upper_bound)]*len(x_pos)
    plt.plot(x_pos, hicnn_upper_bound_line, linewidth=3.0, linestyle='--', color='#FFC107')

    if max(hicnn_upper_bound_line) >= y_max:
        y_max = max(hicnn_upper_bound_line)
    if min(all_y_values) <= y_min:
        y_min = min(all_y_values)

    y_custom_ticks = np.linspace(0, y_max, 3, dtype=float)
    y_custom_ticks = np.round(y_custom_ticks, 2)

    for i in range(len(x_pos)):
        plt.text(i, all_y_values[i], all_pvalues[i], ha = 'center')

    plt.bar(x_pos, all_y_values, color=all_colors)
    plt.xticks(x_pos, all_methods)
    plt.xticks(fontsize= 20, rotation=90)
    plt.yticks(y_custom_ticks, y_custom_ticks, fontsize= 25)
    plt.ylabel(metric, fontsize= 20)
    plt.tight_layout()
    
    plt.savefig('results/figures/new_figures/downsampled-methods_lrc_datasets/barchart_{}_{}.png'.format(metric, cell_line))
    plt.cla()
    plt.clf()
    plt.close()

for metric in metrics:
    create_line_graph_downsampled_models_lrc_dataset(gm12878_results_path, metric)
    create_barchart_downsampled_models_lrc_dataset(gm12878_results_path, 'encode1', 'results/gm12878_downsampled-dataset_results.txt', metric, downsampling_ratio=100, xlabel=False)
    create_barchart_downsampled_models_lrc_dataset(imr90_results_path, 'hic057', 'results/imr90_downsampled-dataset_results.txt', metric, downsampling_ratio=16, xlabel=False)
    create_barchart_downsampled_models_lrc_dataset(k562_results_path, 'hic073', 'results/k562_downsampled-dataset_results.txt', metric, downsampling_ratio=16)





def plot_hic_matrices_for_lrc_datasets(chromosome_files_path, dataset, downsampling_ratio=50, cell_line='GM12878', chr=22): 
    for method in methods_dictionary:
        method_name = ''
        if method == 'smoothing':
            method_name = 'smoothing-gaussian'
        elif method == 'vehicle':
            method_name = method
        else:
            method_name = method + '-' + str(downsampling_ratio)
        
        if method == 'vehicle' and dataset == 'hic026':
            return 

        chrom_path = os.path.join(chromosome_files_path, '{}_{}_{}'.format(cell_line, method_name, dataset), 'chr{}.npz'.format(chr))
        data, _ = read_npz_file(chrom_path)
        output_path = os.path.join(generated_results_output_path, 'downsampled-methods_lrc_datasets/visualizations/{}/{}/chr{}'.format(method, cell_line, chr))
        create_entire_path_directory(output_path)
        visualize_hic_data(data, output_path, visualization_regions[chr])


    



chromosome_files_path = '/media/murtaza/ubuntu/hic_data/chromosome_files/real/'

# for cell_line in cell_lines:
#     for chr in visualization_regions:
#         plot_hic_matrices_for_lrc_datasets(chromosome_files_path, 'encode1', 100, cell_line, chr)






######################################################### Figure 5 ##############################################

lrc_methods_dictionary = {
    'hicnn-100': {
        'Name': 'Downsampled',
        'Color': '#FFC107',
        'Symbol': '*',
    },
    'hicnn-encode0': {
        'Name': 'LRC-1',
        'Color': '#8885DC',
        'Symbol': 'X',
    },
    'hicnn-encode2': {
        'Name': 'LRC-3',
        'Color': '#55E6B1',
        'Symbol': '<',
    },
    'hicnn-hic026': {
        'Name': 'LRC-4',
        'Color': '#CF4D23',
        'Symbol': '>',
    },
    'hicnn-synthetic-ensemble': {
        'Name': 'Ens-DSP',
        'Color': '#85033C',
        'Symbol': "1",
    },
    'hicnn-real-ensemble': {
        'Name': 'Ens-LRC',
        'Color': '#678E88',
        'Symbol': "2",
    },
    'hicnn-gaussian-noise': {
        'Name': 'Gaussian-N',
        'Color': '#16188a',
        'Symbol': "3",
    },
    'hicnn-uniform-noise': {
        'Name': 'Uniform-N',
        'Color': '#16188a',
        'Symbol': "4",
    },
    'hicnn-random-noise': {
        'Name': 'Random-N',
        'Color': '#16188a',
        'Symbol': "8",
    },
    
}


def plot_hic_matrices_for_retrained_models_datasets(chromosome_files_path, dataset, downsampling_ratio=50, cell_line='GM12878', chr=22): 
    for method in lrc_methods_dictionary:
        
        chrom_path = os.path.join(chromosome_files_path, '{}_{}_{}'.format(cell_line, method, dataset), 'chr{}.npz'.format(chr))
        data, _ = read_npz_file(chrom_path)
        output_path = os.path.join(generated_results_output_path, 'retrained-methods-lrc-datasets/visualizations/{}/{}/chr{}'.format(method, cell_line, chr))
        create_entire_path_directory(output_path)
        visualize_hic_data(data, output_path, visualization_regions[chr])



# for chr in visualization_regions:
#     plot_hic_matrices_for_retrained_models_datasets(chromosome_files_path, 'encode1', 100, 'GM12878', chr)

def create_barchart_retrained_models_lrc_dataset(results_path, hic_file, metric, downsampling_ratio=100, xlabel=True, methods_dictionary=lrc_methods_dictionary):
    # Get HiCNN upper bound 
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))
    filtered_data = list(filter(lambda x: 'hicnn-{}'.format(downsampling_ratio) == x[0], data))
    hicnn_upper_bound = filtered_data[0][-1][metric]
    
    # Get results on real-world LRC dataset
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))

    cell_line = results_path.split('/')[1].split('_')[0]

    # We hard code the x-values and ignore all other results
    y_min = 9999999999
    y_max = -999999999
    all_y_values = []
    all_methods = []
    all_colors = []
    all_pvalues = []
    all_precent_decrease = []

    for method in methods_dictionary.keys():
        compiled_results = {}
        # if method == 'hicnn-100':
        #     continue
        
        filtered_data = list(filter(lambda x: method == x[0], data))

        for mtd, dataset, result  in filtered_data:
            if dataset != hic_file:
                continue
                
            if dataset not in compiled_results.keys():
                compiled_results[dataset] = []
            compiled_results[dataset].append(result[metric])
            
            
        for key in compiled_results.keys():
            t_test = ttest_ind(hicnn_upper_bound, compiled_results[key][0])
            all_precent_decrease.append(percentage_change(compiled_results[key][0], hicnn_upper_bound))

            
            if t_test.pvalue <= 0.005:
                all_pvalues.append('*')
            else:
                all_pvalues.append(' ')

            compiled_results[key] = np.mean(compiled_results[key][0])

        all_y_values.append(list(compiled_results.values())[0])
        all_methods.append(methods_dictionary[method]['Name'])
        all_colors.append(methods_dictionary[method]['Color'])

    x_pos = np.arange(len(all_y_values))
    
    print(metric ,all_methods, all_precent_decrease, all_pvalues)

    # hicnn_upper_bound_line = [np.mean(hicnn_upper_bound)]*len(x_pos)
    # plt.plot(x_pos, hicnn_upper_bound_line, linewidth=3.0, linestyle='--', color='#FFC107')

    # if max(hicnn_upper_bound_line) >= y_max:
    #     y_max = max(hicnn_upper_bound_line)
    if max(all_y_values) >= y_max:
        y_max = max(all_y_values)
    if min(all_y_values) <= y_min:
        y_min = min(all_y_values)

    y_custom_ticks = np.linspace(0, y_max, 3, dtype=float)
    y_custom_ticks = np.round(y_custom_ticks, 2)




    # for i in range(len(x_pos)):
    #     plt.text(i, all_y_values[i], all_pvalues[i], ha = 'center')

    plt.bar(x_pos, all_y_values, color=all_colors)
    plt.xticks(x_pos, all_methods)
    plt.xticks(fontsize= 25, rotation = 90)
    plt.yticks(y_custom_ticks, y_custom_ticks, fontsize= 25)
    
    if metric == '3d_reconstruction_tmscore':
        metric = 'TM-Score'
    plt.ylabel(metric, fontsize= 20)
    

    plt.tight_layout()
    
    plt.savefig('results/figures/new_figures/retrained-methods-lrc-datasets/barchart_{}_{}.png'.format(metric, cell_line))
    plt.cla()
    plt.clf()
    plt.close()




def create_line_graph_lrc_models_lrc_dataset(results_path, metric, methods_dictionary=lrc_methods_dictionary):
    data = open(results_path).read().split('\n')[:-1]
    data = list(filter(lambda x: metric in x, data))
    data = list(map(parse_line, data))

    cell_line = results_path.split('/')[1].split('_')[0]

    # We hard code the x-values and ignore all other results
    y_min = 9999999999
    y_max = -999999999

    for method in methods_dictionary.keys():
        compiled_results = {}
        
        filtered_data = list(filter(lambda x: method == x[0], data))
        
        for mtd, dataset, result  in filtered_data:
            if dataset not in compiled_results.keys():
                compiled_results[dataset] = []
            compiled_results[dataset].append(np.mean(result[metric]))
        
            
        for key in compiled_results.keys():
            compiled_results[key] = np.mean(compiled_results[key])
        

        x_values = list(map(lambda x: lrc_datasets_to_sparsity[x], compiled_results.keys()))
        y_values = list(compiled_results.values())

        values =  sorted(zip(x_values, y_values), key = lambda x: x[0])
        
        plt.plot(
            list(map(lambda x: str(x[0]), values)), 
            list(map(lambda x: x[1], values)), 
            color=methods_dictionary[method]['Color'], 
            linewidth=3.0, 
            marker=methods_dictionary[method]['Symbol'], 
            markersize=10,
            label= methods_dictionary[method]['Name']
        )

        if min(y_values) <= y_min:
            y_min = min(y_values)
        if max(y_values) >= y_max:
            y_max = max(y_values)
    
    plt.legend()
    
    upper_bound_line = [metric_replicate_upperbound[metric]]*len(x_values)
    if max(upper_bound_line) >= y_max:
        y_max = max(upper_bound_line)
    
    plt.plot(list(map(lambda x: str(x[0]), values)), upper_bound_line, linewidth=3.0, linestyle='--', color='black', label='replicate')

    y_custom_ticks = np.linspace(y_min, y_max, 3, dtype=float)
    y_custom_ticks = np.round(y_custom_ticks, 2)

    plt.xticks(fontsize= 20)
    plt.yticks(y_custom_ticks, y_custom_ticks, fontsize= 20)
    plt.xlabel('Sparsity', fontsize= 20)
    if metric == '3d_reconstruction_tmscore':
        metric = 'TM-Score'
        
    plt.ylabel(metric, fontsize= 20)
    


    plt.tight_layout()
    
    plt.savefig('results/figures/new_figures/retrained-methods-lrc-datasets/{}_{}.png'.format(metric, cell_line))
    plt.cla()
    plt.clf()
    plt.close()



for metric in metrics:
    create_line_graph_lrc_models_lrc_dataset(gm12878_results_path, metric)
    create_barchart_retrained_models_lrc_dataset(gm12878_results_path, 'encode1', metric, downsampling_ratio=100, xlabel=False)
    create_barchart_retrained_models_lrc_dataset(imr90_results_path, 'hic057',  metric, downsampling_ratio=16, xlabel=False)
    create_barchart_retrained_models_lrc_dataset(k562_results_path, 'hic073',  metric, downsampling_ratio=16)






downstream_methods_dictionary = {
    'smoothing': {
        'Name': 'Smoothing',
        'Color': '#D81B60',
        'Symbol': 'o',
    },
    'hicplus-100': {
        'Name': 'HiCPlus',
        'Color': '#1E88E5',
        'Symbol': 's',  
    },
    'hicnn2':{
        'Name': 'HiCNN2',
        'Color': '#004D40',
        'Symbol': '*',
    },
    'deephic-100': {
        'Name': 'DeepHiC',
        'Color': '#8E5DF7',
        'Symbol': '^',
    },
    'vehicle': {
        'Name': 'VeHiCLe',
        'Color': '#229F59',
        'Symbol': 'd',
    },
    'hicnn-100': {
        'Name': 'HiCNN',
        'Color': '#FFC107',
        'Symbol': '*',
    },
    # 'hicnn-encode0': {
    #     'Name': 'LRC-1',
    #     'Color': '#8885DC',
    #     'Symbol': 'X',
    # },
    'hicnn-encode2': {
        'Name': 'LRC-3',
        'Color': '#55E6B1',
        'Symbol': '<',
    },
    # 'hicnn-hic026': {
    #     'Name': 'LRC-4',
    #     'Color': '#CF4D23',
    #     'Symbol': '>',
    # },
    # 'hicnn-synthetic-ensemble': {
    #     'Name': 'Ens-DSP',
    #     'Color': '#85033C',
    #     'Symbol': "1",
    # },
    # 'hicnn-real-ensemble': {
    #     'Name': 'Ens-LRC',
    #     'Color': '#678E88',
    #     'Symbol': "2",
    # },
   
}


# for feature in ['Loops', 'Borders', 'Hairpins']:
#     create_line_graph_lrc_models_lrc_dataset(gm12878_results_path, feature, methods_dictionary=downstream_methods_dictionary)

#     create_barchart_retrained_models_lrc_dataset(imr90_results_path, 'hic057',  feature, downsampling_ratio=16, xlabel=False, methods_dictionary=downstream_methods_dictionary)
#     create_barchart_retrained_models_lrc_dataset(k562_results_path, 'hic073',  feature, downsampling_ratio=16, methods_dictionary=downstream_methods_dictionary)










