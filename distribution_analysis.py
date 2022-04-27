from src.evaluation import distribution_analysis
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d



def interpolate(y):
    return gaussian_filter1d(y, sigma=4)


base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_rao_et_al_replicate/',
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_encode0/',
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_encode1/',
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_encode2/',
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_hic026/',
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_hic033/'
]

target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_rao_et_al/'





fig = plt.figure()
fig.set_size_inches(18, 14)
ax = plt.axes()
x = range(200)

for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)


base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/GM12878_synthetic16/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/GM12878_synthetic25/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/GM12878_synthetic50/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/GM12878_synthetic100/',
]
target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/GM12878_rao_et_al/'


for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)



plt.ylim([0, 0.85])
plt.ylabel("PCC", size=16)
plt.xlabel("Genomic Distance (10KB)", size=16)

plt.legend()
plt.savefig('gm12878_dist_analysis.png')












base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/IMR90_hic057/',
]

target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/IMR90_rao_et_al/'

fig = plt.figure()
fig.set_size_inches(18, 14)
ax = plt.axes()
x = range(200)

for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)


base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/IMR90_synthetic16/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/IMR90_synthetic25/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/IMR90_synthetic50/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/IMR90_synthetic100/',
]
target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/IMR90_rao_et_al/'

for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)


plt.ylim([0, 0.85])
plt.ylabel("PCC", size=16)
plt.xlabel("Genomic Distance (10KB)", size=16)

plt.legend()
plt.savefig('imr90_dist_analysis.png')


base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/K562_hic073/',
]

target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/K562_rao_et_al/'

fig = plt.figure()
fig.set_size_inches(18, 14)
ax = plt.axes()
x = range(200)

for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)


base_chromosome_file_paths = [
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/K562_synthetic16/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/K562_synthetic25/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/K562_synthetic50/',
    '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/K562_synthetic100/',
]
target_chromosome_file_path = '/media/murtaza/ubuntu/updated_hic_data/data/synthetic_chromosome_files/K562_rao_et_al/'

for base_chromosome_file_path in base_chromosome_file_paths:
    correlations = distribution_analysis.compute_distribution_analysis_on_experiment_directory(base_chromosome_file_path, target_chromosome_file_path, 'distribution_analysis', -1, -1)
    label = base_chromosome_file_path.split('/')[-2].split('_')[-1]
    print(label)

    ax.plot(x, interpolate(correlations), label=label, linewidth=5)


plt.ylim([0, 0.85])
plt.ylabel("PCC", size=16)
plt.xlabel("Genomic Distance (10KB)", size=16)

plt.legend()
plt.savefig('k562_dist_analysis.png')













































































