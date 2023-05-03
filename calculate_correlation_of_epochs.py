import json
from utils.util_functions import get_proxies
import scipy
import matplotlib.pyplot as plt
base_path = "experiment"
import math
import os

epochs = ["zero_cost_scores"]
for i in range(10):
    epochs.append(f"zero_cost_scores_{i}")
for i in range(11, 46, 2):
    epochs.append(f"zero_cost_scores_{i}")

def validate_architecture(architecture, epoch):
    return all(format in architecture for format in ["val_acc", epoch])

if __name__ == '__main__':
    zero_cost_proxies = get_proxies()

    with open(f'{base_path}/generated_architectures_test.json') as f:
        architectures = json.load(f)

    if not os.path.exists(f'{base_path}/correlations'):
        os.mkdir(f'{base_path}/correlations')
    try:
        with open(f'{base_path}/correlations/correlations.json') as f:
            correlations = json.load(f)
    except:
        # make directory f'{base_path}/correlations'
        correlations = {}
    
    for epoch in epochs:
        if epoch not in correlations:
            correlations[epoch] = {}

        for proxy in zero_cost_proxies:
            proxies = []
            val_accs = []
            for architecture in architectures:
                if validate_architecture(architectures[architecture], epoch):
                    if proxy in architectures[architecture][epoch]:
                        proxies.append(architectures[architecture][epoch][proxy]["score"])
                        val_accs.append(architectures[architecture]["val_acc"])
            correlations[epoch][proxy] = scipy.stats.spearmanr(proxies, val_accs, nan_policy='omit')[0]
            
    with open(f'{base_path}/correlations/correlations.json', 'w') as f:
        json.dump(correlations, f)
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    for i, proxy in enumerate(zero_cost_proxies):
        x = []
        y = []
        for epoch in epochs:
            try:
                epoch_number = int(epoch.split("_")[-1])
            except:
                epoch_number = -1
            x.append(epoch_number)
            y.append(correlations[epoch][proxy])
        plt.plot(x, y, label=proxy, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)], color=colors[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.tight_layout()
    # Save
    plt.savefig(f'{base_path}/correlations/correlations.png')
    num_proxies = len(zero_cost_proxies)
    
    num_columns = 2
    num_rows = math.ceil(num_proxies / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, 5 * num_rows), constrained_layout=True)

    for idx, proxy in enumerate(zero_cost_proxies):
        row = idx // num_columns
        col = idx % num_columns
        ax = axes[row, col]

        x = []
        y = []
        for epoch in epochs:
            try:
                epoch_number = int(epoch.split("_")[-1])
            except:
                epoch_number = -1
            x.append(epoch_number)
            y.append(correlations[epoch][proxy])

        ax.plot(x, y, "-D", label=proxy, markevery=[0])
        ax.legend(loc='upper right')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spearman Correlation")
        ax.set_title(proxy)

    if num_proxies % num_columns != 0:
        for i in range(num_proxies, num_rows * num_columns):
            fig.delaxes(axes.flatten()[i])

    plt.savefig(f'{base_path}/correlations/correlations_grid.png')  
