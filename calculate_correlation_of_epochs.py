import json
from utils.util_functions import get_proxies
import scipy
import matplotlib.pyplot as plt
base_path = "experiment"

epochs = ["zero_cost_scores"]

for i in range(10):
    epochs.append(f"zero_cost_scores_{i}")

def validate_architecture(architecture, epoch):
    return all(format in architecture for format in ["val_acc", epoch])


if __name__ == '__main__':
    zero_cost_proxies = get_proxies()

    with open(f'{base_path}/generated_architectures.json') as f:
        architectures = json.load(f)

    try:
        with open(f'{base_path}/correlations.json') as f:
            correlations = json.load(f)
    except:
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
            
    with open(f'{base_path}/correlations.json', 'w') as f:
        json.dump(correlations, f)

    for proxy in zero_cost_proxies:
        x = []
        y = []
        for epoch in epochs:
            try:
                epoch_number = int(epoch.split("_")[-1])
            except:
                epoch_number = -1
            x.append(epoch_number)
            y.append(correlations[epoch][proxy])
        plt.plot(x, y, "-D", label=proxy, markevery=[0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.tight_layout()
    # Save
    plt.savefig(f'{base_path}/correlations.png')
