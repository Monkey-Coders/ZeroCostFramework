

# Main function
import json
from ZeroCostFramework.utils.util_functions import get_proxies
import scipy

base_path = "architectures"

def validate_architecture(architecture):
    return all(format in architecture for format in ["val_acc", "zero_cost_scores"])


if __name__ == '__main__':
    zero_cost_proxies = get_proxies(["plain", "nwot"])
    

    with open(f'{base_path}/generated_architectures.json') as f:
        architectures = json.load(f)

    try:
        with open(f'{base_path}/correlations.json') as f:
            correlations = json.load(f)
    except:
        correlations = {}

    for proxy in zero_cost_proxies:
        proxies= []
        val_accs = []
        for architecture in architectures:
            if validate_architecture(architectures[architecture]):
                proxies.append(architectures[architecture]["zero_cost_scores"][proxy]["score"])
                val_accs.append(architectures[architecture]["val_acc"])
        correlations[proxy] = scipy.stats.spearmanr(proxies, val_accs)[0]

    with open(f'{base_path}/correlations.json', 'w') as f:
        json.dump(correlations, f)