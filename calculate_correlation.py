

# Main function
import json
import scipy
import os

base_path = "./ZeroCostFramework/"

def validate_architecture(architecture):
    return all(format in architecture for format in ["val_acc", "zero_cost_scores"])

zero_cost_proxies = ["synflow", "flops", "fisher", "snip", "grad_norm", "jacov", "l2_norm", "grasp", "epe_nas", "zen", "params"]


if __name__ == '__main__':
    with open(f'{base_path}generated_architectures.json') as f:
        architectures = json.load(f)

    try:
        with open('correlations.json') as f:
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
        print(proxies)
        print(val_accs)
        correlations[proxy] = scipy.stats.spearmanr(proxies, val_accs)[0]
        print(f"Proxy {proxy} has correlation {correlations[proxy]}")
        print(correlations)

    with open('correlations.json', 'w') as f:
        json.dump(correlations, f)
    



    