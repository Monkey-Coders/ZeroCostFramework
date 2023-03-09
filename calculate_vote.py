import json
from itertools import combinations
import os

from utils.util_functions import get_proxies


path = "experiment"


def init(names):
    with open(f"{path}/generated_architectures.json") as file:
        results = json.load(file)
        
    metrics = {}
    acc = []
    
    for key in results:
        value = results[key]
        if "zero_cost_scores" not in value:
            continue
        for name in names:
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(float(value["zero_cost_scores"][name]["score"]))
        if "val_acc" in value:
            acc.append(float(value["val_acc"]))
        else:
            acc.append(0)
    return (acc, metrics)

def vote(mets, gt):
    numpos = 0
    for m in mets:
        numpos += 1 if m > 0 else 0
    if numpos >= len(mets)/2:
        sign = +1
    else:
        sign = -1
    return sign*gt


def calc(acc, metrics, comb):
    num_pts = len(acc)
    tot=0
    right=0
    for i in range(num_pts):
        for j in range(num_pts):
            if i!=j:
                diff = acc[i] - acc[j]
                if diff == 0:
                    continue
                diffsyn = []
                for m in comb:
                    diffsyn.append(metrics[m][i] - metrics[m][j])
                same_sign = vote(diffsyn, diff)
                right += 1 if same_sign > 0 else 0
                tot += 1
    votes = right/tot
    return (comb, votes)
    
def get_all_combinations(names):
    list_combinations = list()
    for n in range(2, len(names) + 1):
        list_combinations += list(combinations(names, n))
    return list_combinations

if __name__ == "__main__":
    print("STARTING...")
    proxies = get_proxies()
    print("INITIALIZING...")
    acc, metrics = init(proxies)
    comb = get_all_combinations(proxies)
    
    D = {}
    print("CALCULATING...")
    for c in comb:
        a, votes = calc(acc, metrics, c)
        D[str(a)] = votes
    print("SORTING...")
    D = dict(sorted(D.items(), key=lambda item: item[1], reverse=True))
    print("WRITING TO FILE...")
    with open(f"{path}/vote_combinations.json", "w") as file:
        json.dump(D, file)
