import os
from tqdm import tqdm
from ZeroCostFramework.utils.util_functions import get_proxies, calculate_function_runtime
import torch
import gc

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def calculate_zc_proxy_scores(net, data_loader, device, loss_function, save_path):
    proxies = get_proxies()
    folder_name = os.path.dirname(__file__).split('/')[-1]

    scores = {}
    overide = []

    for proxy_name in tqdm(proxies):
        if len(overide):
            if proxy_name not in overide:
                continue
        name = f"{folder_name}.{proxy_name}"
        proxy = import_class(name)
        gc.collect()
        torch.cuda.empty_cache()
        net = net.cpu()
        torch.cuda.empty_cache()
        net = net.to(device)
        try:
            score, elapsed_time = calculate_function_runtime(proxy().calculate_proxy, net, data_loader, device, loss_function)    
            scores[proxy_name] = {
                "score": str(score),
                "time": str(elapsed_time)
                }
        except Exception as e:
            print(f"Error: {e}")


        
    
    return scores