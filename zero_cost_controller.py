import os
from tqdm import tqdm
from ZeroCostFramework.utils.util_functions import calculate_function_runtime

def files_filter(f):
    if f == '__init__.py':
        return False
    if f.endswith('.py'):
        return True
    return False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def calculate_zc_proxy_scores(net, data_loader, device, loss_function, save_path):
    proxies = [f.replace(".py", "") for f in os.listdir(f"{os.path.dirname(__file__)}/zero_cost_proxies") if files_filter(f)]
    folder_name = os.path.dirname(__file__).split('/')[-1]

    scores = {}

    for proxy_name in tqdm(proxies):
        name = f"{folder_name}.{proxy_name}"
        proxy = import_class(name)
        score, elapsed_time = calculate_function_runtime(proxy().calculate_proxy, net, data_loader, device, loss_function)    
        scores[proxy_name] = {
            "score": str(score),
            "time": str(elapsed_time)
            }
        
    
    return scores