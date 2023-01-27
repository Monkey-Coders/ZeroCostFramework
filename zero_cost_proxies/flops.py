import torch
from torchprofile import profile_macs
import numpy as np

from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface 

class flops(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
        
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        dummy_data = torch.from_numpy(np.zeros(data.shape)).float().to(device)
        
        macs = profile_macs(model, dummy_data) // 2
        num_flops = int(macs*2)
        return num_flops