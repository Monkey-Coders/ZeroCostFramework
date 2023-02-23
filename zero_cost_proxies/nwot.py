import numpy as np
import torch
from ZeroCostFramework.utils.util_functions import get_layer_metric_array, initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class nwot(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        def counting_forward_hook(module, inp, out):
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        
        if len(data.size()) == 5:
            N, _C, _T, _V, M = data.size()
        else:
            N = len(labels)
            M = 1
        
        K_size = N * M
        model.K = np.zeros((K_size, K_size))
        
        for name, module in model.named_modules():
            module_type = str(type(module))
            if ('ReLU' in module_type) and ('naslib' not in module_type):
                module.register_forward_hook(counting_forward_hook)

        x = torch.clone(data)
        model(x)
        s, jc = np.linalg.slogdet(model.K)

        return jc