import numpy as np
import torch
from ZeroCostFramework.utils.util_functions import get_layer_metric_array, initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class nwot(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        batch_size = len(labels)
   
        def counting_forward_hook(module, inp, out):
            inp = inp[0].view(inp[0].size(0), -1)
            x = (inp > 0).float() # binary indicator 
            K = x @ x.t() 
            K2 = (1.-x) @ (1.-x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy() # hamming distance 
                    
        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        model.K = np.zeros((batch_size, batch_size))
        for name, module in model.named_modules():
            module_type = str(type(module))
            if ('ReLU' in module_type)  and ('naslib' not in module_type):
                # module.register_full_backward_hook(counting_backward_hook)
                module.register_forward_hook(counting_forward_hook)

        x = torch.clone(data)
        model(x)
        s, jc = np.linalg.slogdet(model.K)

        return jc