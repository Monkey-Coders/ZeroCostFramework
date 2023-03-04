import torch
from ZeroCostFramework.utils.util_functions import get_score, initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class plain(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
     
        output = model(data)
        loss = loss_function(output, labels)
        loss.backward()

        # select the gradients that we want to use for search/prune
        def plain(layer):
            if layer.weight.grad is not None:
                return layer.weight.grad * layer.weight
            else:
                return torch.zeros_like(layer.weight)
            
        score = get_score(model, plain, "param")
        return score