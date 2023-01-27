import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class grad_norm(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        output, _ = model(data)
        loss = loss_function(output, labels)
        loss.backward()
        score = get_score(model, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
        return score