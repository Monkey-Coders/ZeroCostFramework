import torch
import torch.nn as nn
import torch.nn.functional as F
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score

import types

from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class snip(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)

        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(self.snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(self.snip_forward_linear, layer)

        N = data.shape[0]
        
        outputs, _ = model.forward(data)
        loss = loss_function(outputs, labels)
        loss.backward()

        def snip(layer):
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight_mask.grad)
            else:
                return torch.zeros_like(layer.weight)
        
        score = get_score(model, snip, "param")

        return score
    
    def snip_forward_conv2d(self, x):
            return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    def snip_forward_linear(self, x):
            return F.linear(x, self.weight * self.weight_mask, self.bias)
