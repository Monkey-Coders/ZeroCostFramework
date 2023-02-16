import torch
import torch.nn as nn
import torch.nn.functional as F

import types
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_layer_metric_array, sum_arr, reshape_elements
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class fisher(ZeroCostProxyInterface):

    def fisher_forward_conv2d(self, f_self, x):
        x = F.conv2d(x, f_self.weight, f_self.bias, f_self.stride, f_self.padding, f_self.dilation, f_self.groups)

        f_self.act = f_self.dummy(x)
        return f_self.act

    def fisher_forward_linear(self, f_self, x):
        x = F.linear(x, f_self.weight, f_self.bias)

        f_self.act = f_self.dummy(x)
        return f_self.act

    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        mode = "channel"
        if mode == "param":
            raise ValueError("Fisher is not implemented for param mode")

        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.fisher = None
                layer.act = 0.
                layer.dummy = nn.Identity()

                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(self.fisher_forward_conv2d, layer)
                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(self.fisher_forward_linear, layer)

                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        act = layer.act.detach()
                        grad = grad_output[0].detach()
                        if len(act.shape) > 2:
                            g_nk = torch.sum((act * grad), list(range(2,len(act.shape))))
                        else:
                            g_nk = act * grad
                        del_k = g_nk.pow(2).mean(0).mul(0.5)
                        if layer.fisher is None:
                            layer.fisher = del_k
                        else:
                            layer.fisher += del_k
                        del layer.act #without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                    return hook
                #register backward hook on identity fcn to compute fisher info
                layer.dummy.register_backward_hook(hook_factory(layer))

        
        outputs = model(data)
        loss = loss_function(outputs, labels)
        loss.backward()

        def fisher(layer):
            if layer.fisher is not None:
                return torch.abs(layer.fisher.detach())
            else:
                return torch.zeros(layer.weight.shape[0])
        
        grads_abs_ch = get_layer_metric_array(model, fisher, mode)

        shapes = get_layer_metric_array(model, lambda x : x.weight.shape[1:], mode)
        grads_abs = reshape_elements(grads_abs_ch, shapes, device)
        score = sum_arr(grads_abs)
        return score
