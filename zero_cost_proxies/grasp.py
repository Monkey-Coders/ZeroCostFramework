import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
import torch.autograd as autograd

from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class grasp(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        
        # get all applicable weights
        weights = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer.weight.requires_grad_(True)
                weights.append(layer.weight)
        #forward/grad pass #1
        outputs = model.forward(data)
        loss = loss_function(outputs, labels)
        grad_w_p = autograd.grad(loss, weights, allow_unused=True)
        grad_w = list(grad_w_p)
        
        # forward/grad pass #2 w
        outputs = model.forward(data)
        loss = loss_function(outputs, labels)
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        # accumulate gradients computed in previous step and call backwards
        z, count = 0,0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

        # compute final sensitivity metric and put in grads
        def grasp(layer):
            if layer.weight.grad is not None:
                return -layer.weight.data * layer.weight.grad   # -theta_q Hg
                #NOTE in the grasp code they take the *bottom* (1-p)% of values
                #but we take the *top* (1-p)%, therefore we remove the -ve sign
                #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.weight)
        
        score = get_score(model, grasp, "param")
        return score
