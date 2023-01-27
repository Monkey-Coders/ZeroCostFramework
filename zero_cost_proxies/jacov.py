import torch
import numpy as np
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface


class jacov(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        try:
            jacobs, labels = self.get_batch_jacobian(model, data, labels)
            jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
            jc = self.eval_score(jacobs, labels)
        except Exception as e:
            print(e)
            jc = np.nan
        return jc
    
    def get_batch_jacobian(net, x, target):
        net.zero_grad()
        x.requires_grad = True
        out, features = net(x)
        out.backward(torch.ones_like(out))
        jacob = x.grad.detach()
        return jacob, target.detach()


    def eval_score(jacob, labels=None):
        corrs = np.corrcoef(jacob)
        v, _ = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1.0 / (v + k))
