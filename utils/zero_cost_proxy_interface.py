class ZeroCostProxyInterface:
    def calculate_proxy(self, net, data_loader, hyperparameters, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
        pass