import torch

class ZeroCostProxyInterface:
    def calculate_proxy(
        self,
        net: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str,
        loss_function: torch.nn.L1Loss
        | torch.nn.MSELoss
        | torch.nn.CrossEntropyLoss
        | torch.nn.CTCLoss
        | torch.nn.NLLLoss
        | torch.nn.PoissonNLLLoss
        | torch.nn.GaussianNLLLoss
        | torch.nn.KLDivLoss
        | torch.nn.BCELoss
        | torch.nn.BCEWithLogitsLoss
        | torch.nn.MarginRankingLoss
        | torch.nn.HingeEmbeddingLoss
        | torch.nn.MultiLabelMarginLoss
        | torch.nn.HuberLoss
        | torch.nn.SmoothL1Loss
        | torch.nn.SoftMarginLoss
        | torch.nn.MultiLabelSoftMarginLoss
        | torch.nn.CosineEmbeddingLoss
        | torch.nn.MultiMarginLoss
        | torch.nn.TripletMarginLoss
        | torch.nn.TripletMarginWithDistanceLoss,
        eval: bool = False,
        train: bool = True,
        single_batch: bool = True,
        bn: bool = False,
    ) -> float:
        pass
