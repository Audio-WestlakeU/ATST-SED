import tqdm
import torch


class TorchScaler(torch.nn.Module):
    """
    This torch module implements scaling for input tensors, both instance based
    and dataset-wide statistic based.

    Args:
        statistic: str, (default='dataset'), represent how to compute the statistic for normalisation.
            Choice in {'dataset', 'instance'}.
             'dataset' needs to be 'fit()' with a dataloader of the dataset.
             'instance' apply the normalisation at an instance-level, so compute the statitics on the instance
             specified, it can be a clip or a batch.
        normtype: str, (default='standard') the type of normalisation to use.
            Choice in {'standard', 'mean', 'minmax'}. 'standard' applies a classic normalisation with mean and standard
            deviation. 'mean' substract the mean to the data. 'minmax' substract the minimum of the data and divide by
            the difference between max and min.
    """

    def __init__(self, statistic="dataset", normtype="standard", dims=(1, 2), eps=1e-8):
        super(TorchScaler, self).__init__()
        assert statistic in ["dataset", "instance", None]
        assert normtype in ["standard", "mean", "minmax", None]
        if statistic == "dataset" and normtype == "minmax":
            raise NotImplementedError(
                "statistic==dataset and normtype==minmax is not currently implemented."
            )
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(TorchScaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.statistic == "dataset":
            super(TorchScaler, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def fit(self, dataloader, transform_func=lambda x: x[0]):
        """
        Scaler fitting

        Args:
            dataloader (DataLoader): training data DataLoader
            transform_func (lambda function, optional): Transforms applied to the data.
                Defaults to lambdax:x[0].
        """
        indx = 0
        for batch in tqdm.tqdm(dataloader):

            feats = transform_func(batch)
            if indx == 0:
                mean = torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared = (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            else:
                mean += torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared += (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            indx += 1

        mean /= indx
        mean_squared /= indx

        self.register_buffer("mean", mean)
        self.register_buffer("mean_squared", mean_squared)

    def forward(self, tensor):

        if self.statistic is None or self.normtype is None:
            return tensor

        if self.statistic == "dataset":
            assert hasattr(self, "mean") and hasattr(
                self, "mean_squared"
            ), "TorchScaler should be fit before used if statistics=dataset"
            assert tensor.ndim == self.mean.ndim, "Pre-computed statistics "
            if self.normtype == "mean":
                return tensor - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (tensor - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        else:
            if self.normtype == "mean":
                return tensor - torch.mean(tensor, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (tensor - torch.mean(tensor, self.dims, keepdim=True)) / (
                    torch.std(tensor, self.dims, keepdim=True) + self.eps
                )
            elif self.normtype == "minmax":
                return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (
                    torch.amax(tensor, dim=self.dims, keepdim=True)
                    - torch.amin(tensor, dim=self.dims, keepdim=True)
                    + self.eps
                )
