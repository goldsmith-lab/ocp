import torch

DEFAULT_METRICS = [
    "sum",
    "mean",
    # 'median',
    "var",
]


class ConvLayerProcessor:
    # def __init__(self):  # , metrics=DEFAULT_METRICS):
    # self.metrics = metrics
    # self.funcs = {
    #     "sum": torch.sum,
    #     "mean": torch.mean,
    #     # 'median': torch.median,
    #     "var": torch.var,
    # }

    def __call__(self, per_image_latent_features):
        result = []
        for latent in per_image_latent_features:
            r = latent.cpu().detach().to(torch.float16)
            # n = r.shape[0]
            mean = torch.mean(r, dim=0)
            # median = torch.median(r, dim=0)
            std = torch.std(r, dim=0)
            # fp_skewness = (n*(n-1))**0.5/(n-2) * torch.pow(r - mean, 3) / (n * torch.pow(std, 3))
            # p2_skewness = 3 * (mean - median) / std
            # kurtosis = torch.pow(r - mean, 4) / (n * torch.pow(std, 4))

            r = torch.stack(
                [mean, std]  # median, fp_skewness, p2_skewness, kurtosis]
                # [self.funcs[metric](r, dim=0) for metric in self.metrics]
            )
            result.append(r)
        result = torch.stack(result)
        return result

    def __bool__(self):
        return True  # len(self.metrics) > 0
