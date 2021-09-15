import torch

DEFAULT_METRICS = [
    "sum",
    "mean",
    # 'median',
    "std",
]


class LatentProcessor:
    def __init__(self, metrics=DEFAULT_METRICS):
        self.metrics = metrics
        self.funcs = {
            "sum": torch.sum,
            "mean": torch.mean,
            # 'median': torch.median,
            "std": torch.std,
        }

    def __call__(self, per_image_latent_features):
        result = []
        for latent in per_image_latent_features:
            r = latent.cpu().detach().to(torch.float16)
            r = torch.stack(
                [self.funcs[metric](r, dim=0) for metric in self.metrics]
            )
            result.append(r)
        result = torch.stack(result)
        return result

    def __bool__(self):
        return len(self.metrics) > 0
