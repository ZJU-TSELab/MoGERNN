import torch
from torch import nn
from torch.nn import functional as F


def mape_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))


def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true))


class PinBallLoss(nn.Module):

    def __init__(self, quantile=0.5):
        self.quantile = quantile
        super(PinBallLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(
            torch.max(self.quantile * (y_true - y_pred),
                      (1 - self.quantile) * (y_pred - y_true)))


class MultiPinBallLoss(nn.Module):

    def __init__(self, quantiles=[0.1, 0.9, 0.5]):
        self.quantiles = quantiles
        super(MultiPinBallLoss, self).__init__()

    def forward(self, y_preds, y_trues, y_omi=None):
        if y_omi == None:
            y_omi = torch.zeros_like(y_trues, dtype=torch.bool)
        pinball = 0
        for i in range(len(self.quantiles)):
            pinball += torch.mean(
                torch.max(
                    self.quantiles[i] *
                    (y_trues[~y_omi] - y_preds[..., i:i + 1][~y_omi]),
                    (1 - self.quantiles[i]) *
                    (y_preds[..., i:i + 1][~y_omi] - y_trues[~y_omi])))
        quantiles = torch.tensor(self.quantiles).to(y_preds.device)
        #给数据重新排序 0.1,0.5,0.9，计算单调性损失
        y_preds = y_preds[..., torch.argsort(quantiles)]
        monotonicity = monotonicity_regularization(y_preds)
        return pinball + monotonicity
        # return pinball


def monotonicity_regularization(predictions):
    # Compute the differences between adjacent quantiles
    diffs = predictions[..., 1:] - predictions[..., :-1]
    # Penalize negative differences
    return torch.mean(torch.relu(-diffs))


class MSEPinballLoss(nn.Module):

    def __init__(self, quantiles=[0.1, 0.9]):
        self.quantiles = quantiles
        super(MSEPinballLoss, self).__init__()

    def forward(self, y_preds, y_trues, y_omi=None):
        if y_omi == None:
            y_omi = torch.zeros_like(y_trues, dtype=torch.bool)
        pinball = 0
        for i in range(len(self.quantiles)):
            pinball += torch.mean(
                torch.max(
                    self.quantiles[i] *
                    (y_trues[~y_omi] - y_preds[..., i:i + 1][~y_omi]),
                    (1 - self.quantiles[i]) *
                    (y_preds[..., i:i + 1][~y_omi] - y_trues[~y_omi])))
        mse = torch.nn.functional.mse_loss(y_preds[..., i + 1:][~y_omi],
                                           y_trues[~y_omi])
        return pinball + mse


def variance(y1, y2):
    mean = (y1 + y2) / 2
    return (y1 - mean)**2 + (y2 - mean)**2


def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean(
        (1 - label) * torch.pow(euclidean_distance, 2) + label *
        torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss


class SlitPhyLoss():

    def __init__(self, features, criterias=['mape', 'mae', 'rmse']):
        target = {
            'q': [0],
            'o': [1],
            'v': [2],
            'qv': [0, 2],
            'qov': [0, 1, 2],
        }
        names = ['flow (veh/h/lane)', 'occupancy', 'speed (km/h)']
        self.features = [names[ind] for ind in target[features]]
        self.criterias = criterias
        self.sub_metrics = [
            f'{feature}/{criteria}' for feature in self.features
            for criteria in criterias
        ]

    def __call__(self, y_pred, y_true, miss_indicator):
        """
        miss_indicator: 1 for missing, 0 for not missing
        """
        assert torch.min(
            y_true[~miss_indicator]
        ) >= 0, 'there should have no negative value in y_true'
        loss = {}
        for i, feature in enumerate(self.features):
            feature_y_pred = y_pred[..., i][~miss_indicator[..., i]]
            feature_y_true = y_true[..., i][~miss_indicator[..., i]]
            for criteria in self.criterias:
                if criteria == 'mape':
                    metric = mape_loss(feature_y_pred, feature_y_true).item()
                elif criteria == 'rmse':
                    metric = rmse_loss(feature_y_pred, feature_y_true).item()
                elif criteria == 'mse':
                    metric = torch.nn.functional.mse_loss(
                        feature_y_pred, feature_y_true).item()
                elif criteria == 'mae':
                    metric = mae_loss(feature_y_pred, feature_y_true).item()
                else:
                    raise ValueError(f'criteria {criteria} not supported')
                loss.update({f'{feature}/{criteria}': metric})
        return loss


def kde(data, sample, data_dim=1):
    data = data.reshape(-1)
    sample = sample.reshape(-1)
    n = data.shape[0]
    bandwidth = n**(-1 / (data_dim + 4))

    diff = sample[:, None] - data[None, :]
    bandwidth = torch.tensor(bandwidth, dtype=torch.float32)
    const = torch.sqrt(torch.tensor(2 * torch.pi, dtype=torch.float32))
    kernels = torch.exp(-0.5 * (diff / bandwidth)**2) / (bandwidth * const)
    return torch.sum(kernels, dim=1) / n


def kl_divergence(p, q):
    epsilon = 1e-10
    epsilon = torch.tensor(epsilon, dtype=torch.float32)
    return torch.sum(p * torch.log((p + epsilon) / (q + epsilon)))


# 自定义损失函数，支持掩码
def masked_gaussian_nll_loss(y_hat, y, mask):
    mask = ~mask
    sigma, mu = y_hat[..., 0:1], y_hat[..., 1:]
    criterion = nn.GaussianNLLLoss()
    loss = criterion(mu[mask], y[mask], (sigma**2)[mask])
    return loss
