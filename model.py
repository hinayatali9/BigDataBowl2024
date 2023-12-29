import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PREPROCESS_STD = 6.8557
PREPROCESS_MEAN = -0.1061


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b, c)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(b, c, 1, 1)
        return x * out


class MaxAndAvgPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class MatchupAttention(nn.Module):
    def __init__(self, in_channels):
        super(MatchupAttention, self).__init__()
        self.pooled = MaxAndAvgPool()
        self.compress = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.matchups = nn.Linear(10, 10)
    def forward(self, x):
        pooled = self.pooled(x)
        x_compress = self.compress(pooled).squeeze(1)
        x_out = self.matchups(x_compress).unsqueeze(1)
        attention = F.sigmoid(x_out)

        return x * attention


# Define the PyTorch model
class ExpectedYardsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=10, out_channels=128, kernel_size=(1, 1))
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(1, 1))
        self.conv2d_3 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1, 1))
        self.conv2d_4 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1))

        self.conv1d_1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv1d_3 = nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1)

        self.dense_1 = nn.Linear(96, 96)
        self.dense_2 = nn.Linear(96, 256) #256
        self.dense_3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.2)

        self.batchnorm_conv2d = nn.BatchNorm2d(128)
        self.batchnorm_conv1d = nn.BatchNorm1d(128)
        self.batchnorm_dense = nn.BatchNorm1d(96)
        self.batchnorm_flat = nn.BatchNorm1d(256)

        # SE Blocks
        self.se_1 = SqueezeExcitation(128)
        self.se_2 = SqueezeExcitation(160)
        self.se_3 = SqueezeExcitation(128)

        # Matchup Attention Block
        self.ma_0 = MatchupAttention(10)
        self.ma_1 = MatchupAttention(128)
        self.ma_2 = MatchupAttention(160)
        self.ma_3 = MatchupAttention(128)

    def forward(self, x):
        # Conv2D and Activation layers
        x = F.relu(self.conv2d_1(x))
        x = self.se_1(x)
        x = self.ma_1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2d_2(x))
        x = self.se_2(x)
        x = self.ma_2(x)
        x = self.dropout(x)
        x = F.relu(self.conv2d_3(x))
        x = self.se_3(x)
        x = self.ma_3(x)
        x = self.dropout(x)
        
        # MaxPooling2D and AvgPool2D layers
        xmax = F.max_pool2d(x, (1, 10)) * 0.3
        xavg = F.avg_pool2d(x, (1, 10)) * 0.7
        x = xmax + xavg
        x = torch.squeeze(x, 2)
        #x = self.batchnorm_conv2d(x)

        # Reshape for Conv1D
        x = x.permute(0, 3, 1, 2).squeeze(1)

        # Conv1D and Activation layers

        x = F.relu(self.conv1d_1(x))
        #x = self.batchnorm_conv1d(x)
        x = F.relu(self.conv1d_2(x))
        #x = self.batchnorm_conv1d(x)
        x = F.relu(self.conv1d_3(x))
        #x = self.batchnorm_dense(x)

        # MaxPooling1D and AvgPool1D layers
        xmax = F.max_pool1d(x, 11) * 0.3
        xavg = F.avg_pool1d(x, 11) * 0.7
        x = xmax + xavg
        x = torch.squeeze(x, 2)

        # Dense and Activation layers
        x = F.relu(self.dense_1(x))
        #x = self.dropout(x)
        #x = self.batchnorm_dense(x)
        x = F.relu(self.dense_2(x))
        #x = self.dropout(x)
        #x = self.batchnorm_flat(x)
        x = self.dense_3(x)
        x = torch.squeeze(x, 1)

        return x


def load_expected_yards_model():
    model = ExpectedYardsModel()
    model.load_state_dict(torch.load('./model_e10_new.pt'))
    return model


def get_input_normalizations():
    return {'std': 6.8475, 'mean': -0.2011}