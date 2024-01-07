import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PREPROCESS_STD = 6.8557
PREPROCESS_MEAN = -0.1061


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        # Initial pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected (dense) layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        # Get the batch size and channels
        b, c, _, _ = x.size()

        # View reshapes the tensor to (b, c)
        pool_outputs = self.pool(x).view(b, c)

        # Pass through the fully connected layers
        dense_1_outputs = F.relu(self.fc1(pool_outputs))
        dense_2_outputs = torch.sigmoid(self.fc2(dense_1_outputs))

        # Reshape to original tensor shape, so that multuplication is possible
        dense_2_outputs = dense_2_outputs.view(b, c, 1, 1)
        return x * dense_2_outputs


# Perform Max and Average Pooling as described by Hu et al.
class MaxAndAvgPool(nn.Module):
    def forward(self, x):
        # Calculate the max and mean along the channel dimmension
        max_tensor = torch.max(x,1)[0].unsqueeze(1)
        mean_tensor = torch.mean(x,1).unsqueeze(1)
        return torch.cat((max_tensor, mean_tensor), dim=1)


class MatchupAttention(nn.Module):
    def __init__(self, in_channels):
        super(MatchupAttention, self).__init__()
        # Perfom pooling
        self.pooled = MaxAndAvgPool()
        # Convolve across the max and mean pooled channels
        self.compress = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        # Matchup attention
        self.matchups = nn.Linear(10, 10)

    def forward(self, x):
        pooled = self.pooled(x)
        x_compress = self.compress(pooled).squeeze(1)
        x_focused = self.matchups(x_compress).unsqueeze(1)
        attention = F.sigmoid(x_focused)

        # Return both the output and the attention (for interpretability)
        return x * attention, attention


# Define the PyTorch model
class ExpectedYardsModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Set the return_attention flag to True to return the attention maps
        self.return_attention = False

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
        # Conv2D, SE, and Matchup attention layers
        x = F.relu(self.conv2d_1(x))
        x = self.se_1(x)
        x, a1 = self.ma_1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2d_2(x))
        x = self.se_2(x)
        x, a2 = self.ma_2(x)
        x = self.dropout(x)
        x = F.relu(self.conv2d_3(x))
        x = self.se_3(x)
        x, a3 = self.ma_3(x)
        x = self.dropout(x)

        # MaxPooling2D and AvgPool2D layers
        xmax = F.max_pool2d(x, (1, 10)) * 0.3
        xavg = F.avg_pool2d(x, (1, 10)) * 0.7
        x = xmax + xavg
        x = torch.squeeze(x, 2)

        # Reshape for Conv1D
        x = x.permute(0, 3, 1, 2).squeeze(1)

        # Conv1D layers
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        x = F.relu(self.conv1d_3(x))

        # MaxPooling1D and AvgPool1D layers
        xmax = F.max_pool1d(x, 11) * 0.3
        xavg = F.avg_pool1d(x, 11) * 0.7
        x = xmax + xavg
        x = torch.squeeze(x, 2)

        # Dense and Activation layers
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        x = torch.squeeze(x, 1)

        if self.return_attention:
            return x, a1, a2, a3
        
        return x


def load_expected_yards_model():
    model = ExpectedYardsModel()
    model.load_state_dict(torch.load('./expected_yards_model.pt'))
    model.eval()
    return model
