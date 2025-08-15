import torch.nn as nn


def conv_block(c_in: int, c_out: int):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class MyNet(nn.Module):
    def __init__(self, n_classes=10, base_channels=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, base_channels),
            conv_block(base_channels, base_channels),
            conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            conv_block(base_channels * 2, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            conv_block(base_channels * 4, base_channels * 4),
            conv_block(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.net(X)
