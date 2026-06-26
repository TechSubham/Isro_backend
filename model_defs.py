import torch
import torch.nn as nn


class GBlock(nn.Module):
    def __init__(self, L, nc, u=256, nl=4):
        super().__init__()
        layers = [nn.Linear(L, u), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Linear(u, u), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.back = nn.Linear(u, L)
        self.head = nn.Linear(u, nc)

    def forward(self, x):
        h = self.fc(x)
        return self.back(h), self.head(h)


class NBeatsCls(nn.Module):
    def __init__(self, L=60, nc=2, u=256, blocks=9):
        super().__init__()
        self.bl = nn.ModuleList([GBlock(L, nc, u) for _ in range(blocks)])

    def forward(self, x):
        res = x
        logits = 0.0
        for block in self.bl:
            backcast, out = block(res)
            res = res - backcast
            logits = logits + out
        return logits


class FcstCNN(nn.Module):
    def __init__(self, ci=3, no=3):
        super().__init__()

        def block(i, o, k, d=1):
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=(k // 2) * d, dilation=d),
                nn.BatchNorm1d(o),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            block(ci, 32, 5),
            block(32, 64, 3),
            block(64, 64, 3, 2),
            block(64, 128, 3, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, no),
        )

    def forward(self, x):
        return self.head(self.net(x))