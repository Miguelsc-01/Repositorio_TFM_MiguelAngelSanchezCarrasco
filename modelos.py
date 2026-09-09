# Arquitecturas (CNNs, recurrentes, clásicos) y registro de modelos.
from __future__ import annotations
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DroneCNN(nn.Module):
    # CNN compacta

    def __init__(self, n_features: int = 40, n_frames: int = 32,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            nn.MaxPool2d(2),    # F/2, T/2
            _conv_block(16, 32),
            nn.MaxPool2d(2),    # F/4, T/4
            _conv_block(32, 64),
            nn.AdaptiveAvgPool2d((1, 1)),   # global average pooling -> (64,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DroneCRNN(nn.Module):
    def __init__(self, n_features: int = 40, n_frames: int = 32,
                 n_classes: int = 2, rnn_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        # Pooling solo en frecuencia (kernel (2,1)) para conservar la resolución temporal que utilizará el LSTM.
        self.cnn = nn.Sequential(
            _conv_block(1, 16),
            nn.MaxPool2d((2, 1)),   # F -> F/2
            _conv_block(16, 32),
            nn.MaxPool2d((2, 1)),   # F -> F/4
        )
        feat_per_step = 32 * (n_features // 4)  # canales * frecuencia reducida
        self.lstm = nn.LSTM(
            input_size=feat_per_step, hidden_size=rnn_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * 2, n_classes),
        )

    def forward(self, x):
        z = self.cnn(x) # (B, C, F', T)
        b, c, fp, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * fp) # (B, T, C*F')
        out, _ = self.lstm(z)   # (B, T, 2H)
        z = out.mean(dim=1) # pooling temporal medio
        return self.classifier(z)


# CNN grande
def _cbr(i, o, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class DroneCNNBig(nn.Module):
    def __init__(self, n_features=128, n_frames=61, n_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _cbr(1, 32), _cbr(32, 32), nn.MaxPool2d(2),
            _cbr(32, 64), _cbr(64, 64), nn.MaxPool2d(2),
            _cbr(64, 128), _cbr(128, 128), nn.MaxPool2d(2),
            _cbr(128, 256), _cbr(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# ResNet
class _BasicBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or i != o:
            self.down = nn.Sequential(
                nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = self.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.relu(y + idt)


class DroneResNet(nn.Module):
    def __init__(self, n_features=128, n_frames=61, n_classes=2,
                 channels=(32, 64, 128, 256), blocks=2, dropout=0.2):
        super().__init__()
        self.stem = _cbr(1, channels[0])
        layers, c_in = [], channels[0]
        for ci in channels:
            for b in range(blocks):
                stride = 2 if (b == 0 and ci != channels[0]) else 1
                layers.append(_BasicBlock(c_in, ci, stride))
                c_in = ci
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.body(self.stem(x))))


# MobileNetV2
class _InvertedResidual(nn.Module):
    def __init__(self, i, o, stride, expand=6):
        super().__init__()
        h = i * expand
        self.use_res = (stride == 1 and i == o)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(i, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(h, h, 3, stride, 1, groups=h, bias=False),    # depthwise
            nn.BatchNorm2d(h), nn.ReLU6(inplace=True),
            nn.Conv2d(h, o, 1, bias=False), nn.BatchNorm2d(o)]  # proyección lineal
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class DroneMobileNet(nn.Module):
    def __init__(self, n_features=128, n_frames=61, n_classes=2, dropout=0.2):
        super().__init__()
        # (expand, out, repeticiones, stride) para versión compacta para audio
        cfgs = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
                (6, 64, 3, 2), (6, 96, 2, 1), (6, 160, 2, 2)]
        layers = [nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True)]
        c_in = 32
        for t, c, n, s in cfgs:
            for k in range(n):
                layers.append(_InvertedResidual(c_in, c, s if k == 0 else 1, t))
                c_in = c
        layers += [nn.Conv2d(c_in, 320, 1, bias=False), nn.BatchNorm2d(320), nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(320, n_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class DroneRNN(nn.Module):
    def __init__(self, n_features=128, n_frames=63, n_classes=2,
                 hidden=128, layers=2, dropout=0.3, cell="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        # (B, 1, F, T) -> (B, T, F)
        x = x.squeeze(1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.classifier(out.mean(dim=1))


def build_classifiers(seed: int = 42) -> dict:
    # Devuelve un diccionario
    return {
        "svm": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1,
        ),
    }


def build_model(name: str, meta: dict):
    # Instancia un modelo dado su nombre y la metadata del dataset
    name = name.lower()
    nf, nt = meta["n_features"], meta["n_frames"]

    if name == "cnn":
        return DroneCNN(n_features=nf, n_frames=nt, n_classes=2)
    if name == "crnn":
        return DroneCRNN(n_features=nf, n_frames=nt, n_classes=2)
    if name == "cnn_big":
        return DroneCNNBig(n_features=nf, n_frames=nt, n_classes=2)
    if name == "resnet":
        return DroneResNet(n_features=nf, n_frames=nt, n_classes=2)
    if name == "mobilenet":
        return DroneMobileNet(n_features=nf, n_frames=nt, n_classes=2)
    if name == "lstm":
        return DroneRNN(n_features=nf, n_frames=nt, n_classes=2, cell="lstm")
    if name == "gru":
        return DroneRNN(n_features=nf, n_frames=nt, n_classes=2, cell="gru")

    raise ValueError(
        f"Modelo desconocido: {name!r}. Disponibles: {AVAILABLE_TORCH_MODELS}")


AVAILABLE_TORCH_MODELS = ["cnn", "crnn", "cnn_big", "resnet", "mobilenet", "lstm", "gru"]
