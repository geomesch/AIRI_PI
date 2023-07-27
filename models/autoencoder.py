import torch
import torch.nn as nn
import torch.nn.functional as F

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
        )

        self.fc = nn.Linear(256 * 64 * 64, 100)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 256, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            SwishActivation(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Преобразовать в плоский вектор
        x = self.fc(x)
        x = F.relu(x)  # Функция активации после полносвязного слоя
        x = x.view(x.size(0), 100, 1, 1)  # Преобразовать обратно в 4D тензор
        x = self.decoder(x)
        return x
