import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=20, input_size=64, latent_size=100):
        super(Autoencoder, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.fc = nn.Linear(256 * input_size * input_size, latent_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Преобразовать в плоский вектор
        x = self.fc(x)
        x = F.relu(x)  # Функция активации после полносвязного слоя
        x = x.view(x.size(0), self.latent_size, 1, 1)  # Преобразовать обратно в 4D тензор
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    input_size = 64
    latent_size = 100
    model = Autoencoder(
        in_channels=1, out_channels=20,
        input_size=input_size, latent_size=latent_size
    )
    print(model)
