import torch.nn as nn
import torch.nn.functional as F
import torch


class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, latent_size):
        super(ConvAutoencoder, self).__init__()

        # Parameters for image size, channels, and latent space size
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.image_size = image_size
        self.latent_size = latent_size

        # Encoder layers
        self.encoder = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=2, padding=1),
            SwishActivation(),

            # Convolutional layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            SwishActivation(),

            # Convolutional layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            SwishActivation(),

            # Convolutional layer 4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            SwishActivation()
        )

        # Fully connected layer for latent space
        self.fc_latent = nn.Linear(128 * (image_size // 16) * (image_size // 16), latent_size)
        self.fc_decoder = nn.Linear(latent_size, 128 * (image_size // 16) * (image_size // 16))

        # Decoder layers
        self.decoder = nn.Sequential(
            # Transposed convolutional layer 1
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            SwishActivation(),

            # Transposed convolutional layer 2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            SwishActivation(),

            # Transposed convolutional layer 3
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            SwishActivation(),

            # Transposed convolutional layer 4
            nn.ConvTranspose2d(16, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
            SwishActivation(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_latent(x)
        x = self.fc_decoder(x)
        x = x.view(x.size(0), 128, self.image_size // 16, self.image_size // 16)  # Reshape back to 4D tensor
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    input_channels = 1  # Number of input channels (e.g., for RGB images)
    output_channels = 20  # Number of output channels
    image_size = 64  # Size of the input image (image_size x image_size)
    latent_size = 100  # Size of the latent space
    model = ConvAutoencoder(
        in_channels=input_channels,
        out_channels=output_channels,
        image_size=image_size,
        latent_size=latent_size
    )

    # Checking the model architecture
    print(model)
    input_image = torch.randn(4, 1, 64, 64)  # Пример батча из 4 изображений
    output_segmentation_map = model(input_image)
    print(output_segmentation_map.shape)
