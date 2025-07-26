import torch
import torch.nn as nn

padding_mode = 'reflect'

class Generator(nn.Module):
    def __init__(self, z_dim:int, input_channel:int=8, output_channel:int=8, conditional:bool=True):
        """
        Generator Model

        Args:
            z_dim (int): Number of channels in the noise or latent vector
            input_channel (int): Number of channels in the conditional input (e.g., low-resolution image)
            output_channel (int): Number of channels in the final output image
            conditional (bool): Whether to use conditional generation
        """
        super(Generator, self).__init__()
        self.conditional = conditional
        layers = []

        if self.conditional:
            layers.append(nn.Conv2d(z_dim+input_channel, 64, 11, 1, 5, bias=False, padding_mode=padding_mode))  # Conditional case
        else:
            layers.append(nn.Conv2d(z_dim, 64, 11, 1, 5, bias=False, padding_mode=padding_mode))  # Non-conditional case
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(64, 128, 9, 1, 4, bias=False, padding_mode=padding_mode))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(128, 64, 7, 1, 3, bias=False, padding_mode=padding_mode))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(64, 32, 5, 1, 2, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(32, output_channel, 3, 1, 1, bias=False))

        self.main = nn.Sequential(*layers)

    def forward(self, z, c):
        x = torch.concat([z,c],axis=1) if self.conditional else z
        x = self.main(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_channel:int=8, output_channel:int=8, conditional:bool=True):
        """
        Critic Model (Discriminator)

        Args:
            input_channel (int): Number of channels in the input image (e.g., generated image)
            output_channel (int): Number of channels in the conditional input (e.g., ground truth or low-frequency image)
            conditional (bool): Whether to use conditional discrimination
        """

        super(Critic, self).__init__()
        self.conditional = conditional
        layers = []

        if self.conditional:
            layers.append(nn.Conv2d(input_channel + output_channel, 64, 3, 1, 2, bias=False, padding_mode=padding_mode))  # Conditional case
        else:
            layers.append(nn.Conv2d(input_channel, 64, 3, 1, 2, bias=False, padding_mode=padding_mode))  # Non-conditional case
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.2, inplace=True)) 

        layers.append(nn.Conv2d(64, 128, 3, 1, 0, bias=False, padding_mode=padding_mode))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(128, 256, 3, 2, 0, bias=False, padding_mode=padding_mode))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(256, 128, 3, 2, 0, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(128, 64, 3, 2, 0, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(64, 1, 3, 2, 0, bias=False))

        self.main = nn.Sequential(*layers)
        
    def forward(self, input, c):
        x = torch.concat([input,c],axis=1) if self.conditional else input
        x = self.main(x)
        return x