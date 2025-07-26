import torch
import torch.nn.functional as F

def frequency_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply a spatial filter to extract the low-frequency part of an image.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        kernel_size (int): Size of the filtering kernel (must be odd).

    Returns:
        torch.Tensor: Low-frequency filtered image.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if image.dim() != 4:
        raise ValueError("Input tensor must have shape (B, C, H, W).")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    padding = kernel_size // 2
    image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    channels = image.size(1)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size**2)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    return F.conv2d(image, kernel, groups=channels)


def resize_image(image, target_size):
    """
    Resizes an image to the specified target size (H, W), supporting both upsampling and downsampling.

    Args:
        image (torch.Tensor or array-like): Input image data. If not a tensor, it will be converted.
        target_size (tuple): Target size as (H, W).

    Returns:
        torch.Tensor: Resized image.
    """
    # Convert input to Tensor if necessary
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # Ensure input dimensions are either 3D or 4D
    if image.dim() not in {3, 4}:
        raise ValueError("Input must be 3D (C, H, W) or 4D (B, C, H, W).")

    # Add batch dimension for 3D input
    unsqueeze = image.dim() == 3
    if unsqueeze:
        image = image.unsqueeze(0)

    # Resize the image
    resized_image = F.interpolate(image, size=target_size, mode="bilinear" , align_corners=False)

    # Remove batch dimension if it was added
    return resized_image.squeeze(0) if unsqueeze else resized_image
