import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the time step 't' into a vector for the network."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic Residual Block with GroupNorm and SiLU activation."""

    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # Downsample

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        # First convolution
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding injection
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend dimensions
        h = h + time_emb
        # Second convolution
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Downsample or Upsample
        return self.transform(h)


class DiffPlateUNet(nn.Module):
    """
    The Conditional U-Net for Super Resolution.
    Input: Concatenation of Noisy Image (3 ch) + Upscaled LR Image (3 ch) = 6 channels
    """

    def __init__(self, image_height=64, image_width=192):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(image_channels * 2, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )

        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, t, lr_condition):
        # 1. Condition: Concatenate Noisy High-Res (x) and Upscaled Low-Res (lr_condition)
        x = torch.cat((x, lr_condition), dim=1)

        # 2. Embed Time
        t = self.time_mlp(t)

        # 3. Initial Conv
        x = self.conv0(x)

        # 4. U-Net Path
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        for up in self.ups:
            residual = residuals.pop()
            # Resize residual if dimensions don't match (simple fix for odd dims)
            if x.shape != residual.shape:
                x = F.interpolate(x, size=residual.shape[2:], mode="nearest")
            # Concat for skip connection
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)


class DiffPlate(nn.Module):
    """
    Main Diffusion Wrapper.
    Manages the noise schedule, training loss, and sampling.
    """

    def __init__(self, model, image_height=64, image_width=192, device="cuda"):
        super().__init__()
        self.model = model.to(device)
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.noise_steps = 1000

        self.beta = torch.linspace(1e-4, 0.02, self.noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def forward(self, high_res_img, low_res_img):
        """
        TRAINING: Calculates loss between actual noise and predicted noise.
        high_res_img: The ground truth plate (Batch, 3, H, W)
        low_res_img: The input low-quality plate (Batch, 3, h, w)
        """
        # Upscale LR image to match HR dimensions (Bicubic)
        low_res_upscaled = F.interpolate(
            low_res_img, size=high_res_img.shape[2:], mode="bicubic"
        )

        t = self.sample_timesteps(high_res_img.shape[0]).to(self.device)
        x_t, noise = self.noise_images(high_res_img, t)

        # Predict noise based on x_t and the LR condition
        predicted_noise = self.model(x_t, t, low_res_upscaled)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def super_resolve(self, low_res_img):
        """
        INFERENCE: Generates a High-Res plate from a Low-Res input.
        """
        self.model.eval()
        n = low_res_img.shape[0]

        x = torch.randn((n, 3, self.image_height, self.image_width)).to(self.device)

        low_res_upscaled = F.interpolate(
            low_res_img, size=(self.image_height, self.image_width), mode="bicubic"
        )

        # Denoise step-by-step
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = self.model(x, t, low_res_upscaled)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Standard DDPM sampling formula
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
            ) + torch.sqrt(beta) * noise

        self.model.train()

        # Clamp to valid image range [-1, 1] or [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x


# ==========================================
# Example Usage Script
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_height, image_width = 64, 128
    unet = DiffPlateUNet(image_height=image_height, image_width=image_width)
    diff_plate = DiffPlate(unet, image_height=image_height, image_width=image_width, device=device)

    hr_imgs = torch.randn(4, 3, image_height, image_width).to(device)
    lr_imgs = torch.randn(4, 3, image_height // 2, image_width // 2).to(device)

    loss = diff_plate(hr_imgs, lr_imgs)
    print(f"Training Loss: {loss.item()}")

    print("Running Super Resolution...")
    sr_imgs = diff_plate.super_resolve(lr_imgs)
    print(f"Output Shape: {sr_imgs.shape}")
