import torch 
import numpy as np 
from tqdm import tqdm


def broadcast(values, broadcast_shape):
    values = values.flatten() 
    while len(values.shape) < len(broadcast_shape):
        values = values.unsqueeze(-1)
    return values 

class DDPM:
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=1e-2):
        self.model = model
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)


    def forward_diffusion(self, images, timesteps):
        gaussian_noise = torch.randn(images.shape, device=images.device)
        alpha_hat = self.alphas_hat[timesteps].to(images.device)
        alpha_hat = broadcast(alpha_hat, images.shape)

        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise
    def reverse_diffusion(self, model, noisy_images, timesteps):
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, device, save_all_steps=False):
        image = initial_noise 
        all_images = [] 
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Step 1: Predict noise
            ts = timestep * torch.ones((image.shape[0],), dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Step 2 
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t 
            variance = torch.sqrt(beta_t_hat) * torch.randn(image.shape, device=device) if timestep > 0 else 0 

            # Step 3: Update image
            image = torch.pow(alpha_t, -0.5) * (image - beta_t / torch.sqrt((1 - alpha_hat_prev)) * predicted_noise) + variance

            if save_all_steps:
                all_images.append(image.cpu().numpy())
        return all_images if save_all_steps else image.cpu().numpy() 

    # def forward_single_diffusion(self, x_0, t):
    #     noise = torch.randn_like(x_0)
    #     alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
    #     return torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
    # def reverse_single_diffusion(self, x_t, t):
    #     alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
    #     beta_t = self.betas[t].view(-1, 1, 1, 1)
    #     predicted_noise = self.model(x_t, t)
    #     return (x_t - beta_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        

    
