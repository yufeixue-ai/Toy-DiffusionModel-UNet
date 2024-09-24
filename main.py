import torch.nn as nn
import torch
import math
from diff_utils import Unet, ExponentialMovingAverage
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def diffusion_mnist_dataloader(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="../data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="../data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
            
class DiffusionToyModelforMNIST(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults= [1, 2, 4, 8]):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size

        betas=self._cosine_variance_schedule_generator(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas, dim=-1)

        # It is an optional choice to use register_buffer. Other methods are applicable as well.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self,x,noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t)

        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            x_t=self._reverse_diffusion(x_t,t,noise)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t
    
    def _cosine_variance_schedule_generator(self, timesteps, epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        #### Fill in the forward process here ####
        
        # x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        x_t = torch.sqrt(alpha_t_cumprod) * x_0 + torch.sqrt(1.-alpha_t_cumprod) * noise
        return x_t
        
        #### Fill in the forward process here ####


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        #### Fill in the backward process here ####
        sqrt_alphas_t=torch.sqrt(alpha_t)
        mean = (1/sqrt_alphas_t) * (x_t - ((1.-alpha_t) / torch.sqrt(1-alpha_t_cumprod)) * pred)
        #### Fill in the backward process here ####

        if t.min()>0:
            #### Fill in the backward process here ####
            alpha_t_1_cumprod=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            var = ( (1.-alpha_t_1_cumprod) / (1.-alpha_t_cumprod) ) * beta_t
            std=torch.sqrt(var)           
            #### Fill in the backward process here ####
        else:
            #### Fill in the backward process here ####

            #### Fill in the backward process here ####
            std=0.0

        return mean+std*noise 

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128   # If default 128 batch size is too large for your GPU, try reducing it.
model_ema_steps = 10
model_ema_decay = 0.995  # These two parameters are used for the Exponential Moving Average (EMA) of the model. 
log_frequency = 50  # The frequency for printing the log message during training.
lr = 0.001  # Learning rate.
timesteps = 1000      # Sampling steps of the diffusion DDPM process.

# We do not recommand you to change below hyperparameters, maintaining a fair training comparison and unified visualization.
epochs = 100
n_samples = 36
model_base_dim = 64   # The base dimension of the UNet.

print("device:",device)
train_dataloader,test_dataloader=diffusion_mnist_dataloader(batch_size=batch_size,image_size=28)
model=DiffusionToyModelforMNIST(timesteps=timesteps,
            image_size=28,
            in_channels=1,
            base_dim=model_base_dim,
            dim_mults=[2,4]).to(device)

#torchvision ema setting
#https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
adjust = 1 * batch_size * model_ema_steps / epochs
alpha = 1.0 - model_ema_decay
alpha = min(1.0, alpha * adjust)
model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

optimizer=AdamW(model.parameters(),lr=lr)
scheduler=OneCycleLR(optimizer,lr,total_steps=epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
loss_fn=nn.MSELoss(reduction='mean')


global_steps=0
for i in range(epochs):
    model.train()
    for j, (image, target) in enumerate(train_dataloader):
        noise=torch.randn_like(image).to(device)
        image=image.to(device)
        #### Fill in the training process here ####
        #### calcualte the loss and optimize the model ####
        out = model(image, noise)
        loss = loss_fn(out, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       
        
        #### Fill in the training process here ####
        scheduler.step()
        if global_steps % model_ema_steps==0:
            model_ema.update_parameters(model)
        global_steps += 1
        if j % log_frequency ==0:
            print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1, epochs, j, len(train_dataloader),
                                                                loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
    ckpt={"model":model.state_dict(),
            "model_ema":model_ema.state_dict()}

    os.makedirs("results",exist_ok=True)
    torch.save(ckpt,"results/steps_{:0>8}.pt".format(i+1))

    model_ema.eval()
    samples=model_ema.module.sampling(n_samples, clipped_reverse_diffusion=True, device=device)
    # You may need to customize this directory to fit your own device.
    save_image(samples,"results/steps_{:0>8}.png".format(i+1),nrow=int(math.sqrt(n_samples)))
    
    # Optional: You can visualize the generated samples in notebook by uncommenting the following code.
    # grid = make_grid(samples, nrow=int(math.sqrt(n_samples)))
    # plt.figure(figsize=(12, 6))
    # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    # plt.axis('off')
    # plt.show()