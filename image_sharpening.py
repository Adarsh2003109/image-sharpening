import os
import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torch.amp import GradScaler
from torchvision.models import vgg16, VGG16_Weights
from piq import ssim
import gdown
import math
from einops import rearrange

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    torch.backends.cudnn.benchmark = True


# %%
class Config:
   
    TRAIN_BLUR = r"C:\Users\Lenovo\Downloads\image_sharpening_kd_project (2)\image_sharpening_kd_project\data\Gopro\train\blur"
    TRAIN_SHARP_PATH = r"C:\Users\Lenovo\Downloads\image_sharpening_kd_project (2)\image_sharpening_kd_project\data\Gopro\train\sharp"
    TEST_BLUR_PATH = r"C:\Users\Lenovo\Downloads\image_sharpening_kd_project (2)\image_sharpening_kd_project\data\Gopro\test\blur"
    TEST_SHARP_PATH = r"C:\Users\Lenovo\Downloads\image_sharpening_kd_project (2)\image_sharpening_kd_project\data\Gopro\test\sharp"
   
    
    # Training parameters
    BATCH_SIZE = 8  # Increased for GPU
    PATCH_SIZE = 256
    NUM_EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # Windows often has issues with multiprocessing
    
    # Model paths
    TEACHER_WEIGHTS = "motion_deblurring.pth"
    STUDENT_SAVE_PATH = "student_model.pth"
    
    # Preloading cache
    TRAIN_CACHE_PATH = "train_dataset_cache.pkl"
    TEST_CACHE_PATH = "test_dataset_cache.pkl"
    
    # Evaluation
    BENCHMARK_SIZE = 100
    TARGET_RES = (1920, 1080)

config = Config()

# Verify paths
print("\nPath verification:")
for path_type in ['TRAIN_BLUR', 'TRAIN_SHARP', 'TEST_BLUR', 'TEST_SHARP']:
    path = getattr(config, f"{path_type}_PATH")
    exists = os.path.exists(path)
    print(f"{path_type}: {'✅' if exists else '❌'} {path}")
    if exists:
        print(f"   Contains {len(os.listdir(path))} files")

# %% [markdown]
## Fixed Restormer Implementation (WithBias Version)

# %%
# Restormer components with WithBias LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        return self.body(x)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight[None, :, None, None]

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        x_normalized = (x - mu) / torch.sqrt(sigma + 1e-5)
        return x_normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

# Security registration for model loading
try:
    torch.serialization.add_safe_globals([Restormer])
    print("Added Restormer to safe globals for secure model loading")
except AttributeError:
    print("Warning: torch.serialization.add_safe_globals not available in this PyTorch version")

# %% [markdown]
## Download Teacher Weights

# %%
if not os.path.exists(config.TEACHER_WEIGHTS):
    print("Downloading teacher weights...")
    gdown.download(
        "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth", 
        config.TEACHER_WEIGHTS, 
        quiet=False
    )
    print("Download complete!")

# %% [markdown]
## Model Definitions (GPU Optimized)

# %%
# Lightweight Student Model with memory optimizations
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder with inplace ReLU to save memory
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.output = nn.Conv2d(32, 3, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)

# Teacher Model
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Restormer()
        
        # Load weights with explicit weights_only=False
        state_dict = torch.load(config.TEACHER_WEIGHTS, map_location=config.DEVICE, weights_only=False)
        self.model.load_state_dict(state_dict['params'])
        self.model.eval()
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

# Initialize models with GPU optimization
teacher = TeacherModel().to(config.DEVICE)
student = StudentModel().to(config.DEVICE)

# Freeze teacher model
for param in teacher.parameters():
    param.requires_grad = False

print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M")
print(f"Student parameters: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")

# Test models on GPU
if torch.cuda.is_available():
    with torch.no_grad():
        test_input = torch.randn(1, 3, 256, 256).to(config.DEVICE)
        output = teacher(test_input)
        print(f"Teacher test output shape: {output.shape}")
        output = student(test_input)
        print(f"Student test output shape: {output.shape}")

# %% [markdown]
## Dataset Preparation with Preloading Cache

# %%
class GoProDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, patch_size=256, train=True, cache_path=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.patch_size = patch_size
        self.train = train
        self.cache_path = cache_path
        
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading preloaded dataset from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.blur_images, self.sharp_images = pickle.load(f)
        else:
            self.blur_files = sorted(os.listdir(blur_dir))
            self.sharp_files = sorted(os.listdir(sharp_dir))
            
            assert len(self.blur_files) == len(self.sharp_files), "Mismatched dataset sizes"
            
            # Preload images to RAM
            print(f"Preloading {len(self.blur_files)} images...")
            self.blur_images = []
            self.sharp_images = []
            for i in tqdm(range(len(self.blur_files))):
                self.blur_images.append(Image.open(os.path.join(blur_dir, self.blur_files[i])).convert('RGB'))
                self.sharp_images.append(Image.open(os.path.join(sharp_dir, self.sharp_files[i])).convert('RGB'))
            print("Preloading complete!")
            
            # Save to cache if path provided
            if cache_path:
                print(f"Saving preloaded dataset to cache: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump((self.blur_images, self.sharp_images), f)

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blur_img = self.blur_images[idx]
        sharp_img = self.sharp_images[idx]
        
        if self.train:
            w, h = blur_img.size
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            blur_img = blur_img.crop((x, y, x+self.patch_size, y+self.patch_size))
            sharp_img = sharp_img.crop((x, y, x+self.patch_size, y+self.patch_size))
            
            if random.random() > 0.5:
                blur_img = blur_img.transpose(Image.FLIP_LEFT_RIGHT)
                sharp_img = sharp_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            blur_img = blur_img.resize(config.TARGET_RES)
            sharp_img = sharp_img.resize(config.TARGET_RES)
        
        # Convert to tensor
        blur_tensor = TF.to_tensor(blur_img)
        sharp_tensor = TF.to_tensor(sharp_img)
        
        return blur_tensor, sharp_tensor

# Create datasets with caching
train_dataset = GoProDataset(
    config.TRAIN_BLUR_PATH,
    config.TRAIN_SHARP_PATH,
    config.PATCH_SIZE,
    train=True,
    cache_path=config.TRAIN_CACHE_PATH
)

test_dataset = GoProDataset(
    config.TEST_BLUR_PATH,
    config.TEST_SHARP_PATH,
    config.PATCH_SIZE,
    train=False,
    cache_path=config.TEST_CACHE_PATH
)

# DataLoaders with GPU optimizations
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if config.NUM_WORKERS > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

# Save sample images instead of showing to prevent blocking
os.makedirs("debug", exist_ok=True)
blur, sharp = next(iter(train_loader))
print(f"Batch shape: {blur.shape}")

def save_tensor_image(tensor, path):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

save_tensor_image(blur[0], "debug/blurry_sample.jpg")
save_tensor_image(sharp[0], "debug/sharp_sample.jpg")
print("Saved sample images to debug/ folder!")

# %% [markdown]
## Training Setup (GPU Optimized)

# %%
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Use modern weights API
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval().to(config.DEVICE)
        
        self.l1_loss = nn.L1Loss()
        
    def forward(self, student_out, teacher_out, target):
        rec_loss = self.l1_loss(student_out, target)
        s_features = self.vgg(student_out)
        with torch.no_grad():
            t_features = self.vgg(teacher_out)
        perc_loss = self.l1_loss(s_features, t_features)
        feat_loss = self.l1_loss(student_out, teacher_out)
        return (self.alpha * rec_loss + 
                self.beta * perc_loss + 
                self.gamma * feat_loss)

criterion = DistillationLoss().to(config.DEVICE)
optimizer = torch.optim.Adam(student.parameters(), lr=config.LR)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Mixed precision scaler (updated API)
scaler = torch.amp.GradScaler()

# Training checkpoint
def save_checkpoint(epoch, history):
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_ssim': best_ssim
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
    print(f"Saved checkpoint at epoch {epoch}")

# %% [markdown]
## Training Loop (GPU Optimized)

# %%
def validate():
    student.eval()
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for blur, sharp in test_loader:
            blur = blur.to(config.DEVICE, non_blocking=True)
            sharp = sharp.to(config.DEVICE, non_blocking=True)
            
            # Run in full precision for validation
            output = student(blur)
            
            # Clamp output to [0, 1] range for SSIM calculation
            output_clamped = output.clamp(0, 1)
            
            # Convert to float32 for SSIM calculation
            total_ssim += ssim(
                output_clamped.float(), 
                sharp.float(), 
                data_range=1.0
            ).item()
            count += 1
            
            if count >= config.BENCHMARK_SIZE:
                break
    
    return total_ssim / count

# Training with GPU optimizations
best_ssim = 0.0
history = {'loss': [], 'ssim': []}

# Try to load checkpoint
start_epoch = 0
checkpoint_files = [f for f in os.listdir() if f.startswith('checkpoint_epoch_')]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE)
    student.load_state_dict(checkpoint['student_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    history = checkpoint['history']
    best_ssim = checkpoint['best_ssim']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

print("Starting training...")
for epoch in range(start_epoch, config.NUM_EPOCHS):
    student.train()
    epoch_loss = 0.0
    steps = 0
    
    # Optimized progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", total=len(train_loader))
    for blur, sharp in pbar:
        # Async data transfer
        blur = blur.to(config.DEVICE, non_blocking=True)
        sharp = sharp.to(config.DEVICE, non_blocking=True)
        
        with torch.no_grad():
            # Updated autocast API
            with torch.amp.autocast(device_type=config.DEVICE.type, dtype=torch.float16):
                teacher_out = teacher(blur)
        
        # Mixed precision training
        optimizer.zero_grad(set_to_none=True)
        
        # Updated autocast API
        with torch.amp.autocast(device_type=config.DEVICE.type, dtype=torch.float16):
            student_out = student(blur)
            loss = criterion(student_out, teacher_out, sharp)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        steps += 1
        
        # Update progress bar
        pbar.set_postfix(loss=loss.item())
    
    # Validation and checkpointing
    avg_ssim = validate()
    epoch_loss /= len(train_loader)
    scheduler.step(avg_ssim)
    
    history['loss'].append(epoch_loss)
    history['ssim'].append(avg_ssim)
    
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        torch.save(student.state_dict(), config.STUDENT_SAVE_PATH)
        print(f"Saved best model (SSIM: {best_ssim:.4f})")
    
    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {epoch_loss:.4f} | SSIM: {avg_ssim:.4f}")
    
    # Save checkpoint every epoch
    save_checkpoint(epoch, history)
    
    # Manual GPU utilization report
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {mem_used:.2f}/{mem_total:.2f} GB ({mem_used/mem_total:.1%})")

# Final model save
torch.save(student.state_dict(), "final_student_model.pth")

# %% [markdown]
## Evaluation & Results (GPU Optimized)

# %%
# Load best model
student.load_state_dict(torch.load(config.STUDENT_SAVE_PATH))
student.eval()

# Test SSIM
def test_ssim():
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for blur, sharp in test_loader:
            blur = blur.to(config.DEVICE)
            sharp = sharp.to(config.DEVICE)
            
            output = student(blur)
            output_clamped = output.clamp(0, 1)
            
            # Convert to float32 for SSIM
            total_ssim += ssim(
                output_clamped.float(), 
                sharp.float(), 
                data_range=1.0
            ).item()
            count += 1
            
            if count >= config.BENCHMARK_SIZE:
                break
    
    return total_ssim / count

# Test speed
def test_fps():
    student.eval()
    dummy_input = torch.randn(1, 3, *config.TARGET_RES).to(config.DEVICE)
    
    # Warmup
    for _ in range(10):
        _ = student(dummy_input)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = student(dummy_input)
    torch.cuda.synchronize()  # Wait for GPU to finish
    elapsed = time.time() - start
    
    return 100 / elapsed

# Run evaluation
avg_ssim = test_ssim()
fps = test_fps()

print(f"\n{'='*40}")
print(f"Final Evaluation:")
print(f"{'='*40}")
print(f"SSIM: {avg_ssim:.4f}")
print(f"FPS: {fps:.2f} @ {config.TARGET_RES}")
print(f"{'='*40}")

# Save visual comparison
blur, sharp = next(iter(test_loader))
with torch.no_grad():
    output = student(blur.to(config.DEVICE)).cpu()

def save_tensor_image(tensor, path):
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

save_tensor_image(blur[0], "debug/input_blurry.jpg")
save_tensor_image(output[0], "debug/output_sharpened.jpg")
save_tensor_image(sharp[0], "debug/ground_truth.jpg")
print("Saved comparison images to debug/ folder!")