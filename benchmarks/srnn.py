import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import sys
import psutil
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from statistics import mean, stdev
import os
from scipy.ndimage import gaussian_filter


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)  # MB


# Performance optimization: Control verbose logging via environment variable
VERBOSE = os.environ.get('SRNN_VERBOSE', '0') == '1'


class NoisyMNIST(torchvision.datasets.MNIST):
    def __init__(self, root="./data", train=True, noise_std=0.3, download=True, cache_noise=False):
        transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, train=train, transform=transform, download=download)
        self.noise_std = noise_std
        self.cache_noise = cache_noise
        self._noise_cache = {}

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        # Optionally cache noise patterns for consistent evaluation
        if self.cache_noise and idx in self._noise_cache:
            noise = self._noise_cache[idx]
        else:
            noise = self.noise_std * torch.randn_like(img)
            if self.cache_noise:
                self._noise_cache[idx] = noise
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        return noisy_img, img


class ConvSR_ESN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        res_channels=4,
        spec_rad=0.98,
        sparsity=0.9,
        alpha=2.0,
        sigma=1e-6,
        dt=1.0,
    ):
        if VERBOSE:
            print("Initializing ConvSR_ESN...")
            print(f"Memory before init: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        super().__init__()
        self.res_channels = res_channels
        self.alpha = alpha
        self.sigma = sigma
        self.dt = dt
        self.conv_in = nn.Conv2d(in_channels, res_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(res_channels, res_channels, kernel_size=3, padding=1)
        # Manual sparsity: zero out fraction of weights
        mask = (
            torch.rand_like(self.conv_res.weight) < sparsity
        )  # Zero sparsity fraction
        self.conv_res.weight.data[mask] = 0.0
        self.readout = nn.Conv2d(res_channels, in_channels, kernel_size=1)
        if VERBOSE:
            print("ConvSR_ESN initialized.")
            print(f"Memory after init: {get_memory_usage():.2f} MB")
            sys.stdout.flush()

    def collect_states(self, x, add_noise=True, steps=1, hybrid_tanh=False):
        if VERBOSE:
            print("Starting collect_states...")
            print(f"Input shape: {x.shape}")
            print(f"Memory before collect: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        h = torch.zeros_like(self.conv_in(x))
        for step in range(steps):
            if VERBOSE:
                print(f"collect_states step {step + 1}/{steps}")
                sys.stdout.flush()
            s_in = self.conv_in(x)
            s_rec = self.conv_res(h)
            s = s_in + s_rec
            noise = (
                math.sqrt(2 * self.sigma) * torch.randn_like(h)
                if add_noise
                else torch.zeros_like(h)
            )
            dh = self.alpha * (h - h**3) + noise
            h = h + (dh + s) * self.dt
            h = torch.clamp(h, -10.0, 10.0)
            if hybrid_tanh:
                h = torch.tanh(h)
            if VERBOSE:
                print(f"Memory during step {step + 1}: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
        if VERBOSE:
            print("collect_states completed.")
            print(f"Memory after collect: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        return self.readout(h)


class Baseline_ConvESN(nn.Module):
    def __init__(self, in_channels=1, res_channels=4, spec_rad=0.98, sparsity=0.9):
        if VERBOSE:
            print("Initializing Baseline_ConvESN...")
            print(f"Memory before init: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        super().__init__()
        self.res_channels = res_channels
        self.conv_in = nn.Conv2d(in_channels, res_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(res_channels, res_channels, kernel_size=3, padding=1)
        # Manual sparsity
        mask = torch.rand_like(self.conv_res.weight) < sparsity
        self.conv_res.weight.data[mask] = 0.0
        self.readout = nn.Conv2d(res_channels, in_channels, kernel_size=1)
        if VERBOSE:
            print("Baseline_ConvESN initialized.")
            print(f"Memory after init: {get_memory_usage():.2f} MB")
            sys.stdout.flush()

    def collect_states(self, x, add_noise=False, steps=1, hybrid_tanh=True):
        if VERBOSE:
            print("Starting collect_states for baseline...")
            print(f"Input shape: {x.shape}")
            print(f"Memory before collect: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        h = torch.zeros_like(self.conv_in(x))
        for step in range(steps):
            if VERBOSE:
                print(f"collect_states step {step + 1}/{steps} for baseline")
                sys.stdout.flush()
            h = torch.tanh(self.conv_in(x) + self.conv_res(h))
            if VERBOSE:
                print(f"Memory during step {step + 1}: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
        if VERBOSE:
            print("collect_states completed for baseline.")
            print(f"Memory after collect: {get_memory_usage():.2f} MB")
            sys.stdout.flush()
        return self.readout(h)


def psnr(mse, max_val=1.0):
    return -10 * math.log10(mse) if mse > 0 else float("inf")


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1**2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, 1.5) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return np.mean(ssim_map)


@dataclass
class Result:
    label: str
    test_mse: float
    test_psnr: float
    test_ssim: float
    train_duration: float
    infer_duration: float


def compute_noisy_psnr(test_loader):
    criterion = nn.MSELoss()
    noisy_mse = 0
    num = 0
    for noisy, clean in test_loader:
        noisy_mse += criterion(noisy, clean).item()
        num += 1
    avg = noisy_mse / num
    return avg, psnr(avg)


def train_denoise(
    model,
    train_loader,
    test_loader,
    device,
    label,
    epochs=1,
    lr=0.001,
    pretrained_path=None,
):
    criterion = nn.MSELoss()
    train_duration = 0.0
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_start = time.time()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            if VERBOSE:
                print(f"Memory before epoch: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
            model.train()
            train_loss = 0
            for batch_idx, (noisy, clean) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not VERBOSE)
            ):
                if VERBOSE:
                    print(
                        f"Processing batch {batch_idx + 1}/{len(train_loader)} in epoch {epoch + 1}"
                    )
                    print(f"Memory before batch: {get_memory_usage():.2f} MB")
                    sys.stdout.flush()
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                out = model.collect_states(
                    noisy, add_noise=isinstance(model, ConvSR_ESN)
                )
                loss = criterion(out, clean)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if VERBOSE:
                    print(f"Batch {batch_idx + 1} completed, loss: {loss.item():.4f}")
                    print(f"Memory after batch: {get_memory_usage():.2f} MB")
                    sys.stdout.flush()
                del out, loss, noisy, clean
                torch.cuda.empty_cache()
            print(
                f"Epoch {epoch + 1} completed, Train Loss: {train_loss / len(train_loader):.4f}"
            )
            if VERBOSE:
                print(f"Memory after epoch: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
        train_duration = time.time() - train_start
        torch.save(model.state_dict(), f"{label}_pretrained.pth")
        print(f"Saved pretrained model to {label}_pretrained.pth")

    infer_start = time.time()
    model.eval()
    test_mse = 0
    test_ssim = 0
    # Use inference_mode for better performance than no_grad
    with torch.inference_mode():
        for batch_idx, (noisy, clean) in enumerate(tqdm(test_loader, desc="Testing", disable=not VERBOSE)):
            if VERBOSE:
                print(f"Processing test batch {batch_idx + 1}/{len(test_loader)}")
                print(f"Memory before test batch: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
            noisy, clean = noisy.to(device), clean.to(device)
            out = model.collect_states(
                noisy, add_noise=False if isinstance(model, ConvSR_ESN) else False
            )
            test_mse += criterion(out, clean).item()
            # SSIM average over batch
            batch_ssim = 0
            for i in range(out.size(0)):
                batch_ssim += ssim(out[i], clean[i])
            test_ssim += batch_ssim / out.size(0)
            if VERBOSE:
                print(f"Test batch {batch_idx + 1} completed")
                print(f"Memory after test batch: {get_memory_usage():.2f} MB")
                sys.stdout.flush()
            del out, noisy, clean
    avg_mse = test_mse / len(test_loader)
    avg_ssim = test_ssim / len(test_loader)
    avg_psnr = psnr(avg_mse)
    infer_duration = time.time() - infer_start
    print(
        f"[{label}] Test MSE: {avg_mse:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}"
    )
    print(f"[{label}] Train Duration: {train_duration:.2f} seconds")
    print(f"[{label}] Infer Duration: {infer_duration:.2f} seconds")
    if VERBOSE:
        sys.stdout.flush()

    # Visualize sample only on first run to avoid overhead
    if VERBOSE:
        try:
            noisy, clean = next(iter(test_loader))
            noisy, clean = noisy[0:4].to(device), clean[0:4]
            out = model.collect_states(
                noisy, add_noise=False if isinstance(model, ConvSR_ESN) else False
            )[:4]
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                axs[0, i].imshow(clean[i].squeeze().cpu(), cmap="gray")
                axs[0, i].set_title("Clean")
                axs[1, i].imshow(noisy[i].squeeze().cpu(), cmap="gray")
                axs[1, i].set_title("Noisy")
                axs[2, i].imshow(out[i].squeeze().cpu().detach(), cmap="gray")
                axs[2, i].set_title("Denoised")
            plt.savefig("denoise_example.png")
            print("Saved visualization to 'denoise_example.png'")
            sys.stdout.flush()
        except Exception as e:
            print(f"Visualization failed: {e}")
            sys.stdout.flush()

    return Result(label, avg_mse, avg_psnr, avg_ssim, train_duration, infer_duration)


def run_benchmark(batch_size=64, device=torch.device("cpu"), num_runs=3):
    print("Loading datasets...")
    if VERBOSE:
        print(f"Memory before load: {get_memory_usage():.2f} MB")
        sys.stdout.flush()
    full_train_dataset = NoisyMNIST(train=True, noise_std=0.3)
    full_test_dataset = NoisyMNIST(train=False, noise_std=0.5)  # Higher test noise
    # Subsample for efficiency
    train_indices = list(range(0, len(full_train_dataset), 10))  # ~6k samples
    test_indices = list(range(0, len(full_test_dataset), 10))  # ~1k
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    # Use more workers for faster data loading
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print("Datasets loaded.")
    if VERBOSE:
        print(f"Memory after load: {get_memory_usage():.2f} MB")
        sys.stdout.flush()

    noisy_mse, noisy_psnr = compute_noisy_psnr(test_loader)
    print(f"Noisy Input Baseline MSE: {noisy_mse:.4f}, PSNR: {noisy_psnr:.2f} dB")

    configs = [
        ("ConvSR_ESN", ConvSR_ESN),
        ("Baseline_ConvESN", Baseline_ConvESN),
    ]
    results = {}
    for label, ctor in configs:
        mses = []
        psnrs = []
        ssims = []
        train_durs = []
        infer_durs = []
        for run in range(num_runs):
            print(f"\n=== Running {label} Run {run+1}/{num_runs} ===")
            if VERBOSE:
                sys.stdout.flush()
            model = ctor().to(device)
            pretrained_path = f"{label}_pretrained.pth"
            metrics = train_denoise(
                model,
                train_loader,
                test_loader,
                device,
                label,
                pretrained_path=pretrained_path if run > 0 else None,
            )
            mses.append(metrics.test_mse)
            psnrs.append(metrics.test_psnr)
            ssims.append(metrics.test_ssim)
            train_durs.append(metrics.train_duration)
            infer_durs.append(metrics.infer_duration)
        avg_mse = mean(mses)
        avg_psnr = mean(psnrs)
        avg_ssim = mean(ssims)
        avg_train_dur = mean(train_durs)
        avg_infer_dur = mean(infer_durs)
        std_mse = stdev(mses) if num_runs > 1 else 0
        std_psnr = stdev(psnrs) if num_runs > 1 else 0
        std_ssim = stdev(ssims) if num_runs > 1 else 0
        std_train_dur = stdev(train_durs) if num_runs > 1 else 0
        std_infer_dur = stdev(infer_durs) if num_runs > 1 else 0
        results[label] = (
            avg_mse,
            avg_psnr,
            avg_ssim,
            avg_train_dur,
            avg_infer_dur,
            std_mse,
            std_psnr,
            std_ssim,
            std_train_dur,
            std_infer_dur,
        )

    print("\n=== Summary (Averages over {num_runs} runs) ===")
    print(
        f"{'Model':<15} {'Avg MSE (±std)':<15} {'Avg PSNR (±std)':<15} {'Avg SSIM (±std)':<15} {'Avg Train Dur (±std)':<20} {'Avg Infer Dur (±std)':<20}"
    )
    for label in results:
        (
            avg_mse,
            avg_psnr,
            avg_ssim,
            avg_train_dur,
            avg_infer_dur,
            std_mse,
            std_psnr,
            std_ssim,
            std_train_dur,
            std_infer_dur,
        ) = results[label]
        print(
            f"{label:<15} {avg_mse:.4f} (±{std_mse:.4f}) {avg_psnr:.2f} (±{std_psnr:.2f}) {avg_ssim:.4f} (±{std_ssim:.4f}) {avg_train_dur:.2f} (±{std_train_dur:.2f}) {avg_infer_dur:.2f} (±{std_infer_dur:.2f})"
        )


if __name__ == "__main__":
    run_benchmark()
