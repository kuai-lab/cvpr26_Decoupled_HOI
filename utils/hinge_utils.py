import torch
import torch.nn.functional as F

# ------------------------------
# Hinge utilities (sequence-aware)
# ------------------------------
def d_hinge_global(real_logit, fake_logit):
    # real_logit, fake_logit: [B,1]
    return F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()

def masked_mean(x_bt1, pm_bt):
    # x_bt1: [B,T,1], pm_bt: [B,T] with 1=valid
    w = pm_bt.float().unsqueeze(-1)          # [B,T,1]
    denom = w.sum().clamp_min(1.0)
    return (x_bt1 * w).sum() / denom

def d_hinge_seq(real_seq_logit, fake_seq_logit, pm_bt):
    # real_seq_logit, fake_seq_logit: [B,T,1]
    # pm_bt: [B,T]
    r = F.relu(1 - real_seq_logit)
    f = F.relu(1 + fake_seq_logit)
    return masked_mean(r, pm_bt) + masked_mean(f, pm_bt)

def g_hinge_global(fake_logit):
    # fake_logit: [B,1]
    return (-fake_logit).mean()

def g_hinge_seq(fake_seq_logit, pm_bt):
    # fake_seq_logit: [B,T,1]
    return -masked_mean(fake_seq_logit, pm_bt)

# ------------------------------
# Optional contact smoothing for discriminator stability
# ------------------------------
def soften_binary_for_D(x, eps=0.05, noise_std=0.02):
    # Smooth binary ground-truth labels to better match probabilistic predictions.
    x = x * (1 - 2*eps) + eps
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x.clamp(0., 1.)
