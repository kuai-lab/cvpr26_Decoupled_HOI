import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce

from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from manip.data.cano_traj_dataset import quat_fk_torch, quat_ik_torch 

from manip.model.transformer_module import (
    MotionTransformerDecoder,
    TemporalConvResidualBlock,
    maybe_apply_spectral_norm,
)
from manip.lafan1.utils import rotate_at_frame_w_obj_global, rotate_at_frame_w_obj, quat_slerp 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def wxyz_to_xyzw(input_quat):
    w = input_quat[:, :, :, 0:1]
    x = input_quat[:, :, :, 1:2]
    y = input_quat[:, :, :, 2:3]
    z = input_quat[:, :, :, 3:4]

    return torch.cat((x, y, z, w), dim=-1)

def xyzw_to_wxyz(input_quat):
    x = input_quat[:, :, :, 0:1]
    y = input_quat[:, :, :, 1:2]
    z = input_quat[:, :, :, 2:3]
    w = input_quat[:, :, :, 3:4]

    return torch.cat((w, x, y, z), dim=-1)

def interpolate_transition(prev_obj_com_pos, prev_obj_rot_mat, prev_jpos, prev_rot_6d, window_obj_com_pos, \
                        window_obj_rot_mat, window_jpos, window_rot_6d):
    num_overlap_frames = prev_jpos.shape[1]

    fade_out = torch.linspace(1, 0, num_overlap_frames)[None, :, None].to(prev_jpos.device)
    fade_in = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(prev_jpos.device)

    window_obj_com_pos[:, :num_overlap_frames, :] = fade_out * prev_obj_com_pos + \
        fade_in * window_obj_com_pos[:, :num_overlap_frames, :]  
    window_jpos[:, :num_overlap_frames, :, :] = fade_out[:, :, None, :] * prev_jpos + \
        fade_in[:, :, None, :] * window_jpos[:, :num_overlap_frames, :, :]

    slerp_weight = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(prev_rot_6d.device)
   
    prev_obj_q = transforms.matrix_to_quaternion(prev_obj_rot_mat)
    window_obj_q = transforms.matrix_to_quaternion(window_obj_rot_mat) # 1 X w X 4 

    prev_rot_mat = transforms.rotation_6d_to_matrix(prev_rot_6d)
    prev_q = transforms.matrix_to_quaternion(prev_rot_mat)
    window_rot_mat = transforms.rotation_6d_to_matrix(window_rot_6d) 
    window_q = transforms.matrix_to_quaternion(window_rot_mat) # 1 X w X 22 X 4 

    obj_q_left = prev_obj_q[:, :, None, :]
    obj_q_right = window_obj_q[:, :num_overlap_frames, None, :]

    human_q_left = prev_q.clone()
    human_q_right = window_q[:, :num_overlap_frames, :, :]

    obj_q_left = wxyz_to_xyzw(obj_q_left)
    obj_q_right = wxyz_to_xyzw(obj_q_right)
    human_q_left = wxyz_to_xyzw(human_q_left)
    human_q_right = wxyz_to_xyzw(human_q_right)

    slerped_obj_q = quat_slerp(obj_q_left, obj_q_right, slerp_weight)
    slerped_human_q = quat_slerp(human_q_left, human_q_right, slerp_weight)

    slerped_obj_q = xyzw_to_wxyz(slerped_obj_q)
    slerped_human_q = xyzw_to_wxyz(slerped_human_q)

    new_obj_q = torch.cat((slerped_obj_q.squeeze(2), window_obj_q[:, num_overlap_frames:, :]), dim=1)
    new_human_q = torch.cat((slerped_human_q, window_q[:, num_overlap_frames:, :, :]), dim=1)

    new_obj_rot_mat = transforms.quaternion_to_matrix(new_obj_q)
    new_human_rot_mat = transforms.quaternion_to_matrix(new_human_q)
    new_human_rot_6d = transforms.matrix_to_rotation_6d(new_human_rot_mat)

    return window_obj_com_pos, new_obj_rot_mat, window_jpos, new_human_rot_6d 

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        self.motion_transformer = MotionTransformerDecoder(input_dim=d_input_feats, model_dim=self.d_model, \
            num_layers=self.n_dec_layers, num_heads=self.n_head, key_dim=self.d_k, value_dim=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, bps_embed, language_embedding=None, padding_mask=None):
        src = torch.cat((src, bps_embed), dim=-1)

        noise_t_embed = self.time_mlp(noise_t)[:, None, :]
        language_embedding = language_embedding[:, None, :] if language_embedding is not None else None

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool()

        pos_vec = torch.arange(num_steps)+1
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)

        data_input = src.transpose(1, 2)
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed, \
                                               language_embedding=language_embedding)
       
        output = self.linear_out(feat_pred[:, 1:])

        return output
    

class TransformerDiffusionModelPath(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = 4
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        self.motion_transformer = MotionTransformerDecoder(input_dim=d_input_feats, model_dim=self.d_model, \
            num_layers=self.n_dec_layers, num_heads=self.n_head, key_dim=self.d_k, value_dim=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, 6)    # obj pos(3) + human root pos(3)

        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, bps_embed, language_embedding=None, padding_mask=None):
        src = torch.cat((src, bps_embed), dim=-1)

        noise_t_embed = self.time_mlp(noise_t)[:, None, :]
        language_embedding = language_embedding[:, None, :] if language_embedding is not None else None

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool()

        pos_vec = torch.arange(num_steps)+1
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)

        data_input = src.transpose(1, 2)
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed, \
                                               language_embedding=language_embedding)
       
        output = self.linear_out(feat_pred[:, 1:])

        return output
    

class HandObjectInteractionDiscriminator(nn.Module):
    """
    Input:
      x_concat: [B, T, D] where D = 4(hand_contact) + 12(hand_jnts) + 3(obj_pos) + 3*K(seq_obj_kpts)
      padding_mask (optional): [B,T] or [B,1,T], 1=valid, 0=pad

    Output:
      {
        "logits_seq": [B, T, 1],   # per-frame logits in a PatchGAN-style setup
        "logit":      [B, 1],      # sequence-global logit (masked mean)
      }
    """
    def __init__(
        self,
        in_dim: int,           # 4 + 12 + 3 + 3*K
        d_model: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        base_dilation: int = 1,   # Use 1 here and scale with 2**i below for 1, 2, 4, ...
        dropout: float = 0.1,
        use_spectral: bool = True,
        use_input_norm: bool = True,  # Lightweight LayerNorm to stabilize the input distribution
    ):
        super().__init__()
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(in_dim) if use_input_norm else nn.Identity()
        self.input_proj = maybe_apply_spectral_norm(nn.Linear(in_dim, d_model), use_spectral)

        self.tcn = nn.ModuleList([
            TemporalConvResidualBlock(d_model,
                                      kernel_size=kernel_size,
                                      dilation=(2 ** i) * base_dilation,
                                      dropout=dropout,
                                      use_spectral_norm=use_spectral)
            for i in range(num_layers)
        ])

        self.head_seq = maybe_apply_spectral_norm(nn.Linear(d_model, 1), use_spectral)
        self.head_global = maybe_apply_spectral_norm(nn.Linear(d_model, 1), use_spectral)

    @staticmethod
    def _masked_mean(x_btC, padding_mask):
        if padding_mask is None:
            return x_btC.mean(dim=1)  # [B,C]
        pm = padding_mask
        if pm.dim() == 3:   # [B,1,T]
            pm = pm[:, 0, :]
        pm = pm.float()
        denom = pm.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        w = pm.unsqueeze(-1)                                 # [B,T,1]
        return (x_btC * w).sum(dim=1) / denom               # [B,C]

    def forward(self, x_concat, padding_mask=None):
        x = self.input_norm(x_concat)
        x = self.input_proj(x)          # [B,T,d]
        for blk in self.tcn:
            x = blk(x)                  # [B,T,d]

        logits_seq = self.head_seq(x)   # [B,T,1]
        g = self._masked_mean(x, padding_mask)  # [B,d]
        logit = self.head_global(g)     # [B,1]
        return {"logits_seq": logits_seq, "logit": logit}


class ObjectConditionedGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        input_first_human_pose=False, 
        use_object_keypoints=False, 
    ):
        super().__init__()

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            )

        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            )

        self.input_first_human_pose = input_first_human_pose 
        
        self.use_object_keypoints = use_object_keypoints 

        obj_feats_dim = 256 
        d_input_feats = 2*d_feats+obj_feats_dim

        self.denoise_fn = TransformerDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, \
                    d_model=d_model, n_head=n_head, d_k=d_k, d_v=d_v, \
                    n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 

        self.denoise_fn_path = TransformerDiffusionModelPath(d_input_feats=d_input_feats, d_feats=d_feats, \
                    d_model=d_model, d_k=d_k, d_v=d_v, \
                    n_dec_layers=n_dec_layers, max_timesteps=max_timesteps)

        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=True):
        path_output = self.denoise_fn_path(x, t, x_cond, language_embedding, padding_mask) # BS X T X 6
        x[:, :, :3] = path_output[:, :, :3] # obj com pos
        x[:, :, 12:15] = path_output[:, :, 3:6] # human root pos

        model_output = self.denoise_fn(x, t, x_cond, language_embedding, padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_reconstruction_guidance(self, x, t, bps_embed, guidance_fn, \
                                    language_embedding=None, padding_mask=None, \
                                    rest_human_offsets=None, data_dict=None, \
                                    cond_mask=None, \
                                    prev_window_cano_rot_mat=None, \
                                    prev_window_init_root_trans=None, \
                                    contact_labels=None, \
                                    curr_window_ref_obj_rot_mat=None, \
                                    clip_denoised=True, \
                                    x_pose_cond=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            path_output = self.denoise_fn_path(x, t, bps_embed, language_embedding, padding_mask)

            x = torch.cat([
                            path_output[..., :3],
                            x[..., 3:12],
                            path_output[..., 3:6],
                            x[..., 15:]
                        ], dim=-1)

            model_output = self.denoise_fn(x, t, bps_embed, language_embedding, padding_mask)

            if self.objective == 'pred_noise':
                x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
            elif self.objective == 'pred_x0':
                x_start = model_output
            else:
                raise ValueError(f'unknown objective {self.objective}')

            x_pose_cond = x_pose_cond.detach() 

            x_pose_cond[..., :3] = path_output[..., :3]
            x_pose_cond[..., 12:15] = path_output[..., 3:6]

            classifier_scale = 1e3
        
            loss = guidance_fn(t, x_start, x_pose_cond, cond_mask, \
                rest_human_offsets, data_dict, \
                contact_labels=contact_labels, \
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                prev_window_init_root_trans=prev_window_init_root_trans)

            gradient = torch.autograd.grad(-loss, x_start)[0] * classifier_scale
            tmp_posterior_variance = extract(self.posterior_variance, t, x_start.shape)
            x_start = x_start + tmp_posterior_variance * gradient.float()

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_guided_reconstruction_guidance(self, x, t, x_cond, guidance_fn, language_embedding=None, \
                    clip_denoised=True, \
                    rest_human_offsets=None, data_dict=None, cond_mask=None, padding_mask=None, \
                    prev_window_cano_rot_mat=None, prev_window_init_root_trans=None, \
                    contact_labels=None, curr_window_ref_obj_rot_mat=None, x_pose_cond=None):
        b, *_ = x.shape

        model_mean, _, model_log_variance = self.p_mean_variance_reconstruction_guidance(x=x, t=t, bps_embed=x_cond, \
                                    guidance_fn=guidance_fn, language_embedding=language_embedding, \
                                    clip_denoised=clip_denoised, cond_mask=cond_mask, padding_mask=padding_mask, \
                                    rest_human_offsets=rest_human_offsets, data_dict=data_dict, \
                                    prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                                    prev_window_init_root_trans=prev_window_init_root_trans, \
                                    contact_labels=contact_labels, \
                                    curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                                    x_pose_cond=x_pose_cond)

        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop_guided(self, shape, x_start, x_cond, guidance_fn=None, language_embedding=None, opt_fn=None, \
                    rest_human_offsets=None, data_dict=None, contact_labels=None, \
                    cond_mask=None, padding_mask=None, cond_mask_path=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        only_clean_cond_path = x_start * (1. - cond_mask_path)

        x_pose_cond = x_start * (1. - cond_mask_path)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            
            only_noise_cond = x * cond_mask_path # BS X T X D
            x = only_clean_cond_path + only_noise_cond # BS X T X D

            if guidance_fn is not None and i > 0 and i < 10: 
                x = self.p_sample_guided_reconstruction_guidance(x, torch.full((b,), i, device=device, dtype=torch.long), \
                            x_cond, guidance_fn, language_embedding=language_embedding, \
                            rest_human_offsets=rest_human_offsets, \
                            data_dict=data_dict, contact_labels=contact_labels, \
                            cond_mask=cond_mask, padding_mask=padding_mask, x_pose_cond=x_pose_cond)  
            else:
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, \
                    language_embedding, padding_mask=padding_mask)

        return x # BS X T X D

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, language_embedding=language_embedding, \
            padding_mask=padding_mask, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, language_embedding=None, padding_mask=None, \
                cond_mask_path=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        only_clean_cond_path = x_start * (1. - cond_mask_path)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            only_noise_cond = x * cond_mask_path # BS X T X D
            x = only_clean_cond_path + only_noise_cond # BS X T X D

            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, language_embedding, \
            padding_mask=padding_mask)    

        return x
    
    def p_sample_loop_sliding_window_w_canonical(self, ds, object_names, trans2joint, \
                                x_start, ori_x_cond, cond_mask, padding_mask, \
                                overlap_frame_num=1, input_waypoints=False, contact_labels=None, language_input=None, \
                                rest_human_offsets=None, data_dict=None, \
                                guidance_fn=None, opt_fn=None):
        shape = x_start.shape 

        device = self.betas.device

        b = shape[0]
        
        x_all = torch.randn(shape, device=device)

        whole_sample_res = None # BS X T X D (3+9+24*3+22*6)  

        num_steps = shape[1]
        stride = self.seq_len - overlap_frame_num 
        window_idx = 0 

        for t_idx in range(0, num_steps, stride):
            if t_idx == 0:

                curr_x = x_all[:, t_idx:t_idx+self.seq_len] # Random noise. 
                curr_x_start = x_start[:, t_idx:t_idx+self.seq_len] # BS X window_szie X D (3+9+24*3+22*6) 
               
                curr_x_cond = self.bps_encoder(ori_x_cond) # BS X 1 X 256 
                curr_x_cond = curr_x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 

                if contact_labels is None:
                    curr_window_contact_labels = None 
                else:
                    curr_window_contact_labels = contact_labels[:, t_idx:t_idx+self.seq_len] 

                if language_input is not None:
                    language_embedding = self.clip_encoder(language_input[window_idx]) # BS X d_model 
                else:
                    language_embedding = None 

                only_clean_cond_path = curr_x_start * (1. - cond_mask)

                x_pose_cond = curr_x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 

                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    
                    only_noise_cond = curr_x * cond_mask # BS X T X D
                    curr_x = only_clean_cond_path + only_noise_cond # BS X T X D

                    if guidance_fn is not None and i > 0 and i < 10: 
                        curr_x = self.p_sample_guided_reconstruction_guidance(curr_x, torch.full((b,), i, \
                                    device=device, dtype=torch.long), \
                                    curr_x_cond, language_embedding=language_embedding, \
                                    guidance_fn=guidance_fn, \
                                    rest_human_offsets=rest_human_offsets, \
                                    data_dict=data_dict, cond_mask=cond_mask, \
                                    contact_labels=curr_window_contact_labels, x_pose_cond=x_pose_cond)    
                    
                    else: # padding mask is not used now! 
                        curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), \
                                curr_x_cond, language_embedding=language_embedding)     
                   
                whole_sample_res = curr_x.clone() # BS X window_size X D (3+9+24*3+22*6)  
                window_idx += 1 
            else:
                curr_x = x_all[:, t_idx:t_idx+self.seq_len] # Random noise. 
                prev_sample_res = whole_sample_res[:, -overlap_frame_num:, :] # BS X 10 X D 
                curr_x_start_init = x_start[:, t_idx:t_idx+self.seq_len] # BS X window_szie X D (3+9+24*3+22*6) 
                concat_time_frame_idx = overlap_frame_num 
                if contact_labels is not None:
                    curr_window_contact_labels = contact_labels[:, t_idx:t_idx+self.seq_len] 
                
                if curr_x.shape[1] < self.seq_len: # The last window with a smaller size. Better to not use this code. 
                    break 

                global_human_normalized_jpos = prev_sample_res[:, :, 12:12+24*3].reshape(b, -1, 24, 3) # BS X 10 X J(24) X 3
                global_human_jpos = ds.de_normalize_jpos_min_max(global_human_normalized_jpos) # BS X 10 X J X 3 

                global_human_6d = prev_sample_res[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6) # BS X 10 X 22 X 6
                global_human_rot_mat = transforms.rotation_6d_to_matrix(global_human_6d)
                global_human_q = transforms.matrix_to_quaternion(global_human_rot_mat) # BS X 10 X 22 X 4 

                obj_normalized_x = prev_sample_res[:, :, :3] # BS X 10 X 3 
                obj_com_pos = ds.de_normalize_obj_pos_min_max(obj_normalized_x) # BS X 10 X 3 
                
                obj_rel_rot_mat = prev_sample_res[:, :, 3:3+9].reshape(b, -1, 3, 3) # BS X 10 X 3 X 3  
                
                ref_frame_rot_mat = data_dict['reference_obj_rot_mat'].to(obj_rel_rot_mat.device) # 1 X 1 X 3 X 3 
                obj_rot_mat = ds.rel_rot_to_seq(obj_rel_rot_mat, ref_frame_rot_mat) # wrd rest pose object geometry. 
                
                obj_q = transforms.matrix_to_quaternion(obj_rot_mat) # BS X 10 X 4 

                if self.input_first_human_pose:
                    new_glob_jpos, new_glob_q, new_obj_com_pos, new_obj_q = \
                    rotate_at_frame_w_obj(global_human_jpos.data.cpu().numpy(), global_human_q.data.cpu().numpy(), \
                    obj_com_pos.data.cpu().numpy(), obj_q.data.cpu().numpy(), \
                    trans2joint.data.cpu().numpy(), ds.parents, n_past=1, floor_z=True, use_global_human=True)
                else:
                    new_glob_jpos, new_glob_q, new_obj_com_pos, new_obj_q = rotate_at_frame_w_obj_global( \
                    obj_com_pos.data.cpu().numpy(), obj_q.data.cpu().numpy(), ds.parents, n_past=1, floor_z=True, \
                    global_q=global_human_q.data.cpu().numpy(), global_x=global_human_jpos.data.cpu().numpy(), use_global=True) 

                new_glob_jpos = torch.from_numpy(new_glob_jpos).float().to(prev_sample_res.device)
                new_glob_q = torch.from_numpy(new_glob_q).float().to(prev_sample_res.device) 
                new_obj_com_pos = torch.from_numpy(new_obj_com_pos).float().to(prev_sample_res.device)
                new_obj_q = torch.from_numpy(new_obj_q).float().to(prev_sample_res.device) # wrd rest pose's rotation. 

                global_human_root_jpos = new_glob_jpos[:, :, 0, :].clone() # BS X T X 3
                global_human_root_trans = global_human_root_jpos + trans2joint[:, None, :].to(global_human_root_jpos.device) # BS X T X 3 

                move_to_zero_trans = global_human_root_trans[:, 0:1, :].clone() # Move the first frame's root joint x, y to 0,  BS X 1 X 3
                move_to_zero_trans[:, :, 2] = 0 # BS X 1 X 3 

                global_human_root_trans -= move_to_zero_trans 
                global_human_root_jpos -= move_to_zero_trans 
                new_glob_jpos -= move_to_zero_trans[:, :, None, :] 
                new_obj_com_pos = new_obj_com_pos - move_to_zero_trans # BS X T X 3 

                new_glob_rot_mat = transforms.quaternion_to_matrix(new_glob_q) # BS X T X J X 3 X 3 
                new_glob_rot_6d = transforms.matrix_to_rotation_6d(new_glob_rot_mat) # BS X T X J X 6 

                new_obj_rot_mat = transforms.quaternion_to_matrix(new_obj_q) # BS X T X 3 X 3 
                cano_rot_mat = torch.matmul(new_glob_rot_mat[:, 0, 0, :, :], \
                            global_human_rot_mat[:, 0, 0, :, :].transpose(1, 2)) # BS X 3 X 3 
               
                curr_end_frame_init = curr_x_start_init.clone() # BS X W X D (3+9+24*3+22*6)
                curr_end_obj_com_pos = ds.de_normalize_obj_pos_min_max(curr_end_frame_init[:, :, :3]) # BS X W X 3 
                curr_end_obj_com_pos = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, \
                                    curr_end_obj_com_pos.shape[1], 1, 1), \
                                    curr_end_obj_com_pos[:, :, :, None]) # BS X W X 3 X 1 
                curr_end_obj_com_pos = curr_end_obj_com_pos.squeeze(-1) # BS X W X 3
                curr_end_obj_com_pos -= move_to_zero_trans # BS X W X 3 

                curr_end_frame = torch.zeros_like(curr_end_frame_init) # BS X W X D

                curr_end_frame[:, :, :3] = ds.normalize_obj_pos_min_max(curr_end_obj_com_pos)

                curr_obj_bps_list = []
                new_obj_com_pos_list = []
                for bs_idx in range(b):
                    obj_rest_verts, obj_mesh_faces = ds.load_rest_pose_object_geometry(object_names[bs_idx])
                    obj_rest_verts = torch.from_numpy(obj_rest_verts).to(new_obj_rot_mat.device)
                    obj_verts = ds.load_object_geometry_w_rest_geo(new_obj_rot_mat[bs_idx], \
                        new_obj_com_pos[bs_idx], obj_rest_verts.float())
                
                    center_verts = obj_verts.mean(dim=1) # 10 X 3 

                    object_bps = ds.compute_object_geo_bps(obj_verts[0:1].cpu(), center_verts[0:1].cpu()) # 1 X 1024 X 3 

                    curr_obj_bps_list.append(object_bps) 
                    new_obj_com_pos_list.append(center_verts) 

                curr_obj_bps = torch.stack(curr_obj_bps_list)[:, None, :, :].cuda() # BS X 1 X 1024 X 3 
                curr_obj_com_pos = torch.stack(new_obj_com_pos_list).cuda() # BS X 10 X 3 

                curr_x_cond = self.bps_encoder(curr_obj_bps.reshape(b, 1, -1)) # BS X 1 X 256 
                curr_x_cond = curr_x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 

                curr_normalized_obj_com_pos = ds.normalize_obj_pos_min_max(curr_obj_com_pos) # BS X 10 X 3 
                curr_normalized_global_jpos = ds.normalize_jpos_min_max(new_glob_jpos) # BS X T X J X 3 
                curr_rel_rot_mat = ds.prep_rel_obj_rot_mat_w_reference_mat(new_obj_rot_mat, new_obj_rot_mat[:, 0:1]) # BS X T X 3 X 3 
               
                cano_prev_sample_res = torch.cat((curr_normalized_obj_com_pos, curr_rel_rot_mat.reshape(b, -1, 9), \
                                curr_normalized_global_jpos.reshape(b, -1, 24*3), new_glob_rot_6d.reshape(b, -1, 22*6)), dim=-1)
                
                if self.use_object_keypoints:
                    cano_prev_sample_res = torch.cat((cano_prev_sample_res, prev_sample_res[:, :, -4:]), dim=-1)

                curr_start_frame = cano_prev_sample_res[:, 0:1].clone() # BS X 1 X D 

                if input_waypoints:
                    curr_x_start = torch.cat((curr_start_frame, curr_end_frame[:, 1:, :]), dim=1) 
                else:
                    curr_x_start = torch.cat((curr_start_frame, torch.zeros(b, self.seq_len-2, \
                                curr_end_frame.shape[-1]).to(curr_end_frame.device), curr_end_frame[:, -1:, :]), dim=1) 
                
                only_clean_cond_path = curr_x_start * (1. - cond_mask)

                x_pose_cond = curr_x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 

                if language_input is not None:
                    language_embedding = self.clip_encoder(language_input[window_idx]) # BS X d_model 
                else:
                    language_embedding = None 

                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    only_noise_cond = curr_x * cond_mask # BS X T X D
                    curr_x = only_clean_cond_path + only_noise_cond # BS X T X D

                    if guidance_fn is not None and i > 0 and i < 10: 
                        curr_x = self.p_sample_guided_reconstruction_guidance(curr_x, torch.full((b,), i, device=device, dtype=torch.long), \
                                    curr_x_cond, language_embedding=language_embedding, \
                                    guidance_fn=guidance_fn, \
                                    rest_human_offsets=rest_human_offsets, \
                                    data_dict=data_dict, cond_mask=cond_mask, \
                                    prev_window_cano_rot_mat=cano_rot_mat, \
                                    prev_window_init_root_trans=global_human_jpos[:, 0:1, 0, :], \
                                    contact_labels=curr_window_contact_labels, \
                                    curr_window_ref_obj_rot_mat=new_obj_rot_mat[:, 0:1, :, :], x_pose_cond=x_pose_cond)     
                    else:
                        curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), curr_x_cond, \
                                           language_embedding=language_embedding)    
                  
                    if i > 0:
                        prev_conditions = torch.cat((cano_prev_sample_res, torch.zeros(b, \
                                        self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)
                        x_w_conditions = prev_conditions 
                        prev_condition_mask = torch.ones(b, cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)
                        prev_condition_mask = torch.cat((prev_condition_mask, \
                                torch.zeros(b, self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)

                        curr_x = prev_condition_mask * x_w_conditions + (1 - prev_condition_mask) * curr_x 

                prev_com_pos = cano_prev_sample_res[:, :, :3]
                prev_obj_rot_mat = cano_prev_sample_res[:, :, 3:3+9].reshape(b, -1, 3, 3)
                prev_human_jpos = cano_prev_sample_res[:, :, 12:12+24*3].reshape(b, -1, 24, 3)
                prev_human_rot_6d = cano_prev_sample_res[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6)

                curr_x_obj_com_pos = curr_x[:, :, :3] # 1 X w X 3 
                curr_x_obj_rot_mat = curr_x[:, :, 3:3+9].reshape(b, -1, 3, 3) 
                curr_x_human_jpos = curr_x[:, :, 12:12+24*3].reshape(b, -1, 24, 3)
                curr_x_human_rot_6d = curr_x[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6)

                curr_x_obj_com_pos, curr_x_obj_rot_mat, curr_x_human_jpos, curr_x_human_rot_6d = \
                    interpolate_transition(prev_com_pos, prev_obj_rot_mat, prev_human_jpos, prev_human_rot_6d, \
                                    curr_x_obj_com_pos, curr_x_obj_rot_mat, curr_x_human_jpos, curr_x_human_rot_6d)

                if self.use_object_keypoints:
                    curr_x = torch.cat((curr_x_obj_com_pos, curr_x_obj_rot_mat.reshape(b, -1, 9), \
                                        curr_x_human_jpos.reshape(b, -1, 24*3), \
                                        curr_x_human_rot_6d.reshape(b, -1, 22*6), \
                                        curr_x[:, :, -4:]), dim=-1)
                else:
                    curr_x = torch.cat((curr_x_obj_com_pos, curr_x_obj_rot_mat.reshape(b, -1, 9), \
                                        curr_x_human_jpos.reshape(b, -1, 24*3), curr_x_human_rot_6d.reshape(b, -1, 22*6)), dim=-1)

                if self.use_object_keypoints:
                    tmp_curr_x = curr_x[:, :, :-4] 
                else:
                    tmp_curr_x = curr_x.clone() 

                converted_obj_com_pos, converted_obj_rot_mat, converted_human_jpos, converted_rot_6d = \
                    self.apply_rotation_to_data(ds, cano_rot_mat, new_obj_rot_mat, tmp_curr_x)
                converted_obj_rel_rot_mat = ds.prep_rel_obj_rot_mat_w_reference_mat(converted_obj_rot_mat, \
                                    ref_frame_rot_mat) 

                aligned_human_trans = global_human_jpos[:, 0:1, 0, :] - converted_human_jpos[:, 0:1, 0, :]
                converted_human_jpos += aligned_human_trans[:, :, None, :]

                converted_obj_com_pos += aligned_human_trans 
                converted_normalized_obj_com_pos = ds.normalize_obj_pos_min_max(converted_obj_com_pos)

                converted_normalized_human_jpos = ds.normalize_jpos_min_max(converted_human_jpos) 

                converted_curr_x = torch.cat((converted_normalized_obj_com_pos.reshape(b, self.seq_len, -1), \
                            converted_obj_rel_rot_mat.reshape(b, self.seq_len, -1), \
                            converted_normalized_human_jpos.reshape(b, self.seq_len, -1), \
                            converted_rot_6d.reshape(b, self.seq_len, -1)), dim=-1) 
                
                if self.use_object_keypoints:
                    converted_curr_x = torch.cat((converted_curr_x, curr_x[:, :, -4:]), dim=-1) 

                whole_sample_res = torch.cat((whole_sample_res[:, :-concat_time_frame_idx], converted_curr_x), dim=1) 

                window_idx += 1 

        return whole_sample_res # BS X T X D (3+9+24*3+22*6)

    def apply_rotation_to_data(self, ds, cano_rot_mat, new_obj_rot_mat, curr_x):
        bs, timesteps, _ = curr_x.shape 

        pred_obj_normalized_com_pos = curr_x[:, :, :3]
        pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_obj_normalized_com_pos) 
        pred_obj_rel_rot_mat = curr_x[:, :, 3:12].reshape(bs, timesteps, 3, 3)
        pred_obj_rot_mat = ds.rel_rot_to_seq(pred_obj_rel_rot_mat, new_obj_rot_mat)
        pred_human_normalized_jpos = curr_x[:, :, 12:12+24*3]
        pred_human_jpos = ds.de_normalize_jpos_min_max(pred_human_normalized_jpos.reshape(bs, timesteps, 24, 3))
        pred_human_rot_6d = curr_x[:, :, 12+24*3:]

        pred_human_rot_mat = transforms.rotation_6d_to_matrix(pred_human_rot_6d.reshape(bs, timesteps, 22, 6))

        converted_obj_com_pos = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, timesteps, \
                            1, 1).transpose(2, 3), \
                            pred_obj_com_pos[:, :, :, None]).squeeze(-1)

        converted_obj_rot_mat = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, timesteps, \
                            1, 1).transpose(2, 3), pred_obj_rot_mat)
     
        converted_human_jpos = torch.matmul(cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 24, 1, 1).transpose(3, 4), \
                    pred_human_jpos[:, :, :, :, None]).squeeze(-1)
        converted_rot_mat = torch.matmul(cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 22, 1, 1).transpose(3, 4), \
                    pred_human_rot_mat)

        converted_rot_6d = transforms.matrix_to_rotation_6d(converted_rot_mat) 

        return converted_obj_com_pos, converted_obj_rot_mat, converted_human_jpos, converted_rot_6d 

    def sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
            language_input=None, contact_labels=None, rest_human_offsets=None, \
            data_dict=None, guidance_fn=None, opt_fn=None, \
            cond_mask_path=None):
        self.denoise_fn.eval() 
        self.bps_encoder.eval()
        self.clip_encoder.eval()

        if ori_x_cond is not None:
            x_cond = self.bps_encoder(ori_x_cond)
            x_cond = x_cond.repeat(1, self.seq_len, 1)
        else:
            x_cond = None 
       
        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 
        
        if guidance_fn is not None:
            sample_res = self.p_sample_loop_guided(x_start.shape, x_start, \
                    x_cond, guidance_fn, opt_fn=opt_fn, \
                    language_embedding=language_embedding, rest_human_offsets=rest_human_offsets, \
                    data_dict=data_dict, contact_labels=contact_labels, \
                    cond_mask=cond_mask, padding_mask=padding_mask, cond_mask_path=cond_mask_path)
        else:
            sample_res = self.p_sample_loop(x_start.shape, x_start, x_cond, \
                    language_embedding=language_embedding, padding_mask=padding_mask, \
                    cond_mask_path=cond_mask_path)

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.clip_encoder.train() 

        return sample_res  

    def sample_sliding_window_w_canonical(self, ds, object_names, trans2joint, \
                                x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
                                overlap_frame_num=1, input_waypoints=False, \
                                contact_labels=None, language_input=None, \
                                rest_human_offsets=None, data_dict=None, \
                                guidance_fn=None, opt_fn=None):
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.clip_encoder.eval()


        sample_res = self.p_sample_loop_sliding_window_w_canonical(ds, object_names, \
                trans2joint, x_start, \
                ori_x_cond, cond_mask=cond_mask, padding_mask=padding_mask, \
                overlap_frame_num=overlap_frame_num, input_waypoints=input_waypoints, \
                contact_labels=contact_labels, language_input=language_input, \
                rest_human_offsets=rest_human_offsets, data_dict=data_dict, \
                guidance_fn=guidance_fn, opt_fn=opt_fn)

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.clip_encoder.train() 
      
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, x_cond, t, language_embedding=None, noise=None, \
        padding_mask=None, rest_human_offsets=None, data_dict=None, ds=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.denoise_fn(x, t, x_cond, language_embedding=language_embedding, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss_reshaped = loss.reshape(x_start.shape[0], self.seq_len, -1) 

        loss_object = loss_reshaped[:, :, :12]

        if loss_reshaped.shape[-1] == 12:
            loss_human = torch.zeros(1) 
        else:
            loss_human = loss_reshaped[:, :, 12:]

        if self.use_object_keypoints:
            hand_idx = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]

            bs, num_steps, _ = model_out.shape 

            gt_global_jpos = target[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            gt_global_jpos = ds.de_normalize_jpos_min_max(gt_global_jpos)
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            global_jpos = model_out[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            global_jpos = ds.de_normalize_jpos_min_max(global_jpos)

            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, num_steps, 1, 1).cuda()
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(bs*num_steps, 24, 3)
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(bs*num_steps, 24, 3)[:, 0, :]
            
            global_joint_rot_6d = model_out[:, :, 12+24*3:12+24*3+22*6].reshape(bs, num_steps, 22, 6)
            global_joint_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d)
            local_joint_rot_mat = quat_ik_torch(global_joint_rot_mat.reshape(-1, 22, 3, 3))
            _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts.reshape(bs, num_steps, 24, 3)

            pred_global_hand_jpos = human_jnts[:, :, hand_idx, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :]

            fk_hand_loss = self.loss_fn(
                pred_global_hand_jpos, gt_global_hand_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            fk_hand_loss = reduce(fk_hand_loss, "b ... -> b (...)", "mean")

            fk_hand_loss = fk_hand_loss * extract(self.p2_loss_weight, t, fk_hand_loss.shape)

            fk_foot_loss = self.loss_fn(
                pred_global_foot_jpos, gt_global_foot_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            fk_foot_loss = reduce(fk_foot_loss, "b ... -> b (...)", "mean")

            fk_foot_loss = fk_foot_loss * extract(self.p2_loss_weight, t, fk_foot_loss.shape)

            fk_loss = fk_hand_loss + fk_foot_loss 

            model_semantic_contact = model_out[:, :, -4:]
            foot_loss = self.loss_fn(
                model_semantic_contact, target[:, :, -4:], reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None]
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

            rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].to(model_out.device)
            gt_seq_obj_kpts = data_dict['ori_obj_keypoints'].to(model_out.device)

            pred_obj_rel_rot_mat = model_out[:, :, 3:3+9].reshape(bs, num_steps, 3, 3)
            ref_obj_rot_mat = data_dict['reference_obj_rot_mat'].to(model_out.device)
            ref_obj_rot_mat = ref_obj_rot_mat.repeat(1, pred_obj_rel_rot_mat.shape[1], 1, 1)
            pred_obj_rot_mat = torch.matmul(pred_obj_rel_rot_mat, ref_obj_rot_mat.to(pred_obj_rel_rot_mat.device))

            pred_normalized_obj_com_pos = model_out[:, :, :3]
            pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_com_pos)

            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, rest_pose_obj_kpts.shape[1], 1, 1), \
                    rest_pose_obj_kpts[:, None, :, :, None].repeat(1, num_steps, 1, 1, 1)) + pred_obj_com_pos[:, :, None, :, None]

            pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1)

            loss_obj_pts = self.loss_fn(
                pred_seq_obj_kpts, gt_seq_obj_kpts, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            loss_obj_pts = reduce(loss_obj_pts, "b ... -> b (...)", "mean")

            loss_obj_pts = loss_obj_pts * extract(self.p2_loss_weight, t, loss_obj_pts.shape)
           
            return loss.mean(), loss_object.mean(), loss_human.mean(), \
                foot_loss.mean(), fk_loss.mean(), loss_obj_pts.mean()   
        
        return loss.mean(), loss_object.mean(), loss_human.mean() 
    
    def p_losses_ca(self, x_start, bps_embed, cond_mask, t, language_embedding=None, noise=None, \
        padding_mask=None, rest_human_offsets=None, data_dict=None, ds=None, cond_mask_path=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_path = x_start * (1. - cond_mask_path) + noise * cond_mask_path 

        path_output = self.denoise_fn_path(x_path, t, bps_embed, language_embedding=language_embedding, padding_mask=padding_mask)
        
        if self.objective == 'pred_noise':
            target_path = torch.cat([noise[..., :3], noise[..., 12:15]], dim=-1)
        elif self.objective == 'pred_x0':
            target_path = torch.cat([x_start[..., :3], x_start[..., 12:15]], dim=-1)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        if padding_mask is not None:
            loss_path = self.loss_fn(path_output, target_path, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss_path = self.loss_fn(path_output, target_path, reduction = 'none')

        loss_path = reduce(loss_path, 'b ... -> b (...)', 'mean')
        loss_path = loss_path * extract(self.p2_loss_weight, t, loss_path.shape)
        loss_path = loss_path.mean()
        
        only_clean_cond = x_start * (1. - cond_mask)
        only_noise_cond = x * cond_mask

        x = only_clean_cond + only_noise_cond

        model_out = self.denoise_fn(x, t, bps_embed, language_embedding=language_embedding, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss_reshaped = loss.reshape(x_start.shape[0], self.seq_len, -1) 

        loss_object = loss_reshaped[:, :, :12]

        if loss_reshaped.shape[-1] == 12:
            loss_human = torch.zeros(1) 
        else:
            loss_human = loss_reshaped[:, :, 12:]

        if self.use_object_keypoints:
            hand_idx = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]

            bs, num_steps, _ = model_out.shape 

            gt_global_jpos = target[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            gt_global_jpos = ds.de_normalize_jpos_min_max(gt_global_jpos)
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            global_jpos = model_out[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            global_jpos = ds.de_normalize_jpos_min_max(global_jpos)

            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, num_steps, 1, 1).cuda()
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(bs*num_steps, 24, 3)
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(bs*num_steps, 24, 3)[:, 0, :]
            
            global_joint_rot_6d = model_out[:, :, 12+24*3:12+24*3+22*6].reshape(bs, num_steps, 22, 6)
            global_joint_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d)
            local_joint_rot_mat = quat_ik_torch(global_joint_rot_mat.reshape(-1, 22, 3, 3))
            _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts.reshape(bs, num_steps, 24, 3)

            pred_global_hand_jpos = human_jnts[:, :, hand_idx, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :]

            fk_hand_loss = self.loss_fn(
                pred_global_hand_jpos, gt_global_hand_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            fk_hand_loss = reduce(fk_hand_loss, "b ... -> b (...)", "mean")

            fk_hand_loss = fk_hand_loss * extract(self.p2_loss_weight, t, fk_hand_loss.shape)

            fk_foot_loss = self.loss_fn(
                pred_global_foot_jpos, gt_global_foot_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            fk_foot_loss = reduce(fk_foot_loss, "b ... -> b (...)", "mean")

            fk_foot_loss = fk_foot_loss * extract(self.p2_loss_weight, t, fk_foot_loss.shape)

            fk_loss = fk_hand_loss + fk_foot_loss 

            model_semantic_contact = model_out[:, :, -4:]
            foot_loss = self.loss_fn(
                model_semantic_contact, target[:, :, -4:], reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None]
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

            rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].to(model_out.device)
            gt_seq_obj_kpts = data_dict['ori_obj_keypoints'].to(model_out.device)

            pred_obj_rel_rot_mat = model_out[:, :, 3:3+9].reshape(bs, num_steps, 3, 3)
            ref_obj_rot_mat = data_dict['reference_obj_rot_mat'].to(model_out.device)
            ref_obj_rot_mat = ref_obj_rot_mat.repeat(1, pred_obj_rel_rot_mat.shape[1], 1, 1)
            pred_obj_rot_mat = torch.matmul(pred_obj_rel_rot_mat, ref_obj_rot_mat.to(pred_obj_rel_rot_mat.device))

            pred_normalized_obj_com_pos = model_out[:, :, :3]
            pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_com_pos)

            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, rest_pose_obj_kpts.shape[1], 1, 1), \
                    rest_pose_obj_kpts[:, None, :, :, None].repeat(1, num_steps, 1, 1, 1)) + pred_obj_com_pos[:, :, None, :, None]

            pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1)

            loss_obj_pts = self.loss_fn(
                pred_seq_obj_kpts, gt_seq_obj_kpts, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None]
            loss_obj_pts = reduce(loss_obj_pts, "b ... -> b (...)", "mean")

            loss_obj_pts = loss_obj_pts * extract(self.p2_loss_weight, t, loss_obj_pts.shape)

           
            return (loss.mean(), loss_object.mean(), loss_human.mean(), \
                foot_loss.mean(), fk_loss.mean(), loss_obj_pts.mean(), loss_path), \
                (pred_global_hand_jpos, gt_global_hand_jpos, \
                pred_seq_obj_kpts, gt_seq_obj_kpts, pred_global_foot_jpos, gt_global_foot_jpos)
        
        return loss.mean(), loss_object.mean(), loss_human.mean(), loss_path

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
        language_input=None, contact_labels=None, rest_human_offsets=None, data_dict=None, ds=None, \
        cond_mask_path=None): 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        if ori_x_cond is not None:
            x_cond = self.bps_encoder(ori_x_cond)
            x_cond = x_cond.repeat(1, self.seq_len, 1)
        else:
            x_cond = None 

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 

        if self.use_object_keypoints:
            (curr_loss, curr_loss_obj, curr_loss_human, curr_loss_feet, curr_loss_fk, curr_loss_obj_pts, curr_loss_path), pred_batch = \
                        self.p_losses_ca(x_start, x_cond, cond_mask, t, \
                        language_embedding=language_embedding, padding_mask=padding_mask, \
                        rest_human_offsets=rest_human_offsets, data_dict=data_dict, ds=ds, cond_mask_path=cond_mask_path)  

            return (curr_loss, curr_loss_obj, curr_loss_human, curr_loss_feet, curr_loss_fk, curr_loss_obj_pts, curr_loss_path), (pred_batch)
        else:
            curr_loss, curr_loss_obj, curr_loss_human = self.p_losses(x_start, x_cond, t, \
                        language_embedding=language_embedding, padding_mask=padding_mask, \
                        rest_human_offsets=rest_human_offsets, data_dict=data_dict, ds=ds)  

            return curr_loss, curr_loss_obj, curr_loss_human 
        
