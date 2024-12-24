import os
import torch

from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from models.dit3d import DiT3D_models
from utils.misc import Evaluator
from models.dit3d_window_attn import DiT3D_models_WindAttn
from data_module.data_module import PointCloudPartSegWhithSketch, RandamPartialPointCloudWhithSketch, Sketch_our

import numpy as np
import cv2
import timm

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y,f, clip_denoised: bool, return_pred_xstart: bool):
        
        model_output = denoise_fn(data, t, y, f)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, f,clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, f=f,clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y,f,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,f=f,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def classifier_p_mean_variance(self, denoise_fn, data, t, y,f, clip_denoised: bool, return_pred_xstart: bool):
        emp_f = torch.zeros(f.shape, device= f.device)
        model_sketch = denoise_fn(data, t, y, emp_f)

        emp_y = torch.zeros(y.shape, device= y.device)
        model_points = denoise_fn(data, t, emp_y, f)

        
        model_none = denoise_fn(data, t, emp_y, emp_f)
        # model_full = denoise_fn(data, t, y, f)

        w_y = 3
        w_f = 4

        model_output = (w_y*model_sketch - w_y * model_none) +((1+w_f)*model_points - w_f*model_none)
        # model_output = model_full+ w_y *(model_full - model_sketch ) +w_f*(model_full -model_points )
        # model_output = ((1 + w_f) * model_points - w_f * model_none)
        # model_output = model_none

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance


    def classifier_p_sample(self, denoise_fn, data, t, noise_fn, y, f,clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.classifier_p_mean_variance(denoise_fn, data=data, t=t, y=y, f=f,clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def classifier_p_sample_loop(self, denoise_fn, shape, device, y,f,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.classifier_p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,f=f,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t
    
    def ddim_sample(self, denoise_fn, x_t, t, t_prev, y,f, eta=0.0):
        alpha_t = self._extract(self.alphas_cumprod.to(t.device), t, x_t.shape)
        alpha_t_prev = self._extract(self.alphas_cumprod.to(t.device), t_prev, x_t.shape)
        epsilon_theta_t = denoise_fn(x_t, t,y,f)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )

        return x_t_minus_one
    
    def ddim_sample_loop(self, denoise_fn, shape, device, y, f,noise_fn=torch.randn, eta=0.0, clip_denoised=True):
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        
        steps = 50
        a = self.num_timesteps // steps
        ranges = np.asarray(list(range(0, self.num_timesteps,a)))
        ranges_prev = np.concatenate([[0],ranges[:-1]])

        for i in reversed(range(0, len(ranges))):
            t = torch.full((shape[0],), ranges[i], device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), ranges_prev[i], device=device, dtype=torch.long)
            img_t = self.ddim_sample(denoise_fn, img_t, t, t_prev, y,f, eta=eta)
            # if clip_denoised:
            #     img_t = torch.clamp(img_t, -1., 1.)
        return img_t

    def reconstruct(self, x0, t, y, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        
        # DiT-3d
        self.model = DiT3D_models[args.model_type](input_size=args.voxel_size, num_classes=args.num_classes)

        # self.model = DiT3D_models_WindAttn[args.model_type](
        #                                            input_size=args.voxel_size, 
        #                                            window_size=4, 
        #                                            window_block_indexes=[0,3,6,9], 
        #                                            num_classes=args.num_classes
        #                                         )

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y,f):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y,f)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, y=None):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, f,noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, f=f,noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)
    
    def classifier_gen_samples(self, shape, device, y, f,noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.classifier_p_sample_loop(self._denoise, shape=shape, device=device, y=y, f=f,noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)
    
    def gen_samples_ddim(self, shape, device, y, f,noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.ddim_sample_loop(self._denoise, shape=shape, device=device, y=y, f=f,noise_fn=noise_fn,
                                            clip_denoised=clip_denoised)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y,f=f, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    # tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
    #     categories=[category], split='train',
    #     tr_sample_size=npoints,
    #     te_sample_size=npoints,
    #     scale=1.,
    #     normalize_per_shape=False,
    #     normalize_std_per_axis=False,
    #     random_subsample=True, use_mask = use_mask)
    # te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
    #     categories=[category], split='val',
    #     tr_sample_size=npoints,
    #     te_sample_size=npoints,
    #     scale=1.,
    #     normalize_per_shape=False,
    #     normalize_std_per_axis=False,
    #     all_points_mean=tr_dataset.all_points_mean,
    #     all_points_std=tr_dataset.all_points_std,
    #     use_mask=use_mask
    # )
    # tr_dataset = PointCloudPartSegWhithSketch(root=dataroot,
    #                                           split='val',
    #                                         categories=['chair'],
    #                                         get_images = ['edit_sketch'],
    #                                         )
    # te_dataset = PointCloudPartSegWhithSketch(root=dataroot,
    #                                           split='val',
    #                                         categories=['chair'],
    #                                         get_images = ['edit_sketch'],
    #       
    #                                   )
    path_root = '../cross_sketch_all_Mesh/output_our/04379243'
    tr_dataset = Sketch_our(root=path_root,
                                            )
    te_dataset = Sketch_our(root=path_root,
                                            )

    
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=True, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def generate_eval(model, opt, gpu, outf_syn, evaluator):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    _, test_dataloader, _, test_sampler = get_dataloader(opt, test_dataset, test_dataset)


    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=device)
    
    with torch.no_grad():

        samples = []

        feature_extractor = timm.create_model(
            'vit_huge_patch14_224_clip_laion2b',
            pretrained=True).cuda()

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            # x = data['test_points'].transpose(1,2)
            m, s = data['shift'].float(), data['scale'].float()

            x = data['sample_points'].transpose(1,2)
            
            f = data['fix_points'].transpose(1,2)
            print(f.shape)
            y = data['edit_sketch']
            image_features = feature_extractor.forward_features(y.cuda())
            # y = image_features[:, 0, :]
            y = image_features
            

            # gen = model.gen_samples(x.shape, gpu, new_y_chain(gpu,y.shape[0],opt.num_classes), clip_denoised=False).detach().cpu()
            # gen = model.gen_samples(x.shape, gpu, y,f, clip_denoised=False).detach().cpu()
            gen = model.classifier_gen_samples(x.shape, gpu, y,f, clip_denoised=False).detach().cpu()
            
            # gen = model.gen_samples_ddim(x.shape, gpu, y,f, clip_denoised=False).detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()
            f = f.transpose(1,2).contiguous()

            gen = gen * s + m
            x = x * s + m
            f = f * s + m
            samples.append(gen.to(gpu).contiguous())

            visualize_pointcloud_batch(os.path.join(outf_syn, f'{i}_{gpu}.png'), gen, None,
                                       None, None)
            
            # Compute metrics
            results = compute_all_metrics(gen, x, opt.bs)
            results = {k: (v.cpu().detach().item()
                        if not isinstance(v, float) else v) for k, v in results.items()}

            jsd = JSD(gen.numpy(), x.numpy())

            out_path = os.path.join(outf_syn,'test/')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            ## y画像の保存
            for j in range(data['edit_sketch'].shape[0]):
                s = data['edit_sketch'][j]*255
                cv2.imwrite(os.path.join(outf_syn,'test/', f'{i}_{j}_y.png'), s.permute(1,2,0).cpu().numpy())
            ## xの点群をnpyファイルで一つ一つ保存
            for j in range(x.shape[0]):
                np.save(os.path.join(outf_syn,'test/', f'{i}_{j}_x.npy'), x[j].cpu().numpy())
            ## genの点群をnpyファイルで一つ一つ保存
            for j in range(gen.shape[0]):
                np.save(os.path.join(outf_syn,'test/', f'{i}_{j}_gen.npy'), gen[j].cpu().numpy())
            for j in range(f.shape[0]):
                np.save(os.path.join(outf_syn,'test/', f'{i}_{j}_f.npy'), f[j].cpu().numpy())

            evaluator.update(results, jsd)
            # if (i+1)%10 == 0:
            #     break   

        stats = evaluator.finalize_stats()

        samples = torch.cat(samples, dim=0)
        # samples_gather = concat_all_gather(samples)

        # torch.save(samples_gather, opt.eval_path)


    return stats


def main(opt):

    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        test(opt.gpu, opt, output_dir)


def test(gpu, opt, output_dir):

    logger = setup_logging(output_dir)


    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu,find_unused_parameters=True)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters())/1e6
        logger.info("Total_params = %s MB " % str(total_params))    # S4: 32.81 MB

    model.eval()

    evaluator = Evaluator(results_dir=output_dir)    

    with torch.no_grad():
        
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'])

        opt.eval_path = os.path.join(outf_syn, 'samples.pth')
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
        
        stats = generate_eval(model, opt, gpu, outf_syn, evaluator)

        if should_diag:
            logger.info(stats)
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit3d', help='experiment name (used for checkpointing and logging)')

    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=55)

    parser.add_argument('--bs', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)

    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-S/4")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--model', default='',required=True, help="path to model (to continue training)")

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    # parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
    #                     help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    
    '''eval'''

    parser.add_argument('--eval_path',
                        default='')

    parser.add_argument('--manualSeed', default=41, type=int, help='random seed')

    opt = parser.parse_args()


    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
