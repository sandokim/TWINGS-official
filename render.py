#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import PILtoTorch
###
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, resol=1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    PSNR = []
    SSIM = []
    LPIPS = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        
        PSNR.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        SSIM.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))
        LPIPS.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

    psnr_mean = torch.tensor(PSNR).mean().item()
    ssim_mean = torch.tensor(SSIM).mean().item()
    lpips_mean = torch.tensor(LPIPS).mean().item()

    print('PSNR : {:>12.7f}'.format(psnr_mean))
    print('SSIM : {:>12.7f}'.format(ssim_mean))
    print('LPIPS : {:>12.7f}'.format(lpips_mean))

    with open(os.path.join(model_path, 'metrics_{0}.txt'.format(iteration)), 'w') as f:
        f.write('PSNR : {:>12.7f}\n'.format(psnr_mean))
        f.write('SSIM : {:>12.7f}\n'.format(ssim_mean))
        f.write('LPIPS : {:>12.7f}\n'.format(lpips_mean))
        
        
def render_pseudo_set(model_path, name, iteration, views, gaussians, pipeline, background, resol=1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, str(idx) + '.png'))
  
  
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
             
        # render_pseudo_set(dataset.model_path, "pseudo", scene.loaded_iter, scene.getPseudoCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
