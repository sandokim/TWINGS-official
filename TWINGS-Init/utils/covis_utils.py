import os
import torch
import numpy as np
from typing import Dict, Any
from PIL import Image
import cv2
from utils.bg_utils import ensure_white_background
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images as load_images_for_mast3r

def covisibility_map_generation_w_all_training_views(
    extrinsics: Dict[int, Any],
    image_dir: str,
    SCENE_DIR: str,
    resize_target_long_side: int,
    ref_image_id: int,
    n_views: int,
    model: Any,
    device: torch.device,
    data_type: str = "nerf_llff_data",
    white_background: bool = False,
):
    image_id_to_path = {
        image_id: os.path.join(image_dir, extr.name)
        for image_id, extr in extrinsics.items()
    }
    target_ids = [i for i in extrinsics if i != ref_image_id]

    covisibility_map_path = os.path.join(SCENE_DIR, "covisibility_map", f"{n_views}_views")
    os.makedirs(covisibility_map_path, exist_ok=True)
    covisibility_map = None

    for tgt_id in target_ids:
        ref_path = image_id_to_path[ref_image_id]
        tgt_path = image_id_to_path[tgt_id]

        if white_background:
            wb_cache = os.path.join(SCENE_DIR, 'train_wb')
            ref_path = ensure_white_background(ref_path, cache_dir=wb_cache)
            tgt_path = ensure_white_background(tgt_path, cache_dir=wb_cache)

        images_pair = load_images_for_mast3r([ref_path, tgt_path], size=resize_target_long_side)

        with torch.no_grad():
            output = inference([tuple(images_pair)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=1, device=device, dist='dot', block_size=2**13) # subsample_or_initxy1=1 -> per pixel dense prediction for covis map

        H0, W0 = view1['true_shape'][0]
        valid_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                    (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
        H1, W1 = view2['true_shape'][0]
        valid_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                    (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
        valid = valid_im0 & valid_im1
        matches_im0 = matches_im0[valid]
        if covisibility_map is None:
            covisibility_map = np.zeros((int(H0), int(W0)), dtype=np.int32)
        if matches_im0.shape[0] > 0:
            xi = matches_im0[:, 0].astype(np.int32)
            yi = matches_im0[:, 1].astype(np.int32)
            covisibility_map[yi, xi] += 1

    if covisibility_map is None:
        # fallback empty
        covisibility_map = np.zeros((int(H0), int(W0)), dtype=np.int32)

    ref_img_path = image_id_to_path[ref_image_id]
    ori_w, ori_h = Image.open(ref_img_path).size
    if data_type == "nerf_llff_data":
        resized_W = round(ori_w / 8)
        resized_H = round(ori_h / 8)
    elif data_type == "mipnerf360":
        resized_W = round(ori_w / 4)
        resized_H = round(ori_h / 4)
    elif data_type == "DTU":
        resized_W = round(ori_w / 4)
        resized_H = round(ori_h / 4)
    elif data_type == "blender":
        resized_W = round(ori_w / 2)
        resized_H = round(ori_h / 2)

    cov_resized = cv2.resize(covisibility_map, (resized_W, resized_H), interpolation=cv2.INTER_NEAREST)
    np.save(os.path.join(covisibility_map_path, f"{ref_image_id}.npy"), cov_resized) # (resized_H, resized_W)

    max_val = float(cov_resized.max())
    viz = (cov_resized / max_val * 255).astype(np.uint8) if max_val > 0 else cov_resized.astype(np.uint8)
    cv2.imwrite(os.path.join(covisibility_map_path, f"{ref_image_id}.png"), viz)
    
    print(f"Covisibility map generated for {ref_image_id}")
    
    return 0