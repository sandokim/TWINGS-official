import numpy as np
from scipy.optimize import least_squares    
from typing import Dict, Tuple, Any
import cv2
import torch
import os
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images as load_images_for_mast3r
from dust3r.utils.image import rgb
n_viz = 180 # number of 2D-2D matching results to visualize

def feature_extract_and_point_tracking(
    extrinsics: Dict[int, Any],
    image_dir: str,
    SCENE_DIR: str,
    resize_target_long_side: int,
    ref_image_id: int,
    model,
    device: torch.device,
    white_background: bool = False,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    image_id_to_path = {
        image_id: os.path.join(image_dir, extr.name)
        for image_id, extr in extrinsics.items()
    }
    target_ids = [i for i in extrinsics if i != ref_image_id]

    point_tracks, pt_ref_to_track_id = {}, {}
    rgb_images = {}
    track_id_counter = 0
    
    for tgt_id in target_ids:
        ref_path = image_id_to_path[ref_image_id]
        tgt_path = image_id_to_path[tgt_id]

        images_pair = load_images_for_mast3r(
            [ref_path, tgt_path], size=resize_target_long_side) # coarse prediction 
        with torch.no_grad():
            output = inference([tuple(images_pair)], model, device, batch_size=1, verbose=False)
            
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, # subsample_or_initxy1=8 for TPS
                                                       device=device, dist='dot', block_size=2**13)

        # valid coordinate filtering
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3) # only valid coordinates within 3px from the image edges

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3) # only valid coordinates within 3px from the image edges

        # valid matching filtering
        valid_matches = valid_matches_im0 & valid_matches_im1 
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches] # matches_im0.shape = matches_im1.shape => (# of matches, 2) / here 2 is integer pixel xy coordinates
       
        num_matches = matches_im0.shape[0] 
        print(f'  - Found {num_matches} valid matches.')
        if num_matches == 0:
            print(f"[⚠] No matches found: ref {ref_image_id} ↔ tgt {tgt_id}")
            continue

        # RANSAC filtering (only perform if there are enough matches)
        inlier_matches_im0 = matches_im0
        inlier_matches_im1 = matches_im1
        inlier_num_matches = num_matches
        if num_matches >= 8:
            F, mask = cv2.findFundamentalMat(
                matches_im0, matches_im1, method=cv2.RANSAC, ransacReprojThreshold=0.1)
            if mask is not None:
                inlier_matches_im0 = matches_im0[mask.ravel() == 1]
                inlier_matches_im1 = matches_im1[mask.ravel() == 1]
                inlier_num_matches = inlier_matches_im0.shape[0]
                print(f"inlier_num_matches: {inlier_num_matches}")
            else:
                print("[Info] FundamentalMat failed or mask is None → RANSAC filtering skipped")

        
        matches_im0 = inlier_matches_im0
        matches_im1 = inlier_matches_im1
        num_matches = inlier_num_matches
        if inlier_num_matches == 0:
            print(f"[⚠] No inliers found: ref {ref_image_id} ↔ tgt {tgt_id}")
            continue
        
        # valid matches visualization
        # if num_matches > n_viz:
        #     match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        #     viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        #     image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        #     image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        #     viz_imgs = []
        #     for k, view in enumerate([view1, view2]):
        #         rgb_tensor = view['img'] * image_std + image_mean
        #         viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        #     H0_viz, W0_viz, H1_viz, W1_viz = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        #     img0 = np.pad(viz_imgs[0], ((0, max(H1_viz - H0_viz, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        #     img1 = np.pad(viz_imgs[1], ((0, max(H0_viz - H1_viz, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        #     img = np.concatenate((img0, img1), axis=1)
        #     pl.figure()
        #     pl.imshow(img)
        #     cmap = pl.get_cmap('jet')
        #     for k in range(n_viz):
        #         (x0, y0), (x1, y1) = viz_matches_im0[k].T, viz_matches_im1[k].T
        #         pl.plot([x0, x1 + W0_viz], [y0, y1], '-+', color=cmap(k / (n_viz - 1)), scalex=False, scaley=False)
        #     pl.show(block=True)
                    
        # valid matching based point tracking data structure formation
        '''
        point_tracks = {
            track_id: {
                view_idx (0): (x, y)
                view_idx (1): (x, y)
                view_idx (2): (x, y)
                ...
            }
        }
        pt_ref_to_track_id = {
            (view_x_0, view_y_0): track_id 0
            (view_x_1, view_y_1): track_id 1
            ...
        }
        '''  
        
        # RGB image saving
        if ref_image_id not in rgb_images:
            rgb_images[ref_image_id] = rgb(view1['img'], true_shape=view1['true_shape'][0]).squeeze(0)
        if tgt_id not in rgb_images:
            rgb_images[tgt_id] = rgb(view2['img'], true_shape=view2['true_shape'][0]).squeeze(0)
                
        # point track construction
        for pt0, pt1 in zip(matches_im0, matches_im1):
            pt0_key = tuple(pt0.tolist())
            if pt0_key not in pt_ref_to_track_id:
                pt_ref_to_track_id[pt0_key] = track_id_counter
                point_tracks[track_id_counter] = {ref_image_id: pt0_key}
                track_id_counter += 1
            tid = pt_ref_to_track_id[pt0_key]
            point_tracks[tid][tgt_id] = tuple(pt1.tolist())
            
    # covisibility map saving is handled by a separate function
    print(f"\n[✔] Total number of point tracks: {len(point_tracks)}")
    
    return point_tracks, rgb_images # sorted image paths assigned track_id based point tracks / view_idx (0) = cam0, view_idx (1) = cam1, view_idx (2) = cam2, view_idx (3) = cam3

def refine_point(X_init, points2d, proj_mats):
    if len(proj_mats) != len(points2d):
        raise ValueError(f"Mismatch in number of views: "
                         f"got {len(proj_mats)} proj_mats and {len(points2d)} 2D points")

    def reproj_residuals(X):
        X_h = np.append(X, 1.0)  # homogeneous coordinates
        residuals = []
        for P, pt2d in zip(proj_mats, points2d):
            x_proj = P @ X_h
            x_proj /= x_proj[2]
            residuals.append(x_proj[:2] - pt2d)
        return np.concatenate(residuals)

    res = least_squares(reproj_residuals, X_init, method='lm') # Levenberg-Marquardt, Gauss-Newton series -> https://github.com/braca51e/cs231a-1/blob/master/Lecture%20Notes/04-stereo-systems.pdf page 2-3
    return res.x

def compute_reprojection_errors(X, points2d, proj_mats):
    X_h = np.append(X, 1.0)  # homogeneous coordinates
    errors = []
    for P, pt2d in zip(proj_mats, points2d):
        x_proj = P @ X_h
        x_proj /= x_proj[2]
        error = np.linalg.norm(x_proj[:2] - pt2d)
        errors.append(error)
    return np.array(errors)

def triangulate_with_opencv_sfm(point_tracks: Dict[int, Dict[int, Tuple[float, float]]],
                                rgb_images: Dict[int, np.ndarray],
                                Ps: Dict[int, np.ndarray],
                                ref_image_id: int,
                                min_views: int = 3,
                                reproj_error_minimize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    point_tracks = {
        track_id: {
            ref_view_idx (0): (x, y)
            view_idx (1): (x, y)
            view_idx (2): (x, y)
            ...
        }
    }
    point_tracks.values() => [
        {
            ref_view_idx (0): (x, y)
            view_idx (1): (x, y)
            view_idx (2): (x, y)
            ...
        }
    ]
    track_id = 0 or 1 or 2 or ... or n
    track = {ref_view_idx (0): (x, y), view_idx (1): (x, y), view_idx (2): (x, y), ...}
    len(track) : how many views are tracked for this track_id
        
    points2d: Input vector of vectors of 2d points (the inner vector is per image). Has to be 2 X N.
    projection_matrices: Input vector with 3x4 projections matrices of each image.
    points3d: Output array with computed 3d points. Is 3 x N. --> return Euclidean coordinates
    '''
    points_3d, colors = [], []
    ref_kpts = []

    for track in point_tracks.values():
        if len(track) < min_views:
            continue
        
        proj_mats, points2d = [], []
        for img_id, (x, y) in track.items():
            proj_mats.append(Ps[img_id].astype(np.float32))
            points2d.append(np.array([[x], [y]], dtype=np.float32))

        X_init = cv2.sfm.triangulatePoints(points2d, proj_mats) # https://github.com/opencv/opencv_contrib/blob/4.x/modules/sfm/src/triangulation.cpp#L119-L196
        X_init = X_init[:, 0].reshape(-1)  # (3,1) -> (3,) 3D point as 1D array 
        # minimize reprojection error
        # 1. project 3D point X to all cameras to calculate 2D points
        # 2. adjust 3D point X to minimize the error between the calculated 2D points and the original 2D points    
        
        # reprojection error filtering (before refinement)
        pt2d_list = [pt2d.flatten() for pt2d in points2d]  # (2,)
        reproj_errors = compute_reprojection_errors(X_init, pt2d_list, proj_mats)

        # filter criterion: maximum 30px
        if np.max(reproj_errors) > 30.0: # see mast3r VREC <90px
            continue  # skip this 3D point because it is too far from the original 2D points
        # refine with Levenberg-Marquardt if enabled
        if reproj_error_minimize:
            X = refine_point(X_init, pt2d_list, proj_mats)
        else:
            X = X_init
            
        ref_kpts.append(track[ref_image_id])
        points_3d.append(X)

        # color reference
        first_id = list(track.keys())[0]
        x, y = map(int, track[first_id])
        img = rgb_images[first_id]  # float32, shape (H, W, 3), range [0,1]

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color = (img[y, x] * 255).astype(np.uint8)  # float to uint8
        else:
            color = np.array([255, 255, 255], dtype=np.uint8)   
        colors.append(color)
    
    if len(points_3d) == 0:
        print(f"[⚠] triangulation failed: no valid 3D points (ref: {ref_image_id})")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 2))
        
    return np.array(points_3d), np.array(colors), np.array(ref_kpts) # (points, 3), (points, 3), (points, 2)