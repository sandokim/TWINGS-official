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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import scipy
import matplotlib.pyplot as plt
from torch import nn
import copy
from typing import Dict, Any, Tuple, List
from types import SimpleNamespace
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from scene.cameras import PseudoCamera
from PIL import Image
import os
import json
from pathlib import Path
import sys
import math
from typing import NamedTuple

WARNED = False

class CameraInfo(NamedTuple):
    uid: int
    K: np.array
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array
    focalx: float
    focaly: float


def loadCam(resolution, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.mask is not None:
        loaded_mask = PILtoTorch(cam_info.mask, resolution)
        if loaded_mask.shape[0] == 4:
            loaded_mask = loaded_mask[:3]
    else:
        loaded_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, bounds=cam_info.bounds,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device="cuda")


def cameraList_from_camInfos(cam_infos, resolution_scale, resolution):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(resolution, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def construct_camera_parameters(extrinsics: Dict[int, Any], intrinsics: Dict[int, Any], image_id_to_ori_res: Dict[int, Tuple[int, int]], resize_target_long_side: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    Ks, Es, Ps = {}, {}, {}
    for image_id, extr in extrinsics.items():
        intr = intrinsics[extr.camera_id]
        model = intr.model

        if model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
            fx = fy = intr.params[0]
            cx, cy = intr.params[1:3]
        elif model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2:4]
        elif model == "OPENCV":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            raise NotImplementedError(f"Unsupported camera model: {model}")
        
        # --- start correction --- #
        # In practice, skipping this K correction may produce visually similar 3D point clouds,
        # but applying it is the correct choice for geometric accuracy and consistency.
        # original resolution
        ori_H, ori_W = image_id_to_ori_res[image_id]

        # reproduce MASt3R's resize + crop method exactly
        if ori_W >= ori_H:
            scale = resize_target_long_side / ori_W
            resized_W = resize_target_long_side
            resized_H = round(ori_H * scale)
        else:
            scale = resize_target_long_side / ori_H
            resized_H = resize_target_long_side
            resized_W = round(ori_W * scale)

        # center crop (aligned to multiples of 16, as in MASt3R)
        cx_resize, cy_resize = resized_W // 2, resized_H // 2
        halfw = (2 * cx_resize) // 16 * 8
        halfh = (2 * cy_resize) // 16 * 8
        if not (ori_W != ori_H):  # for square image, correct height to maintain aspect ratio
            halfh = int(3 * halfw / 4)

        crop_offset_x = cx_resize - halfw
        crop_offset_y = cy_resize - halfh

        # 보정된 fx, fy, cx, cy
        fx_corr = fx * scale
        fy_corr = fy * scale
        cx_corr = cx * scale - crop_offset_x
        cy_corr = cy * scale - crop_offset_y

        K = np.array([
            [fx_corr, 0, cx_corr],
            [0, fy_corr, cy_corr],
            [0,     0,       1]
        ], dtype=np.float32)
        # --- 보정 끝 ---

        R = qvec2rotmat(extr.qvec)
        t = extr.tvec.reshape(3, 1)
        E = np.hstack([R, t])
        P = K @ E

        Ks[image_id] = K
        Es[image_id] = E
        Ps[image_id] = P
        
    return Ks, Es, Ps


def construct_pseudo_cam_parameters(pseudo_cam_infos: List[PseudoCamera]) -> Dict[int, np.ndarray]:
    """
    Extract extrinsic matrix [R|t] only from PseudoCamera list
    Returns:
        Es: Dict[idx → E], where E is 3x4 np.ndarray
    """
    Es = {}
    for idx, cam in enumerate(pseudo_cam_infos):
        R = cam.R  # (3, 3)
        t = cam.T.reshape(3, 1)  # (3, 1)
        E = np.hstack([R, t])  # (3, 4)
        Es[idx] = E
    return Es

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_y, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
        K = np.eye(3)
        K[0,0] = focal_length_x
        K[1,1] = focal_length_y
        K[0,2] = width / 2
        K[1,2] = height / 2

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        try:
            masks_folder = os.path.join(path, "masks")
            mask_name = key # Image.open(mask_dir / '{0:05d}.png'.format(idx))
            mask_name = os.path.join(masks_folder, mask_name)
            mask = Image.open(mask_name)
        except:
            mask = None
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=mask, bounds=bounds, focalx=focal_length_x, focaly=focal_length_y, K=K)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background=True, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = cam_name # os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # load RGBA image and create alpha mask + perform background composition
            im_data = np.array(Image.open(image_path).convert("RGBA"))
            alpha = im_data[:, :, 3]
            mask_img = Image.fromarray((alpha >= 128).astype(np.uint8) * 255, mode='L')
            
            # check alpha mask (binary)
            mask_img.save(f"alpha_mask.png")

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1.0 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            # focalx, focaly
            focalx = fov2focal(fovx, image.size[0])
            focaly = fov2focal(fovy, image.size[1])

            K = np.eye(3)
            K[0,0] = focalx
            K[1,1] = focaly
            K[0,2] = image.size[0] / 2
            K[1,2] = image.size[1] / 2

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask_img, K=K,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], bounds=None, focalx=focalx, focaly=focaly))
            
    return cam_infos



def build_pinhole_intrinsics_from_blender_fov(camera_angle_x: float, width: int = 800, height: int = 800):
    fx = 0.5 * width / np.tan(0.5 * camera_angle_x)
    fy = fx
    cx, cy = 0.5 * width, 0.5 * height
    return {
        1: SimpleNamespace(
            id=1,
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy],
        )
    }


def build_blender_extrinsics(frames):
    extrinsics_dict = {}
    for idx, frame in enumerate(frames):
        transform_matrix = np.array(frame["transform_matrix"])  # c2w
        transform_matrix[:3, 1:3] *= -1  # Blender -> COLMAP conversion
        w2c = np.linalg.inv(transform_matrix)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        qvec = rotmat2qvec(R)
        file_path = frame["file_path"]
        image_name = os.path.basename(file_path) + ".png"
        extrinsics_dict[image_name] = SimpleNamespace(
            id=idx,
            qvec=qvec,
            tvec=t,
            camera_id=1,
            name=image_name,
            xys=None,
            point3D_ids=None,
        )
    return extrinsics_dict