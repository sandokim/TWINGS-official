import os
import json
import numpy as np
import torch
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary, read_points3D_text
from PIL import Image
from utils.pose_utils import compute_focus_point_from_Es_via_inv
from utils.viz_utils import visualize_point_tracks_grid, visualize_point_cloud
from utils.depth_utils import backproject_2d_to_3d
from utils.ply_utils import storePly, fetchPly_255
from utils.tps_utils import TPS3D
from utils.multiview_utils import feature_extract_and_point_tracking, triangulate_with_opencv_sfm 
from mast3r.model import AsymmetricMASt3R
from utils.tps_utils import select_n_views, select_near_extrinsics, remove_points_near_colmap, prune_points_by_radius_clustering, sample_points_by_distance
from utils.camera_utils import construct_camera_parameters, build_blender_extrinsics, build_pinhole_intrinsics_from_blender_fov
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def per_scene_triangulation(scenes, dataset_type, BASE_DIR, n_views, reproj_error_minimize, n_nearest_views, MIN_VIEWS_FOR_TRIANGULATION, CBPS_ratio, resize_target_long_side=512, selection_mode="nearest", random_seed=None, visualize_tracks=False):  
    metrics_by_scene = {}
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    for scene in scenes:
        SCENE_DIR = os.path.join(BASE_DIR, scene)
        IMAGE_DIR = os.path.join(SCENE_DIR, "images")
        SAVE_PLY_DIR = os.path.join(SCENE_DIR, "multiview_pcd", f"{n_views}_views")
        if not os.path.exists(SAVE_PLY_DIR):
            os.makedirs(SAVE_PLY_DIR)

        if dataset_type == "nerf_llff_data" or dataset_type == "mipnerf360" or dataset_type == "DTU" or dataset_type == "custom":
            intrinsics = read_intrinsics_binary(os.path.join(SCENE_DIR, "sparse/0/cameras.bin"))
            extrinsics = read_extrinsics_binary(os.path.join(SCENE_DIR, "sparse/0/images.bin"))
        elif dataset_type == "blender":
            train_json_path = os.path.join(SCENE_DIR, "transforms_train.json")
            train_IMAGE_DIR = os.path.join(SCENE_DIR, "train")
            with open(train_json_path, 'r') as f:
                train_meta = json.load(f)
            camera_angle_x = train_meta["camera_angle_x"]
            intrinsics = build_pinhole_intrinsics_from_blender_fov(camera_angle_x, width=800, height=800)
            train_extrinsics = build_blender_extrinsics(train_meta["frames"])

            test_json_path = os.path.join(SCENE_DIR, "transforms_test.json")
            test_IMAGE_DIR = os.path.join(SCENE_DIR, "test")
            with open(test_json_path, 'r') as f:
                test_meta = json.load(f)
            test_extrinsics = build_blender_extrinsics(test_meta["frames"])
        bin_path = os.path.join(SCENE_DIR, "sparse/0/points3D.bin")
        txt_path = os.path.join(SCENE_DIR, "sparse/0/points3D.txt")

        if n_views > 0:
            if "scan110" in SCENE_DIR: # for the scene with no initial dense COLMAP pcd
                COLMAP_pts, COLMAP_rgb = None, None
            else:   
                ply_path = os.path.join(SCENE_DIR, str(n_views) + "_views/dense/fused.ply")    
                pcd = fetchPly_255(ply_path)
                COLMAP_pts, COLMAP_rgb = pcd.points, pcd.colors # COLMAP_rgb 0~255 range
        else:
            try:
                COLMAP_pts, COLMAP_rgb, _ = read_points3D_binary(bin_path) # COLMAP_pts.shape (N,3) / COLMAP_rgb.shape (N,3) / rgb values are 0~255 range
            except:
                COLMAP_pts, COLMAP_rgb, _ = read_points3D_text(txt_path)      

        if dataset_type == "blender":
            train_extrinsics = {train_extrinsics[k].name.split(".")[0]: train_extrinsics[k] for k in train_extrinsics} # access by image name ("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", point3D_ids"])
            train_image_id_to_ori_res = {}
            for i in train_extrinsics:
                img_name = train_extrinsics[i].name # .png, .JPG
                img_key = img_name.split(".")[0]
                path = os.path.join(train_IMAGE_DIR, img_name)
                try:
                    with Image.open(path) as im:
                        w, h = im.size
                except Exception:
                    raise RuntimeError(f"Failed to load image: {path}")
                train_image_id_to_ori_res[img_key] = (h, w)
                
            test_extrinsics = {test_extrinsics[k].name.split(".")[0]: test_extrinsics[k] for k in test_extrinsics} # access by image name ("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", point3D_ids"])
            
            test_image_id_to_ori_res = {}
            for i in test_extrinsics:
                img_name = test_extrinsics[i].name # .png, .JPG
                img_key = img_name.split(".")[0]
                path = os.path.join(test_IMAGE_DIR, img_name)
                try:
                    with Image.open(path) as im:
                        w, h = im.size
                except Exception:
                    raise RuntimeError(f"Failed to load image: {path}")
                test_image_id_to_ori_res[img_key] = (h, w)      
                
            train_cam_infos, test_cam_infos, pseudo_cam_infos, camera_extent = select_n_views(dataset_type=dataset_type,
                                        scene_dir=SCENE_DIR,
                                        extrinsics=train_extrinsics,
                                        intrinsics=intrinsics,
                                        n_views=n_views,
                                        llffhold=8,
                                        eval_mode=True)
        else:
            extrinsics = {extrinsics[k].name.split(".")[0]: extrinsics[k] for k in extrinsics} # access by image name ("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", point3D_ids"])
            image_id_to_ori_res = {}
            for i in extrinsics:
                img_name = extrinsics[i].name # .png, .JPG
                img_key = img_name.split(".")[0]
                path = os.path.join(IMAGE_DIR, img_name)
                try:
                    with Image.open(path) as im:
                        w, h = im.size
                except Exception:
                    raise RuntimeError(f"Failed to load image: {path}")
                image_id_to_ori_res[img_key] = (h, w) 
         
            train_cam_infos, test_cam_infos, pseudo_cam_infos, camera_extent = select_n_views(dataset_type=dataset_type,
                                        scene_dir=SCENE_DIR,
                                        extrinsics=extrinsics,
                                        intrinsics=intrinsics,
                                        n_views=n_views,
                                        llffhold=8,
                                        eval_mode=True)
        
        train_cam_names = [c.image_name for c in train_cam_infos]
        test_cam_names = [c.image_name for c in test_cam_infos]
        
        if dataset_type == "blender":
            train_extrinsics = {k: train_extrinsics[k] for k in train_cam_names}
            test_extrinsics = {k: test_extrinsics[k] for k in test_cam_names}
            train_Ks, train_Es, train_Ps = construct_camera_parameters(train_extrinsics, intrinsics, train_image_id_to_ori_res, resize_target_long_side)
            _, test_Es, _ = construct_camera_parameters(test_extrinsics, intrinsics, test_image_id_to_ori_res, resize_target_long_side)
        else:
            train_extrinsics = {k: extrinsics[k] for k in train_cam_names}
            test_extrinsics = {k: extrinsics[k] for k in test_cam_names}
            train_Ks, train_Es, train_Ps = construct_camera_parameters(train_extrinsics, intrinsics, image_id_to_ori_res, resize_target_long_side)
            _, test_Es, _ = construct_camera_parameters(test_extrinsics, intrinsics, image_id_to_ori_res, resize_target_long_side)

        focus_point = compute_focus_point_from_Es_via_inv(train_Es)

        all_tri_pts_3d, all_tri_colors = [], []
        all_pts, all_colors = [], []

        for viewpoint_cam in train_cam_infos:
            image_name = viewpoint_cam.image_name
            print(f"[1] Near Training Views Selection... (image_name: {image_name})")
            near_extrinsics, near_Es = select_near_extrinsics(
                train_Es, image_name, train_extrinsics,
                n_select=n_nearest_views,
                selection_mode=selection_mode,
                random_seed=random_seed,
            )

            print(f"[2] 2D-2D matching extraction... (image_name: {image_name})")    
            if dataset_type == "blender":
                point_tracks, rgb_images = feature_extract_and_point_tracking(
                    near_extrinsics, train_IMAGE_DIR, SCENE_DIR, resize_target_long_side, image_name, mast3r_model, device, white_background=True)
            else:
                point_tracks, rgb_images = feature_extract_and_point_tracking(
                    near_extrinsics, IMAGE_DIR, SCENE_DIR, resize_target_long_side, image_name, mast3r_model, device, white_background=False)

            if len(point_tracks) == 0:
                print(f"\n [⚠] Skipping {image_name} due to zero point tracks. \n")
                continue
            
            # Point track visualization: display ref image and selected near images horizontally concatenated
            if visualize_tracks:
                try:
                    visualize_point_tracks_grid(
                        point_tracks=point_tracks,
                        rgb_images=rgb_images,
                        ref_image_id=image_name,
                        image_ids=list(near_extrinsics.keys()),
                        num_show_ge3=5,
                        num_show_eq2=5,
                    )
                except Exception as e:
                    print(f"[Warn] visualize_point_tracks_grid failed: {e}")
            
            print(f"[3] 3D point triangulation... (image_name: {image_name}) / reproj_error_minimize: {reproj_error_minimize}")
            tri_pts_3d, tri_colors, ref_kpts = triangulate_with_opencv_sfm(point_tracks, rgb_images, train_Ps, image_name, MIN_VIEWS_FOR_TRIANGULATION, reproj_error_minimize)       
            if tri_pts_3d.shape[0] == 0:
                print(f"[⚠] Skipping {image_name} due to zero triangulated points.")
                continue        
            
            # visualize_point_cloud(tri_pts_3d, tri_colors, train_Es, test_Es, title=f"Triangulated 3D Points with Camera Poses (image_name: {image_name})", focus_point=focus_point, draw_focus_rays=True, near_Es=near_Es)
            # save_tri_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_tri_pts.ply")
            # storePly(save_tri_path, tri_pts_3d, tri_colors)
            # print(f"[✔] PLY file saved: {save_tri_path}")
            
            # Select ref image based on ref_image_id, estimate depth, then perform 2D to 3D backprojection       
            backproj_points_3d, backproj_points_color, backproj_kpts_3d, backproj_kpts_color = backproject_2d_to_3d(rgb_images[image_name], train_Es[image_name], train_Ks[image_name], ref_kpts, mask=None) # viewpoint_cam.mask
              
            # visualize_point_cloud(backproj_points_3d, backproj_points_color, train_Es, test_Es, title=f"Backprojected 3D Points with Camera Poses (image_name: {image_name})")
            # save_backproj_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_backproj_pts.ply")
            # storePly(save_backproj_path, backproj_points_3d, backproj_points_color)
                  
            # visualize_point_cloud(backproj_kpts_3d, backproj_kpts_color, train_Es, test_Es, title=f"Backprojected Keypoints 3D Points with Camera Poses (image_name: {image_name})")
            # save_backproj_kpts_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_backproj_kpts.ply")
            # storePly(save_backproj_kpts_path, backproj_kpts_3d, backproj_kpts_color)

            # Random Decimation
            decim_pts = 30_000
            if backproj_points_3d.shape[0] > decim_pts:
                decim_idx = np.random.choice(backproj_points_3d.shape[0], decim_pts, replace=False)
                backproj_points_3d = backproj_points_3d[decim_idx]
                backproj_points_color = backproj_points_color[decim_idx]
                # save_decimated_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_decim_pts.ply")
                # storePly(save_decimated_path, backproj_points_3d, backproj_points_color)
            
            # TPS Deformation 
            print(f"Performing TPS Deformation... (image_name: {image_name})")
            deform_pts, deform_colors = TPS3D(
                backproj_kpts_3d, tri_pts_3d, backproj_points_3d, backproj_points_color
            )    
                
            # visualize_point_cloud(deform_pts, deform_colors, train_Es, test_Es, title=f"TPS Deformed 3D Points with Camera Poses (image_name: {image_name})")
            # save_tps_pts_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_tps_pts.ply")
            # storePly(save_tps_pts_path, deform_pts, deform_colors)  
            
            # Distance-based Sampling
            distance_threshold = camera_extent / CBPS_ratio 
            deform_pts, mask = sample_points_by_distance(deform_pts, tri_pts_3d, distance_threshold)
            deform_colors = deform_colors[mask]
            
            # save_cbps_pts_path = os.path.join(SAVE_PLY_DIR, f"{image_name}_cbps_pts.ply")
            # storePly(save_cbps_pts_path, deform_pts, deform_colors)
            
            # Filtering
            if COLMAP_pts is not None:
                deform_pts, deform_colors = remove_points_near_colmap(deform_pts, deform_colors, COLMAP_pts, margin=0.05)
            else:
                print(f"[Info] COLMAP_pts is None → remove_points_near_colmap skipped (image_name: {image_name})")
            deform_pts, deform_colors = prune_points_by_radius_clustering(deform_pts, deform_colors, radius=0.05, min_points=5)
            
            all_tri_pts_3d.append(tri_pts_3d)
            all_tri_colors.append(tri_colors)
            all_pts.append(deform_pts)
            all_colors.append(deform_colors)

        all_tri_pts_3d = np.concatenate(all_tri_pts_3d, axis=0)
        all_tri_colors = np.concatenate(all_tri_colors, axis=0)
        all_pts = np.concatenate(all_pts, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        if COLMAP_pts is not None and COLMAP_rgb is not None:
            all_pts = np.vstack([all_pts, COLMAP_pts])
            all_colors = np.vstack([all_colors, COLMAP_rgb])
        else:
            print("[Info] COLMAP point cloud is None → COLMAP_pts concat skipped")

        # print(f"[5] Visualizing all 3D point clouds... : {len(train_cam_names)} ref_image_id used")
        # visualize_point_cloud(all_tri_pts_3d, all_tri_colors, train_Es, test_Es, title=f"ALL Triangulated 3D Points with Camera Poses", focus_point=focus_point, draw_focus_rays=True)
        # save_path = os.path.join(SAVE_PLY_DIR, f"tri_pts.ply")
        # storePly(save_path, all_tri_pts_3d, all_tri_colors)
        # print(f"[✔] Triangulated 3D Points PLY file saved: {save_path}")

        # visualize_point_cloud(all_pts, all_colors, train_Es, test_Es, title=f"Enhanced 3D Points with Camera Poses", focus_point=focus_point, draw_focus_rays=True) 
        save_path = os.path.join(SAVE_PLY_DIR, f"twings_init_pcd.ply")
        storePly(save_path, all_pts, all_colors)
        print(f"[✔] TWINGS-Init PLY file saved: {save_path}")
        
        del all_tri_pts_3d, all_tri_colors, all_pts, all_colors

def main():  
    DATASET_ROOT = "/mai_nas/KHS/nerfs/data" # change this to your dataset root directory
    selection_mode = "nearest"  # "nearest" or "random"
    random_seed = 42
    reproj_error_minimize = True    
    visualize_tracks = False  
    
    for dataset_type in ["nerf_llff_data", "DTU", "mipnerf360"]:
        print(f"[Start] {dataset_type}")
        # A larger CBPS_ratio gives higher accuracy but samples fewer 3D points; see Fig. 3 in the supplementary materials.
        if dataset_type == "nerf_llff_data":
            CBPS_ratio = 8
        else:
            CBPS_ratio = 32
        
        print("[1] Load camera parameters...")
        if dataset_type == "nerf_llff_data":
            BASE_DIR = os.path.join(DATASET_ROOT, "nerf_llff_data")
            scenes=["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]  
        elif dataset_type == "mipnerf360":
            BASE_DIR = os.path.join(DATASET_ROOT, "mipnerf360")
            scenes=["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
        elif dataset_type == "DTU":        
            BASE_DIR = os.path.join(DATASET_ROOT, "DTU", "dtu_corgs")
            scenes=["scan8","scan21","scan30","scan31","scan34","scan38","scan40","scan41","scan45","scan55","scan63","scan82","scan103","scan110","scan114"]
        elif dataset_type == "blender":
            BASE_DIR = os.path.join(DATASET_ROOT, "nerf_synthetic")
            scenes=["chair", "drums", "ficus","hotdog", "lego", "materials", "mic", "ship"]
        elif dataset_type == "custom":
            BASE_DIR = os.path.join(DATASET_ROOT, "multicam")
            scenes = ["my_face"]
        
        if dataset_type == "nerf_llff_data":
            n_views_list = [3, 6, 9]
        elif dataset_type == "mipnerf360":
            n_views_list = [12, 24]
        elif dataset_type == "DTU":
            n_views_list = [3, 6, 9]
        elif dataset_type == "blender":
            n_views_list = [8]
        elif dataset_type == "custom":
            n_views_list = [3, 6, 9]
            
        for n_views in n_views_list:
            print(f"[Start] {dataset_type} {n_views}")
            if n_views == 3:
                MIN_VIEWS_FOR_TRIANGULATION = 2
                n_nearest_views = 2
            elif 3 < n_views <= 9:
                MIN_VIEWS_FOR_TRIANGULATION = 2
                n_nearest_views = MIN_VIEWS_FOR_TRIANGULATION * 2
            elif 9 < n_views <= 24:
                MIN_VIEWS_FOR_TRIANGULATION = 2 
                n_nearest_views = MIN_VIEWS_FOR_TRIANGULATION * 2
            else:
                raise ValueError("n_views must be 24 or less") # You can use more than 24 views, but it is not recommended since this no longer qualifies as sparse-view; see Sec. D in the supplementary materials.

            per_scene_triangulation(
                scenes, dataset_type, BASE_DIR, n_views,
                reproj_error_minimize, n_nearest_views, MIN_VIEWS_FOR_TRIANGULATION,
                CBPS_ratio,
                resize_target_long_side=512,
                selection_mode=selection_mode,
                random_seed=random_seed,
                visualize_tracks=visualize_tracks,
            )

            print(f"[End] {dataset_type} {n_views}")
    
if __name__ == "__main__":
    main()