import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import glob
from scene.cameras import PseudoCamera
from utils.camera_utils import cameraList_from_camInfos
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360, getNerfppNorm
from utils.camera_utils import readColmapCameras, readCamerasFromTransforms
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from typing import NamedTuple

class SceneInfo(NamedTuple):
    # point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    # ply_path: str

def TPS3D(points, ctrlpoints, object_points, object_colors):
    """
    Thin Plate Spline in 3D

    :param points: Control points (nx3 array)
    :param ctrlpoints: Destination control points (nx3 array)
    :param object_points: Points to be transformed (mx3 array)
    :return: Transformed object points (mx3 array)
    """
    # Calculate Parameters
    npnts = points.shape[0]
    K = np.zeros((npnts, npnts))

    # Compute R^2 for control points
    for rr in range(npnts):
        for cc in range(npnts):
            K[rr, cc] = np.sum((points[rr, :] - points[cc, :]) ** 2)
            K[cc, rr] = K[rr, cc]

    # Calculate kernel function R
    K = np.maximum(K, 1e-320)
    K = np.sqrt(K)  # R for 3D

    # Calculate P matrix
    P = np.hstack((np.ones((npnts, 1)), points))  # nX4 for 3D

    # Calculate L matrix
    L = np.vstack((np.hstack((K, P)), np.hstack((P.T, np.zeros((4, 4))))))

    # Solve for parameters
    param = np.linalg.pinv(L).dot(np.vstack((ctrlpoints, np.zeros((4, 3)))))

    # Calculate new coordinates (x', y', z') for each point in the object
    pntsNum = object_points.shape[0]
    K = np.zeros((pntsNum, npnts))
    gx = object_points[:, 0]
    gy = object_points[:, 1]
    gz = object_points[:, 2]

    for nn in range(npnts):
        K[:, nn] = (gx - points[nn, 0]) ** 2 + (gy - points[nn, 1]) ** 2 + (gz - points[nn, 2]) ** 2

    K = np.maximum(K, 1e-320)
    K = np.sqrt(K)  # R for 3D

    # Calculate transformed object points
    P = np.hstack((np.ones((pntsNum, 1)), gx[:, np.newaxis], gy[:, np.newaxis], gz[:, np.newaxis]))
    L = np.hstack((K, P))
    wobject = L.dot(param)

    # Round to 3 decimal places
    wobject[:, 0] = np.round(wobject[:, 0] * 10**3) * 10**-3
    wobject[:, 1] = np.round(wobject[:, 1] * 10**3) * 10**-3
    wobject[:, 2] = np.round(wobject[:, 2] * 10**3) * 10**-3
    
    wobject_colors = object_colors

    return wobject, wobject_colors


def select_n_views(dataset_type: str,
                      scene_dir: str,
                      extrinsics: Dict[str, Any],
                      intrinsics: Dict[int, Any],
                      n_views: int = 3,
                      llffhold: int = 8,
                      eval_mode: bool = True) -> List[str]:
    """
    dataset_type: 'custom' or 'benchmark'
    scene_dir: root directory of dataset (e.g., nerfs/data/nerf_llff_data/horns)
    extrinsics: dict keyed by image_name -> colmap Image object
    intrinsics: dict keyed by camera_id -> colmap Camera object
    return: list of selected image_names
    """
    if dataset_type == "nerf_llff_data" or dataset_type == "mipnerf360" or dataset_type == "DTU" or dataset_type == "custom":
        image_folder = os.path.join(scene_dir, "images")
        # get image name list
        rgb_mapping = sorted([
            f for f in glob.glob(os.path.join(image_folder, '*'))
            if f.lower().endswith(('jpg', 'png'))
        ])

        # Convert image_name to extrinsics key
        extrinsics_named = {v.name: v for v in extrinsics.values()}  # Ensure match by name
        cam_extrinsics = extrinsics_named
        cam_intrinsics = intrinsics

        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=image_folder,
            path=scene_dir,
            rgb_mapping=rgb_mapping
        )
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        
        scene_info_nerf_normalization = getNerfppNorm(cam_infos)
        camera_extent = scene_info_nerf_normalization["radius"]
        print(f"[Camera Extent] {camera_extent}")
        # cam_infos 기반 extent 저장 (추후 단일 라인 로깅 시 사용)
        full_cam_infos_extent = camera_extent
    
    elif dataset_type == "blender":
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(scene_dir, "transforms_train.json")
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(scene_dir, "transforms_test.json")
       
    if dataset_type == "nerf_llff_data" or dataset_type == "mipnerf360":
        # NeRF-LLFF format
        if eval_mode:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

        if n_views > 0:
            idx_sub = np.linspace(0, len(train_cam_infos) - 1, n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
            assert len(train_cam_infos) == n_views
            
    elif dataset_type == "DTU":
        # DTU dataset
        if eval_mode:
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            if n_views > 0:
                train_idx = train_idx[:n_views]
            train_cam_infos = [cam_infos[i] for i in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
    
    elif dataset_type == "blender":
        if not eval_mode:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []    
            
        if n_views > 0:
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
            eval_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % llffhold == 0]
            test_cam_infos = eval_cam_infos
            print(f"len(train_cam_infos) is {len(train_cam_infos)}")
            assert len(train_cam_infos) == n_views

        scene_info_nerf_normalization = getNerfppNorm(train_cam_infos)
        camera_extent = scene_info_nerf_normalization["radius"]
        print(f"[Camera Extent] {camera_extent}")

    elif dataset_type == "custom":
        # Simply select first N image names
        if eval_mode:
            train_idx = [1,4,5]
            test_idx = [0,2,3,6]
            if n_views > 0:
                train_idx = train_idx[:n_views]
                train_cam_infos = [cam_infos[i] for i in train_idx]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []  
            
    scene_info_nerf_normalization = getNerfppNorm(train_cam_infos)
    camera_extent = scene_info_nerf_normalization["radius"]
    print(f"[Train Camera Extent] {camera_extent}")     
    # # log: record scene, number of views, cam_infos based extent, train_cam_infos based extent in one line
    # try:
    #     log_path = os.path.join(os.path.dirname(__file__), "camera_extent_log.txt")
    #     scene_name = os.path.basename(os.path.normpath(scene_dir))
    #     nviews_val = len(train_cam_infos)
    #     full_extent_val = full_cam_infos_extent if 'full_cam_infos_extent' in locals() else ''
    #     with open(log_path, 'a', encoding='utf-8') as f:
    #         f.write(f"{scene_name}\t{nviews_val}\t{full_extent_val}\t{camera_extent}\n")
    # except Exception as e:
    #     print(f"[Warn] Camera extent logging failed: {e}")
         
    train_cameras = {}
    pseudo_cameras = {}

    resolution_scale = 1
    scene_info = SceneInfo(train_cameras=train_cam_infos, test_cameras=test_cam_infos, nerf_normalization=scene_info_nerf_normalization)
    train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, resolution=1)
    pseudo_cams = []

    # generate_random_poses_360 --> transform_poses_pca function scales down all camera positions to fit within a [-1, 1] cube based on the max norm
    # generate_random_poses_dtu --> generate_random_poses_llff function sets the optical axis based on the focus_pt
    if dataset_type =="nerf_llff_data":
        pseudo_poses = generate_random_poses_llff(train_cameras[resolution_scale], n_poses=100) 
    elif dataset_type =="mipnerf360":
        pseudo_poses = generate_random_poses_360(train_cameras[resolution_scale], n_frames=100)
    elif dataset_type =="DTU":
        pseudo_poses = generate_random_poses_llff(train_cameras[resolution_scale], n_poses=100)
    elif dataset_type =="blender":
        pseudo_poses = generate_random_poses_360(train_cameras[resolution_scale], n_frames=100)
    elif dataset_type =="custom":
        pseudo_poses = generate_random_poses_llff(train_cameras[resolution_scale], n_poses=100)

    view = train_cameras[resolution_scale][0]
    for pose in pseudo_poses:
        pseudo_cams.append(PseudoCamera(
            R=pose[:3, :3], T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy, # Not R.T but R
            width=view.image_width, height=view.image_height
        ))
    pseudo_cameras[resolution_scale] = pseudo_cams # {1.0, PseudoCamera(), PseudoCamera(), ...}       
    pseudo_cam_infos = pseudo_cameras[resolution_scale]  

    return train_cam_infos, test_cam_infos, pseudo_cam_infos, camera_extent

def select_near_extrinsics(
    Es: Dict[str, np.ndarray],
    ref_image_id: str,
    extrinsics: Dict[str, Any],  # Image 객체들
    n_select: int = 1,
    selection_mode: str = "nearest",
    random_seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    - extrinsics: image_name → Image object (COLMAP style)
    - Es: image_name → 3x4 extrinsic matrix (numpy array)
    - selection_mode: 'nearest' or 'random'
    - return: (near_extrinsics, near_Es)
    """
    def cartesian_to_spherical(coord):
        x, y, z = coord
        r = np.linalg.norm(coord)
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)
        return az, el, r

    def spherical_distance(s1, s2):
        da = s1[0] - s2[0]
        de = s1[1] - s2[1]
        dr = s1[2] - s2[2]
        return np.sqrt(da**2 + de**2 + dr**2)

    def camera_center(E):
        R, t = E[:, :3], E[:, 3]
        return (-R.T @ t.reshape(3, 1)).flatten()

    if selection_mode == "random":
        other_ids = [img_id for img_id in Es.keys() if img_id != ref_image_id]
        if len(other_ids) == 0:
            selected_ids = [ref_image_id]
        else:
            n_pick = min(n_select, len(other_ids))
            rng = np.random.default_rng(random_seed)
            selected_others = rng.choice(other_ids, size=n_pick, replace=False).tolist()
            selected_ids = [ref_image_id] + selected_others
        print(f"[✔] ref {ref_image_id} based, random {n_select} selected → {selected_ids[1:]}")
    else:
        ref_E = Es[ref_image_id]
        ref_center = camera_center(ref_E)
        ref_sph = cartesian_to_spherical(ref_center)

        dists = []
        for img_id, E in Es.items():
            if img_id == ref_image_id:
                continue
            center = camera_center(E)
            sph = cartesian_to_spherical(center)
            dist = spherical_distance(ref_sph, sph)
            dists.append((img_id, dist))

        # select nearest n views
        dists.sort(key=lambda x: x[1])
        selected_ids = [ref_image_id] + [img_id for img_id, _ in dists[:n_select]]
        print(f"[✔] ref {ref_image_id} based, nearest {n_select} selected → {selected_ids[1:]}")

    near_extrinsics = {k: extrinsics[k] for k in selected_ids}
    near_Es = {k: Es[k] for k in selected_ids}
    return near_extrinsics, near_Es

# remove points near COLMAP based function
def remove_points_near_colmap(deform_pts, deform_colors, colmap_pts, margin=0.05):
    tree = cKDTree(colmap_pts)
    distances, _ = tree.query(deform_pts)
    mask = distances > margin
    print(f"[Filter] {np.sum(~mask)} points removed near COLMAP")
    return deform_pts[mask], deform_colors[mask]

# Radius-based clustering and downsampling
def prune_points_by_radius_clustering(points, colors, radius=0.05, min_points=5):
    tree = KDTree(points)
    indices = tree.query_radius(points, r=radius)
    processed = np.zeros(len(points), dtype=bool)
    pruned_points, pruned_colors = [], []

    for i in range(len(points)):
        if processed[i]:
            continue
        neighbors = indices[i]
        if len(neighbors) >= min_points:
            cluster = points[neighbors]
            centroid = np.mean(cluster, axis=0)
            closest_idx = neighbors[np.argmin(np.linalg.norm(cluster - centroid, axis=1))]
            pruned_points.append(points[closest_idx])
            pruned_colors.append(colors[closest_idx])
            processed[neighbors] = True
        else:
            pruned_points.append(points[i])
            pruned_colors.append(colors[i])
            processed[i] = True

    pruned_points = np.array(pruned_points)
    pruned_colors = np.array(pruned_colors)
    print(f"[Prune] Pruned to {len(pruned_points)} points from {len(points)}")
    return pruned_points, pruned_colors

def sample_points_by_distance(deformed_pts, ref_pts, distance_threshold):
    """
    deformed_pts: (N, 3) array of deformed points
    ref_pts: (M, 3) array of reference points
    distance_threshold: float distance threshold
    return: filtered_pts, mask array of boolean values
    """
    tree = cKDTree(ref_pts)
    distances, _ = tree.query(deformed_pts)
    mask = distances <= distance_threshold
    print(f"[Distance] {np.sum(~mask)} points removed by distance threshold {distance_threshold:.4f}")
    return deformed_pts[mask], mask
