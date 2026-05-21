import torch
from PIL import Image
from transformers import pipeline
import numpy as np  
import open3d as o3d
import torchvision.transforms.functional as tf

def convert_tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.cpu().numpy()

def normalize_depth(depth):
    return (depth-depth.min())/(depth.max()-depth.min())

def estimate_depth_DA_v2(tensor):    
    np_arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    image = Image.fromarray(np_arr)
        
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device='cuda')
    depth = pipe(image)["depth"]

    return np.array(depth)

def backproject_2d_to_3d(rgb, E, K, kpts, mask=None):
    """
    Backproject 2D points to 3D using Open3D.

    Parameters:
    - rgb: numpy array (H, W, 3), values clamped between 0 and 1.
    - depth: numpy array (H, W), relative depth estimated from Depth Anything V2.
    - E: Camera extrinsic matrix tensor of shape (3, 4).
    - K: Camera intrinsic matrix tensor of shape (3, 3).
    - backproj_points_color: (n_points=HxW, 3). Colors from rgb image. dtype: np.float32
    - backproj_points_3d: (n_points=HxW, 3). 3D points backproj from rgb image. dtype: np.float32
    - kpts: (m_points, 2). UV pixel coordinates and Z values. dtype: np.float32
    - kpts_3d: (m_points, 3). 3D points corresponding to kpts. dtype: np.float32
    - kpts_color: (m_points, 3). Colors corresponding to kpts. dtype: np.float32
    
    - n_points > m_points

    """
    height, width = rgb.shape[:2]  # (H, W, 3)
    if mask is not None:
        mask = mask.resize((width, height), resample=Image.Resampling.NEAREST) 
        # check alpha mask (binary)
        mask.save(f"alpha_mask_resize.png") # (512, 384)
        mask = tf.to_tensor(mask) 
        mask = convert_tensor_to_numpy(mask)[0] # [1,H,W] / [0.,1.] -> [H,W] / [0.,1.]
        mask = mask > 0.5 # [0.,1.] -> [False, True]

    hom_coords = np.array([[0, 0, 0, 1]])
    E = np.vstack((E, hom_coords)) # (4, 4)

    # depth estimation
    depth = estimate_depth_DA_v2(torch.tensor(rgb).permute(2,0,1).cuda())  # (H,W,3) -> (3,H,W) -> DA -> (H,W) value int 36~114 np.array
    depth = (1-normalize_depth(depth)) # (H,W) invDepth -> Depth value float64 0.0~1.0
    
    # Convert RGB to uint8 and depth to uint16 
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    rgb_uint8 = np.ascontiguousarray(rgb_uint8)

    depth_uint16 = (depth * 1000).astype(np.uint16)
    
    rgb_o3d = o3d.geometry.Image(rgb_uint8)
    depth_o3d = o3d.geometry.Image(depth_uint16)  # Convert meters to millimeters
    
    # Create Open3D intrinsic object
    height, width = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    # Create an Open3D RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)

    backproj_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic=E)
    
    # Get all points and colors
    backproj_points_3d = np.asarray(backproj_point_cloud.points)  # Shape: (n_points, 3)
    backproj_points_color = np.asarray(backproj_point_cloud.colors)  # Shape: (n_points, 3) / value range: [0.0, 1.0]
    backproj_points_color = (backproj_points_color * 255).astype(np.uint8) # value range: [0, 255]

    # (1) depth > 0 pixels coordinates
    valid_pixel_coords = np.argwhere(depth > 0)  # shape: (N, 2) → (v, u)

    # (2) if mask exists, filter only the corresponding pixels
    if mask is not None:
        # select only the pixels where mask is True
        mask_values = mask[valid_pixel_coords[:, 0], valid_pixel_coords[:, 1]]  # shape: (N,)
        masked_backproj_points_3d = backproj_points_3d[mask_values]
        masked_backproj_points_color = backproj_points_color[mask_values]
    else:
        masked_backproj_points_3d = backproj_points_3d.copy()
        masked_backproj_points_color = backproj_points_color.copy() 

    # Convert kpts to int32 for indexing
    kpts_int = kpts[:, :2].astype(np.int32)  # Get U, V coordinates and convert to int32

    # Initialize lists for keypoints in 3D (based on depth map) and colors (from RGB image)
    backproj_kpts_3d = []
    backproj_kpts_color = []
         
    for i in range(kpts_int.shape[0]):
        U, V = kpts_int[i]
        if not (0 <= U < width and 0 <= V < height):
            continue 
        Z = depth[V, U]  # Get depth value at position
        index = V * width + U
        if index < backproj_points_3d.shape[0]:
            backproj_kpts_3d.append(backproj_points_3d[index])
            backproj_kpts_color.append(backproj_points_color[index])
        else:
            print(f"Out of bounds index: Index {index} is not within points array size {backproj_points_3d.shape[0]}")       
        
    # Convert filtered lists to numpy arrays
    backproj_kpts_3d = np.array(backproj_kpts_3d)  # (m_points, 3)
    backproj_kpts_color = np.array(backproj_kpts_color)  # (m_points, 3)
    
    return masked_backproj_points_3d, masked_backproj_points_color, backproj_kpts_3d, backproj_kpts_color
