import cv2 as cv
import numpy as np
import os
import glob
import json
import logging
import sys
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, "camera.py")) else os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from camera import *

# ----------------- Logger 설정 -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'camera_calibration_extrinsics.log'), mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ----------------- Settings -------------------
CHECKERBOARD = (7, 4)
real_world_distance = 30  # mm
SCENE_UNIT_PER_MM = 1.0 / 30.0  # 30 mm = 1 unit
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
window_size = (640, 480)

# ----------------- Function Definitions -------------------
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(int(i) for i in imgpts[2].ravel()), (255, 0, 0), 3)
    img = cv.line(img, corner, tuple(int(i) for i in imgpts[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(int(i) for i in imgpts[0].ravel()), (0, 0, 255), 3)
    return img

def load_intrinsics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["mtx"]), np.array(data["dist"])

def extract_corners(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        logger.warning(f"Checkerboard not found: {img_path}")
        return None, None, img, gray
    corners_refined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return corners_refined, corners, img, gray, ret

def compute_pose(objp, imgpts, mtx, dist):
    '''
    The cv::solvePnP() returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame, using different methods:
    object(checkerboard) frame's position in camera frame?
    cam0 in board frame -> T_board_to_cam0
    cam1 in board frame -> T_board_to_cam1
    cam2 in board frame -> T_board_to_cam2
    cam3 in board frame -> T_board_to_cam3
    
    The estimated pose is thus the rotation (rvec) and the translation (tvec) vectors that allow transforming a 3D point expressed in the world frame into the camera frame
    
    Here, the rotation_vector and translation_vector together give the position of the camera in world coordinates, assuming the object points are fixed in the world.
    '''
    ret, rvec, tvec = cv.solvePnP(objp, imgpts, mtx, dist) # rvec, tvec -> w2c
    if ret:
        print("Rotation vector:\n", rvec)
        print("Translation vector:\n", tvec)
    else:
        print("solvePnP failed to find a solution.")
    
    return rvec, tvec

def compose_transform(rvec, tvec):
    R, _ = cv.Rodrigues(rvec) # w2c
    T_w2c = np.eye(4) # w2c
    T_w2c[:3,:3] = R
    T_w2c[:3, 3] = tvec.ravel()
    
    R_c2w = R.T
    t_c2w = -R_c2w @ tvec.ravel()
    T_c2w = np.eye(4)
    T_c2w[:3,:3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    
    return T_w2c, T_c2w

def get_object_points():
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * real_world_distance
    return objp


def plot_camera_poses(cam_poses, board_pose=None, title="Camera & Checkerboard Poses", axis_length=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cam_colors = ['r', 'g', 'b']  # X, Y, Z

    for cam_name, T in cam_poses.items():
        T = np.array(T)
        origin = T[:3, 3]
        R = T[:3, :3]

        for i in range(3):
            axis_vec = R[:, i] * axis_length
            ax.quiver(*origin, *axis_vec, color=cam_colors[i], linewidth=3, alpha=0.9)
        ax.text(*origin, cam_name, fontsize=7, color='k')

    if board_pose is not None:
        T = np.array(board_pose)
        origin = T[:3, 3]
        R = T[:3, :3]
        board_colors = ['firebrick', 'seagreen', 'royalblue']
        for i in range(3):
            axis_vec = R[:, i] * axis_length
            ax.quiver(*origin, *axis_vec, color=board_colors[i], linewidth=3, alpha=0.9)
        ax.text(*origin, "checkerboard", fontsize=7, color='k')

    axis_range = 600 
    ax.set_xlim(-axis_range, axis_range)
    ax.set_ylim(-axis_range, axis_range)
    ax.set_zlim(-axis_range, axis_range)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    ax.view_init(elev=30, azim=135)
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])  # force 1:1:1 ratio
    plt.tight_layout()
    plt.show()


# ----------------- Main -------------------
def main():
    base_dir = "multicam/build/Desktop_Qt_6_9_0_MSVC2022_64bit-Release/scene/7_camera_calib_2/checkerboard/multicapture"
    cam_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    logger.info(f"Cam directories: {cam_dirs}")

    objp = get_object_points()
    axis = np.float32([[real_world_distance, 0, 0],
                       [0, real_world_distance, 0],
                       [0, 0, real_world_distance]]).reshape(-1, 3)

    cam_poses = {} 
    cam_errors = {}

    for i, cam_path in enumerate(cam_dirs):
        img_files = sorted(glob.glob(os.path.join(cam_path, "*.jpg")))
        if not img_files:
            logger.warning(f"No image found in {cam_path}")
            continue

        img_path = img_files[0]
        img_filename = os.path.basename(img_path)           # e.g., cam0_20240618.jpg
        cam_name = img_filename                             #
        
        intrinsics_path = os.path.join(cam_path, "intrinsics.json")
        mtx, dist = load_intrinsics(intrinsics_path)

        print("cam name: ", cam_name)
        corners2, _, img, gray, ret= extract_corners(img_path)
        
        # Visualize checkerboard axes in 2D image
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        if corners2 is None:
            logger.warning(f"Skipping {cam_name} due to corner extraction failure.")
            continue

        rvec, tvec_mm = compute_pose(objp, corners2, mtx, dist) # rvec, tvec(mm) -> w2c
        # Apply unit scale for saving (30 mm = 1 unit)
        tvec = tvec_mm * SCENE_UNIT_PER_MM
        T_board_to_cami, T_cami_to_board = compose_transform(rvec, tvec) # w2c, c2w (unit)
        
        Identity = T_board_to_cami @ T_cami_to_board
        print("Check Identity: ", Identity)
        
        # T_cami_to_board = np.linalg.inv(T_board_to_cami)
        
        if i == 0:
            cam_poses[cam_name] = np.eye(4).tolist()
            T_cam0_to_board = T_cami_to_board
            T_board_to_cam0 = T_board_to_cami # position of checkerboard in cam0 frame
        else:
            T_cam0_to_cami = T_board_to_cami @ T_cam0_to_board
            cam_poses[cam_name] = T_cam0_to_cami.tolist()

        # Reprojection error calculation
        # Reprojection error is calculated with the original tvec in mm units (same unit as objp)
        imgpoints2, _ = cv.projectPoints(objp, rvec, tvec_mm, mtx, dist)
        error = cv.norm(corners2, imgpoints2, cv.NORM_L2) / len(imgpoints2)
        cam_errors[cam_name] = error
        logger.info(f"[{cam_name}] Reprojection error: {error:.4f}")

        # Visualize checkerboard axes in 2D image
        imgpts, _ = cv.projectPoints(axis, rvec, tvec_mm, mtx, dist)
        img_vis = draw_axes(img, corners2, imgpts)
        img_resized = cv.resize(img_vis, window_size)
        cv.imshow(f'Pose - {cam_name}', img_resized)
        cv.waitKey(500)
        cv.destroyAllWindows()

    mean_error = np.mean(list(cam_errors.values()))
    logger.info(f"\nMean reprojection error across all cameras: {mean_error:.4f}")

    extrinsics_path = os.path.join(base_dir, "extrinsics.json")
    with open(extrinsics_path, "w") as f:
        json.dump(cam_poses, f, indent=4)
    logger.info(f"Extrinsics saved to: {extrinsics_path}")

    # error in pose visualization -> check poses in COLMAP GUI after running convert_to_COLMAP_fmt.py
    # plot_camera_poses(cam_poses, board_pose=T_board_to_cam0.tolist())    
    
if __name__ == "__main__":
    main()

