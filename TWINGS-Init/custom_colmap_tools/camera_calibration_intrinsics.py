import cv2 as cv
import numpy as np
import os
import glob
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s: %(message)s')

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'camera_calibration_intrinsics.log'), mode='w')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

CHECKERBOARD = (7, 4)
real_world_distance = 30  # mm units

window_size = (640, 480)

def draw_axes(img, corners, imgpts): # BGR -> imgpts[2]:R, imgpts[1]:G, imgpts[0]:B
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(
        img, corner, tuple(int(i) for i in imgpts[2].ravel()), (255, 0, 0), 5
    )  # R -> x axis
    img = cv.line(
        img, corner, tuple(int(i) for i in imgpts[1].ravel()), (0, 255, 0), 5
    )  # G -> y axis
    img = cv.line(
        img, corner, tuple(int(i) for i in imgpts[0].ravel()), (0, 0, 255), 5
    )  # B -> z axis
    return img

def sort_key(filename):
    # Extract only the number part from the filename
    base_name = os.path.basename(filename)
    prefix, suffix = base_name.split("_")
    # Convert prefix and suffix to numbers
    prefix_num = int(prefix)
    suffix_num = int(suffix.split(".")[0])  # remove extension
    return prefix_num, suffix_num

# Find checkerboard corners function
def find_checkerboard_corners(checkerboard_path):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # EPS: specified accuracy, MAX_ITER: specified number of iterations => stop if either is satisfied / 30 iterations, 0.001 accuracy
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * real_world_distance
    
    extensions = ['jpg', 'png']
    checkerboards = [file for ext in extensions for file in glob.glob(os.path.join(checkerboard_path, f"*.{ext}"))]

    num = 0
    for fname in checkerboards:
        img = cv.imread(fname)
        H, W, _ = img.shape
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert image to gray scale

        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None) # input checkerboard pattern CHECKERBOARD = (7,4)
        if ret == True:
            num += 1
            logger.info(f"Find ChessboardCorner --> {os.path.basename(fname)}")
            objpoints.append(objp)
            # Refine pixel coordinates for given 2D points
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # input 11x11 size to detect corners with maximum accuracy
            imgpoints.append(corners2)
            img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            # Resize the image if needed
            img_resized = cv.resize(img, window_size)
            cv.imshow('chessboard', img_resized)
            cv.waitKey(100)
        cv.destroyAllWindows()

    logger.info(f'Total num of chessboards: {len(checkerboards)}')
    logger.info(f'Total num of corner found chessboards: {num}')
    return objpoints, imgpoints


def find_checkerboard_axes(checkerboard_path, mtx, dist):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    * real_world_distance
    )
    extensions = ['jpg', 'png']
    checkerboards = [file for ext in extensions for file in glob.glob(os.path.join(checkerboard_path, f"*.{ext}"))]

    axis = np.float32(
    [
        [real_world_distance, 0, 0],
        [0, real_world_distance, 0],
        [0, 0, real_world_distance],
    ]
    ).reshape(-1, 3)
    for fname in checkerboards:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret == True:
            # Refine pixel coordinates for given 2D points
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors.
            retval, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw_axes(img, corners2, imgpts)

            # Resize the image if needed
            img_resized = cv.resize(img, window_size)
            cv.imshow('chessboard', img_resized)
            cv.waitKey(100)

        cv.destroyAllWindows()
    
def calibrate_camera(objpoints, imgpoints, gray):
    # fix k3 coefficient to improve calibration stability
    flags = cv.CALIB_FIX_K3
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
    return mtx, dist, rvecs, tvecs


def main():
    calib_list = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    # calib_list = ["cam6"] # delete 2 images / calib z-axis coords err
    for cam in calib_list:
        base_path = "multicam/build/Desktop_Qt_6_9_0_MSVC2022_64bit-Release/scene/7_camera_calib_2/checkerboard"
        checkerboard_path = os.path.join(base_path, cam)
        intrinsics_path = os.path.join(base_path, "multicapture", cam, "intrinsics.json")
        logger.info(f'checkerboard path: {checkerboard_path}')
        logger.info(f'save intrinsics_path: {intrinsics_path}')
        objpoints, imgpoints = find_checkerboard_corners(checkerboard_path)
        # set image shape to the resolution of the first image
        image = cv.imread(glob.glob(os.path.join(checkerboard_path, "*.jpg"))[0])
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        H, W, _ = image.shape
        
        mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray)
        find_checkerboard_axes(checkerboard_path, mtx, dist)

        # focal length 
        f_x = mtx[0, 0]
        fov_x = 2 * np.arctan((W / 2) / f_x)
        f_y = mtx[1, 1]
        fov_y = 2 * np.arctan((H / 2) / f_y)
            
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
            total_error = mean_error / len(objpoints)
        logger.info("calibration_error: {}".format(total_error))
        
        
        calibration_data = {
            "resolution": [H, W],
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "mtx": mtx.tolist(), # intrinsic matrix
            "dist": dist.tolist(),
            "calibration_error": total_error
        }
        logger.info(calibration_data)
        
        with open(intrinsics_path, "w") as json_file:
            json.dump(calibration_data, json_file, indent=4)

if __name__ == "__main__":
    main()
    