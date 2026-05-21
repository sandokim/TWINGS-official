import json
import os
import sys
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

# Register COLMAP database.py path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.isdir(os.path.join(CURRENT_DIR, "colmap")) else os.path.dirname(CURRENT_DIR)
COLMAP_PY_PATH = os.path.join(PROJECT_ROOT, "colmap", "scripts", "python")
if COLMAP_PY_PATH not in sys.path:
    sys.path.append(COLMAP_PY_PATH)
from database import COLMAPDatabase, blob_to_array

def load_intrinsics(path):
    with open(path, 'r') as f:
        data = json.load(f)
    mtx = np.array(data["mtx"])
    dist = np.array(data.get("dist", [0, 0, 0, 0, 0])).flatten()
    h, w = data.get("resolution", (None, None))
    if h is None or w is None:
        raise ValueError("resolution not provided in intrinsics.json")
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    return w, h, fx, fy, cx, cy, mtx, dist

def load_extrinsics(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_cmd(colmap_exe, args, desc, return_output=False):
    cmd = [colmap_exe] + args
    print(f"\U0001F680 Running COLMAP {desc}...")
    print("\U0001F6E0️ Command:", " ".join(cmd))
    result = subprocess.run(" ".join(cmd), capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"❌ COLMAP {desc} 실패:")
        print(result.stderr)
    else:
        print(f"✅ COLMAP {desc} 성공")
        print(result.stdout)
    return result.stdout if return_output else None

def run_feature_extractor(colmap_exe, database_path, image_path):
    run_cmd(colmap_exe, [
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.single_camera", "0",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "8192",
    ], "feature_extractor")

def run_exhaustive_matcher(colmap_exe, database_path):
    run_cmd(colmap_exe, [
        "exhaustive_matcher",
        "--database_path", database_path
    ], "exhaustive_matcher")

def run_point_triangulator(colmap_exe, database_path, image_path, input_path, output_path):
    run_cmd(colmap_exe, [
        "point_triangulator",
        "--database_path", database_path,
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--Mapper.tri_ignore_two_view_tracks", "0" # 2-view에서만 match되는 것도 포함
    ], "point_triangulator")

def export_model_to_txt(colmap_exe, input_path):
    run_cmd(colmap_exe, [
        "model_converter",
        "--input_path", input_path,
        "--output_path", input_path,
        "--output_type", "TXT"
    ], "model_converter (BIN → TXT in-place)")
    
def export_model_to_bin(colmap_exe, input_path):
    run_cmd(colmap_exe, [
        "model_converter",
        "--input_path", input_path,
        "--output_path", input_path,
        "--output_type", "BIN"
    ], "model_converter (TXT → BIN in-place)")

def inspect_database(db_path):
    db = COLMAPDatabase.connect(db_path)
    
    print("📸 Registered images:")
    images = db.execute("SELECT image_id, name FROM images").fetchall()
    for img_id, name in images:
        print(f"  ID: {img_id}, Name: {name}")

    print("\n🔍 Keypoints per image:")
    for img_id, _ in images:
        row = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (img_id,)).fetchone()
        if row:
            keypoints = blob_to_array(row[0], dtype=np.float32, shape=(-1, 2))
            print(f"  Image ID {img_id}: {len(keypoints)} keypoints")
        else:
            print(f"  Image ID {img_id}: 0 keypoints")

    match_pairs = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"\n🔗 Total match pairs: {match_pairs}")
    
    db.close()
    
def check_two_view_geometries(db_path):
    db = COLMAPDatabase.connect(db_path)
    count = db.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    print(f"📊 two_view_geometries count: {count}")
    db.close()
    
def run_pose_prior_mapper(colmap_exe, database_path, image_path, input_path, output_path):
    run_cmd(colmap_exe, [
        "pose_prior_mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.tri_ignore_two_view_tracks", "0"
    ], "pose_prior_mapper")
    
def run_undistorter(colmap_exe, images_path, triangulated_path, output_path):
    run_cmd(colmap_exe, [
        "image_undistorter",
        "--image_path", images_path,
        "--input_path", triangulated_path,
        "--output_path", output_path
    ], "image_undistorter")
    
def run_dense_reconstruction(colmap_exe, images_path, triangulated_path, dense_workspace_path):
    run_cmd(colmap_exe, [
        "image_undistorter",
        "--image_path", images_path,
        "--input_path", triangulated_path,
        "--output_path", dense_workspace_path
    ], "image_undistorter")

    run_cmd(colmap_exe, [
        "patch_match_stereo",
        "--workspace_path", dense_workspace_path
    ], "patch_match_stereo")

    run_cmd(colmap_exe, [
        "stereo_fusion",
        "--workspace_path", dense_workspace_path,
        "--output_path", dense_workspace_path + "/fused.ply"
    ], "stereo_fusion")
    

def convert_to_colmap(base_path, calib_path, colmap_exe):
    extrinsics = load_extrinsics(os.path.join(calib_path, "extrinsics.json"))

    database_path = os.path.join(base_path, "database.db")
    images_path = os.path.join(base_path, "images")
    
    cam_names = sorted([
        cam_name for cam_name in os.listdir(images_path)
        if cam_name.endswith(".jpg")
    ])
    
    if os.path.exists(database_path):
        os.remove(database_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    db.execute("ALTER TABLE images ADD COLUMN prior_qw REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_qx REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_qy REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_qz REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_tx REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_ty REAL")
    db.execute("ALTER TABLE images ADD COLUMN prior_tz REAL")

    manually_created_sparse_path = os.path.join(base_path, "manually/created/sparse", "0")
    triangulated_path = os.path.join(base_path, "triangulated", "sparse", "0")
    os.makedirs(triangulated_path, exist_ok=True)
    os.makedirs(manually_created_sparse_path, exist_ok=True)

    cameras_path = os.path.join(manually_created_sparse_path, "cameras.txt")
    images_txt_path = os.path.join(manually_created_sparse_path, "images.txt")
    points3D_path = os.path.join(manually_created_sparse_path, "points3D.txt")

    existing_cams = {}
    camera_id = 1
    
    with open(cameras_path, "w") as cam_file, open(images_txt_path, "w") as img_file:
        cam_file.write("# Camera list with one line of data per camera:\n")
        cam_file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        img_file.write("# Image list with two lines of data per image:\n")
        img_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        img_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for cam_name in cam_names:
            intr_path = os.path.join(calib_path, cam_name.split("_")[0], "intrinsics.json") # cam0/intrinsics.json
            w, h, fx, fy, cx, cy, _, dist = load_intrinsics(intr_path)
            if dist.size < 4:
                raise ValueError(f"At least 4 distortion coefficients are required (k1, k2, p1, p2). 현재: {dist}")
            k1, k2, p1, p2 = dist[:4]
            params_tuple = (w, h, fx, fy, cx, cy, k1, k2, p1, p2)

            if params_tuple in existing_cams:
                cam_id = existing_cams[params_tuple]
            else:
                cam_id = camera_id
                model = 4  # OPENCV
                db.add_camera(
                    model=model,
                    width=w, height=h,
                    params=np.array([fx, fy, cx, cy, k1, k2, p1, p2]),
                    prior_focal_length=True,
                    camera_id=cam_id
                )
                cam_file.write(f"{cam_id} OPENCV {w} {h} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f} {k1:.12f} {k2:.12f} {p1:.12f} {p2:.12f}\n")
                existing_cams[params_tuple] = cam_id
                camera_id += 1

            T = np.array(extrinsics[cam_name])
            R_wc = T[:3, :3]
            t_wc = T[:3, 3]
            
            qvec = R.from_matrix(R_wc).as_quat()
            qw, qx, qy, qz = qvec[3], qvec[0], qvec[1], qvec[2]

            image_name = cam_name

            # ✅ DB에 먼저 넣고 image_id 받아옴
            db.execute(
                "INSERT INTO images (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_name, cam_id, qw, qx, qy, qz, t_wc[0], t_wc[1], t_wc[2])
            )
            image_id = db.execute("SELECT image_id FROM images WHERE name = ?", (image_name,)).fetchone()[0]

            # ✅ 정확한 image_id를 images.txt에 기록
            img_file.write(f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} {t_wc[0]:.12f} {t_wc[1]:.12f} {t_wc[2]:.12f} {cam_id} {image_name}\n")
            img_file.write("\n")

    with open(points3D_path, "w") as p3d_file:
        p3d_file.write("# 3D point list with one line of data per point:\n")
        p3d_file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        p3d_file.write("# Number of points: 0, mean track length: 0\n")

    db.commit()
    db.close()

    print("✅ database.db / cameras.txt / images.txt / points3D.txt 구성 완료")
    export_model_to_bin(colmap_exe, manually_created_sparse_path) # txt → bin 변환
    
    # To reconstruct a sparse map, you first have to recompute features from the images of the known camera poses as follows:
    run_feature_extractor(colmap_exe, database_path, images_path)
        
    # If your known camera intrinsics have large distortion coefficients, you should now manually copy the parameters from your cameras.txt to the database, such that the matcher can leverage the intrinsics. Modifying the database is possible in many ways, but an easy option is to use the provided scripts/python/database.py script. Otherwise, you can skip this step and simply continue as follows:
    run_exhaustive_matcher(colmap_exe, database_path)
    
    # ✅ db에서 Registered images 확인 / feature_extractor로 얻은 이미지 ID별 keypoints 확인 / feature_macther로 얻은 Total match pairs 확인
    inspect_database(database_path)
    check_two_view_geometries(database_path)
    
    run_pose_prior_mapper(colmap_exe, database_path, images_path, input_path=manually_created_sparse_path, output_path=triangulated_path)

    run_point_triangulator(colmap_exe, database_path, images_path, input_path=manually_created_sparse_path, output_path=triangulated_path)
    
    # ✅ 모델 내보내기
    export_model_to_txt(colmap_exe, triangulated_path)
    
    # undistort images
    undistorted_path = os.path.join(base_path, "undistorted")
    os.makedirs(undistorted_path, exist_ok=True)
    run_undistorter(colmap_exe, images_path, triangulated_path, output_path=undistorted_path)
    
    export_model_to_txt(colmap_exe, os.path.join(undistorted_path, "sparse"))
    
    # Note that the sparse reconstruction step is not necessary in order to compute a dense model from known camera poses. Assuming you computed a sparse model from the known camera poses, you can compute a dense model as follows:
    dense_workspace_path = os.path.join(base_path, "dense", "workspace")
    os.makedirs(dense_workspace_path, exist_ok=True)
    run_dense_reconstruction(colmap_exe, images_path, triangulated_path, dense_workspace_path)
   
    
    

if __name__ == "__main__":
    base_path = r"C:/Users/maila/KHS/camera_calibration/multicam/build/Desktop_Qt_6_9_0_MSVC2022_64bit-Release/scene/7_camera_face_2_3" # C:/Users/Kang/Desktop/camera_calibration/multicam/build/Desktop_Qt_6_9_0_MSVC2022_64bit-Release/scene/myface
    calib_path = r"C:/Users/maila/KHS/camera_calibration/multicam/build/Desktop_Qt_6_9_0_MSVC2022_64bit-Release/scene/7_camera_calib_2/checkerboard/multicapture"
    colmap_exe = r"C:/Users/maila/colmap-x64-windows-cuda/COLMAP.bat" # C:/Users/Kang/colmap-x64-windows-cuda/COLMAP.bat or C:/Users/maila/KHS/COLMAP-3.9.1-windows-cuda/COLMAP.bat        
    convert_to_colmap(base_path, calib_path, colmap_exe)


'''
https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses

Why undistortion?

Undistortion prior to dense reconstruction is also just a way to enable efficient patch matching across multiple views without undistorting the coordinates on the fly many times during dense reconstruction. Some individual steps require undistortion from image pixels to ray directions, e.g., for triangulation, etc. 
'''