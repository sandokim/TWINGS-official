# TWINGS-Init 

Set up a Python 3.11 environment for `cv2.sfm`, with **CUDA 12.1** for MASt3R + `faiss-gpu`.

- `cv2.sfm` is part of OpenCV contrib (`OPENCV_EXTRA_MODULES`) and is not included in default OpenCV builds.
- To use `cv2.sfm`, OpenCV must be built from source.

## 1. Build and Install `cv2.sfm` from Source

The PyPI `opencv-python` / `opencv-contrib-python` wheels do not include the `sfm` module (excluded by default due to dependencies such as Ceres, glog, and gflags).  

To use `cv2.sfm` in the Ubuntu `TWINGS-Init` conda environment, install Linux-built OpenCV libraries (`.so`) in that environment.

### 1.1 Create and activate environment

```bash
cd TWINGS/TWINGS-Init
conda create -n TWINGS-Init python=3.11 -y
conda activate TWINGS-Init
git submodule update --init --recursive
```

### 1.2 Install dependencies (conda-forge)

```bash
conda install -y -c conda-forge cmake ninja pkg-config \
  numpy eigen gflags glog ceres-solver suitesparse tbb qt-main
```

### 1.3 Clone OpenCV sources

```bash
SRC_ROOT="/mai_nas/KHS/TWINGS/TWINGS-Init/src" # change this to your local path
rm -rf "$SRC_ROOT" && mkdir -p "$SRC_ROOT" && cd "$SRC_ROOT"
git clone -b 4.11.0 https://github.com/opencv/opencv.git
git clone -b 4.11.0 https://github.com/opencv/opencv_contrib.git
```

### 1.4 Configure build directory and Python site-packages path

```bash
BUILD_DIR="$SRC_ROOT/opencv/build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

PY_SITE=$(python - <<'PY'
import sysconfig; print(sysconfig.get_paths()["purelib"])
PY
)
echo "$PY_SITE"
```

### 1.5 Configure OpenCV with CMake

```bash
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DOPENCV_EXTRA_MODULES_PATH="$SRC_ROOT/opencv_contrib/modules" \
  -DBUILD_opencv_sfm=ON \
  -DBUILD_opencv_world=OFF \
  -DBUILD_opencv_python3=ON \
  -DWITH_EIGEN=ON \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DWITH_VA=OFF -DWITH_VA_INTEL=OFF \
  -DWITH_FFMPEG=OFF \
  -DVIDEOIO_ENABLE_PLUGINS=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
  -DPYTHON3_EXECUTABLE="$(which python)" \
  -DPYTHON3_PACKAGES_PATH="$PY_SITE" \
  -DBUILD_SHARED_LIBS=ON
```

### 1.6 Build and install

```bash
ninja -j"$(nproc)"
ninja install
```

### 1.7 Verify the Python loader

After install, verify that Python loads the OpenCV extension instead of a bare `cv2/` namespace package:

```bash
python - <<'PY'
import cv2
print("cv2.__file__:", cv2.__file__)
print("cv2.__version__:", cv2.__version__)
print("has IMREAD_COLOR:", hasattr(cv2, "IMREAD_COLOR"))
print("has sfm:", hasattr(cv2, "sfm"))
PY
```

Expected result:

- `cv2.__file__` points to `.../site-packages/cv2/__init__.py`
- `has IMREAD_COLOR: True`
- `has sfm: True`

If you instead see `cv2.__file__` as `None` or an error such as `AttributeError: module 'cv2' has no attribute 'IMREAD_COLOR'`, the OpenCV binary was installed but the Python loader files were not copied into `site-packages/cv2`. Copy them from the build tree:

```bash
cp -r "$BUILD_DIR/python_loader/cv2/"* "$PY_SITE/cv2/"
```

## 2. Install MASt3R and ASMK Dependencies

### 2.1 Install MASt3R

```bash
cd ../../..
pip install "numpy<2.0"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
cd utils/submodules/mast3r
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
pip uninstall -y opencv-python opencv-python-headless # Remove PyPI OpenCV so the custom sfm build is used
```

### 2.2 Compile and install ASMK + Faiss

```bash
pip install cython
cd ../../..
cd utils/submodules/asmk/cython/
cythonize *.pyx
cd ..
conda install -c conda-forge faiss-gpu # MASt3R uses Faiss to store correspondences; `faiss-gpu` supports CUDA 11.4/12.1 (Linux x86-64). If unavailable, use `pip install faiss-cpu`.
```

### 2.3 Install transformers & open3d & plyfile

```bash
pip install transformers==4.53.0
pip install open3d==0.19.0
pip install plyfile==1.1.2
```

## 3. Run TWINGS-Init

### 3.1 Generate initial point cloud

```bash
cd TWINGS/TWINGS-Init
python main.py
```

Note: If preprocessing takes much longer than expected, the TPS3D step is CPU-bound and can run faster or slower depending on the CPU, NumPy/BLAS backend, and overall system configuration; in such cases, running TWINGS-Init on a local workstation with a faster CPU/linear-algebra stack may be preferable.