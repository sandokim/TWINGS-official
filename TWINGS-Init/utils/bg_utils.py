import os
from pathlib import Path
from PIL import Image
import numpy as np

def ensure_white_background(image_path: str, cache_dir: str = None) -> str:
    """Return a temporary RGB PNG with white background by compositing RGBA images (e.g., PNG).

    - Input: original image path
    - Output: temporary image path with white background (return original path if already RGB)
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return image_path

    # return original path if not RGBA
    if img.mode != 'RGBA':
        return image_path

    # compositing white background
    im_data = np.array(img.convert('RGBA'))
    bg = np.array([1, 1, 1], dtype=np.float32)
    norm = im_data.astype(np.float32) / 255.0
    arr = norm[:, :, :3] * norm[:, :, 3:4] + bg * (1.0 - norm[:, :, 3:4])
    out = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), 'RGB')

    # save _wb file in cache directory (default: separate cache instead of image folder)
    p = Path(image_path)
    if cache_dir is None:
        cache_dir = str(p.parent / 'train_wb')
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    out_path = os.path.join(cache_dir, p.stem + '_wb' + p.suffix)
    try:
        out.save(out_path)
        return out_path
    except Exception:
        return image_path