import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

def camera_center_from_extrinsics(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t.reshape(3, 1)).reshape(3,)

def visualize_point_cloud(points_3d: np.ndarray,
                          colors: np.ndarray,
                          train_Es: dict, 
                          test_Es: dict,
                          title: str,
                          all_Es: dict = None,
                          near_Es: dict = None,
                          pseudo_Es: dict = None,
                          focus_point: np.ndarray = None,
                          draw_focus_rays: bool = False):
    """
    3D point cloud and camera poses (train/test/near/other) visualization.

    Args:
        points_3d: (N, 3) triangulated 3D points (world coordinates)
        colors:    (N, 3) uint8 RGB values (0-255)
        train_Es: Dict[int, (3,4)] train set camera [R|t]
        test_Es:  Dict[int, (3,4)] test set camera [R|t]
        all_Es:   Dict[int, (3,4)] all cameras [R|t] (train/test included)
        title: string figure title
        near_Es:  Dict[int, (3,4)] current view nearby camera set (selected) — arrows are not displayed
        pseudo_Es: Dict[int, (3,4)] pseudo camera set
        focus_point: (3,) focus point (world coordinates)
        draw_focus_rays: bool draw focus rays (optional)
    """
    all_Es = all_Es or {}
    near_Es = near_Es or {}
    pseudo_Es = pseudo_Es or {}

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- point cloud ---
    if points_3d is not None and len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   c=colors / 255.0, s=1, label="3D Points")

        mean_xyz = np.mean(points_3d, axis=0)
        half_range = 15  # llff:30 / mipnerf360,DTU:5 / blender:5.0 / custom:300 etc.
        ax.set_xlim(mean_xyz[0] - half_range, mean_xyz[0] + half_range)
        ax.set_ylim(mean_xyz[1] - half_range, mean_xyz[1] + half_range)
        ax.set_zlim(mean_xyz[2] - half_range, mean_xyz[2] + half_range)

        cam_axis_len = max(half_range * 0.05, 0.5)
    else:
        cam_axis_len = 1.0

    # color setting
    TRAIN_COLOR = 'red'
    TEST_COLOR  = 'blue'
    NEAR_COLOR  = 'green' 
    OTHER_COLOR = 'gray'
    PSEUDO_COLOR = 'purple'

    # other calculation: train/test/near excluded
    train_ids = set(train_Es.keys())
    test_ids  = set(test_Es.keys())
    near_ids  = set(near_Es.keys())
    other_Es = {k: v for k, v in all_Es.items()
                if k not in train_ids and k not in test_ids and k not in near_ids}

    def _plot_cam_dict(cam_dict, color, label,
                       alpha=1.0, text_color='black', marker='o', size=20,
                       axis_scale=1.0, draw_arrow=True):
        """
        draw_arrow=False means not drawing camera direction arrows.
        axis_scale means meaningful only when draw_arrow=True.
        """
        if not cam_dict:
            return None
        xs, ys, zs = [], [], []
        for img_id, E in cam_dict.items():
            R, t = E[:, :3], E[:, 3]
            center = camera_center_from_extrinsics(R, t)
            z_axis = R.T @ np.array([0, 0, 1.0])
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

            xs.append(center[0]); ys.append(center[1]); zs.append(center[2])
            ax.text(center[0], center[1], center[2], f'{img_id}',
                    color=text_color, fontsize=7)

            if draw_arrow:
                ax.quiver(center[0], center[1], center[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          length=cam_axis_len * axis_scale,
                          color=color, alpha=alpha, arrow_length_ratio=0.2)

        return ax.scatter(xs, ys, zs, c=color, marker=marker, s=size, label=label, alpha=alpha)

    # order: train, test, near(only markers), other
    _plot_cam_dict(train_Es, TRAIN_COLOR, "Train Cameras")
    _plot_cam_dict(test_Es,  TEST_COLOR,  "Test Cameras")
    _plot_cam_dict(near_Es,  NEAR_COLOR,  "Near Cameras",
                   marker='^', size=60, draw_arrow=False, alpha=1.0)  # axis_scale removed, arrows X
    _plot_cam_dict(other_Es, OTHER_COLOR, "Other Cameras")
    _plot_cam_dict(pseudo_Es, PSEUDO_COLOR, "Pseudo Cameras", alpha=0.5)

    # --- focus point display (optional) ---
    if focus_point is not None:
        ax.scatter([focus_point[0]], [focus_point[1]], [focus_point[2]],
                   marker='*', s=250, c='gold', edgecolors='k', linewidths=0.5, label='Focus Point')
        # rays (optional): connect each camera center to focus
        if draw_focus_rays:
            def _draw_rays(cam_dicts: List[dict], ray_alpha=0.2):
                for d in cam_dicts:
                    for _, E in d.items():
                        R, t = E[:, :3], E[:, 3]
                        C = camera_center_from_extrinsics(R, t)
                        ax.plot([C[0], focus_point[0]],
                                [C[1], focus_point[1]],
                                [C[2], focus_point[2]],
                                linestyle='--', linewidth=0.5, alpha=ray_alpha, color='black')
            _draw_rays([train_Es, pseudo_Es])


    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)

    # equal aspect
    try:
        ax.set_box_aspect((ax.get_xlim()[1]-ax.get_xlim()[0],
                           ax.get_ylim()[1]-ax.get_ylim()[0],
                           ax.get_zlim()[1]-ax.get_zlim()[0]))
    except Exception:
        pass

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    
def visualize_point_tracks_grid(
    point_tracks: Dict[Any, Dict[Any, Tuple[float, float]]],
    rgb_images: Dict[Any, np.ndarray],
    ref_image_id: Any,
    image_ids: Optional[List[Any]] = None,
    num_show_ge3: int = 5,
    num_show_eq2: int = 5,
    ref_point_size: int = 50,
    pt_size_ge3: int = 50,
    pt_size_eq2: int = 50,
    line_width_ge3: float = 2.0,
    line_width_eq2: float = 2.0,
    alpha: float = 0.95,
    random_pick: bool = True,
    seed: Optional[int] = None,
):
    """
    ref image based, visualize multiple images together matched as a horizontal concatenation on one screen.
    - point_tracks: {track_id -> {image_id -> (x, y)}}
    - rgb_images: {image_id -> (H, W, 3) float [0,1]}
    - ref_image_id: reference image id (key)
    - image_ids: image id order to display (if given, follow the order). if None, use rgb_images.keys() order
    """
    if image_ids is None:
        image_ids = list(rgb_images.keys())
    # ref comes first by sorting
    image_ids = [i for i in image_ids if i in rgb_images]
    if ref_image_id in image_ids:
        image_ids = [ref_image_id] + [i for i in image_ids if i != ref_image_id]
    else:
        image_ids = [ref_image_id] + image_ids

    # collect images and pad to the same height
    imgs_uint8 = []
    widths, heights = [], []
    for img_id in image_ids:
        if img_id not in rgb_images:
            continue
        img = rgb_images[img_id]
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        imgs_uint8.append(img_uint8)
        widths.append(w)
        heights.append(h)

    if len(imgs_uint8) == 0:
        print("[Warn] visualize_point_tracks_grid: no images to display.")
        return

    max_h = max(heights)
    padded_imgs = []
    for im in imgs_uint8:
        h, w = im.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            im_pad = np.concatenate([im, pad], axis=0)
        else:
            im_pad = im
        padded_imgs.append(im_pad)

    # concatenate horizontally and calculate x-offset
    canvas = np.concatenate(padded_imgs, axis=1)
    x_offsets = np.cumsum([0] + [im.shape[1] for im in padded_imgs[:-1]]).tolist()

    # use only tracks including ref
    tracks = [(tid, tr) for tid, tr in point_tracks.items() if ref_image_id in tr]
    if len(tracks) == 0:
        print("[Warn] visualize_point_tracks_grid: point_tracks is empty.")
        return
    
    # separate categories: 3+ views, exactly 2 views
    tracks_ge3 = [(tid, tr) for tid, tr in tracks if len(tr) >= 3]
    tracks_eq2 = [(tid, tr) for tid, tr in tracks if len(tr) == 2]

    total_ge3 = len(tracks_ge3)
    total_eq2 = len(tracks_eq2)

    # sample maximum num_show for each category
    def pick_items(items, k):
        if k <= 0 or len(items) <= k:
            return items
        if random_pick:
            rng = np.random.default_rng(seed)
            sel_idx = rng.choice(len(items), size=k, replace=False)
            return [items[i] for i in sel_idx]
        else:
            idx = np.round(np.linspace(0, len(items) - 1, k)).astype(int)
            return [items[i] for i in idx]

    tracks_eq3 = [(tid, tr) for tid, tr in tracks_ge3 if len(tr) == 3]
    tracks_ge4 = [(tid, tr) for tid, tr in tracks_ge3 if len(tr) >= 4]
    tracks_eq3_show = pick_items(tracks_eq3, num_show_ge3)
    tracks_ge4_show = pick_items(tracks_ge4, num_show_ge3)
    tracks_eq2_show = pick_items(tracks_eq2, num_show_eq2)

    # visualization
    plt.figure(figsize=(min(32, max(16, canvas.shape[1] / 120)), max(6, max_h / 120)))
    plt.imshow(canvas)
    RED = (220/255, 0/255, 0/255, 1.0)
    BLUE = (0/255, 122/255, 255/255, 1.0)
    GREEN = (0/255, 200/255, 83/255, 1.0)

    # ref offset
    try:
        ref_col_idx = image_ids.index(ref_image_id)
        ref_x_off = x_offsets[ref_col_idx]
    except ValueError:
        ref_x_off = 0

    # first exactly 3 views (RED)
    for (track_id, track) in tracks_eq3_show:
        pts = []
        for img_idx, img_id in enumerate(image_ids):
            if img_id not in track:
                continue
            x, y = track[img_id]
            off = x_offsets[img_idx]
            pts.append((x + off, y, img_id))

        # draw points (ref is larger)
        for x, y, img_id in pts:
            sz = ref_point_size if img_id == ref_image_id else pt_size_ge3
            plt.scatter([x], [y], s=sz, c=[RED], marker='o', linewidths=0.0, alpha=alpha)

        # connect consecutive lines
        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            plt.plot([x0, x1], [y0, y1], '-', color=RED, linewidth=line_width_ge3, alpha=alpha * 0.9)

    # next 4+ views (BLUE)
    for (track_id, track) in tracks_ge4_show:
        pts = []
        for img_idx, img_id in enumerate(image_ids):
            if img_id not in track:
                continue
            x, y = track[img_id]
            off = x_offsets[img_idx]
            pts.append((x + off, y, img_id))

        for x, y, img_id in pts:
            sz = ref_point_size if img_id == ref_image_id else pt_size_ge3
            plt.scatter([x], [y], s=sz, c=[BLUE], marker='o', linewidths=0.0, alpha=alpha)

        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            plt.plot([x0, x1], [y0, y1], '-', color=BLUE, linewidth=line_width_ge3, alpha=alpha * 0.9)

    # next 2 views (green): connect ref -> next in order
    for (track_id, track) in tracks_eq2_show:
        pts = []
        for img_idx, img_id in enumerate(image_ids):
            if img_id not in track:
                continue
            x, y = track[img_id]
            off = x_offsets[img_idx]
            pts.append((x + off, y, img_id))

        for x, y, img_id in pts:
            sz = ref_point_size if img_id == ref_image_id else pt_size_eq2
            plt.scatter([x], [y], s=sz, c=[GREEN], marker='o', linewidths=0.0, alpha=alpha)

        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            plt.plot([x0, x1], [y0, y1], '-', color=GREEN, linewidth=line_width_eq2, alpha=alpha * 0.9)

    # draw separator lines between each image
    cur = 0
    for im in padded_imgs[:-1]:
        cur += im.shape[1]
        plt.plot([cur, cur], [0, max_h], color=(1, 1, 1, 0.5), linewidth=1.0)

    # display category counts in title
    title_text = (
        f"ref: {ref_image_id}  |  3+ view tracks: total {total_ge3}, shown {len(tracks_eq3_show) + len(tracks_ge4_show)}  |  "
        f"2-view tracks: total {total_eq2}, shown {len(tracks_eq2_show)}"
    )
    plt.title(title_text, fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()