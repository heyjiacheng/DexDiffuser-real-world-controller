import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import open3d as o3d
import cv2
from typing import List, Tuple, Dict, Any, Optional
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    SamModel, SamProcessor
)


def pcd_from_rgbd_cpu(color_img_o3d: o3d.geometry.Image,
                      depth_np: np.ndarray,
                      K: np.ndarray,
                      depth_unit: str = "m",
                      flip_for_view: bool = True,
                      add_axis: bool = True,
                      axis_size: float = 0.1,
                      mask: Optional[np.ndarray] = None) -> list[o3d.geometry.Geometry]:
    """
    Build a point cloud (CPU) by manual back-projection, optionally add a coordinate axis.

    Args:
        color_img_o3d: Open3D Image (HxWx3 uint8) for color.
        depth_np:      HxW depth array; meters if depth_unit='m', millimeters if 'mm'.
        K:             3x3 camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]].
        depth_unit:    'm' (meters) or 'mm' (millimeters).
        flip_for_view: If True, flip Y/Z for nicer Open3D view.
        add_axis:      If True, include a coordinate frame mesh at the origin.
        axis_size:     Length scaling for the axis (in meters).
        mask:          Optional HxW boolean array; if provided, only pixels where mask is True are included.

    Returns:
        List of Open3D geometries (point cloud first, then axis if requested).
    """
    if K.shape != (3, 3):
        raise ValueError(f"Expected K as 3x3, got {K.shape}")
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth as HxW, got {depth_np.shape}")

    # Depth → meters
    if depth_unit == "mm":
        z = depth_np.astype(np.float32) / 1000.0
    elif depth_unit == "m":
        z = depth_np.astype(np.float32)
    else:
        raise ValueError("depth_unit must be 'm' or 'mm'")

    color_np = np.asarray(color_img_o3d)  # HxWx3 uint8
    if color_np.ndim != 3 or color_np.shape[2] < 3:
        raise ValueError(f"Color image must be HxWx3, got {color_np.shape}")

    H, W = z.shape
    if color_np.shape[0] != H or color_np.shape[1] != W:
        raise ValueError(f"RGB/Depth size mismatch: color {color_np.shape[:2]} vs depth {(H, W)}")

    # Validate mask if provided
    if mask is not None:
        if mask.shape != (H, W):
            raise ValueError(f"Mask shape mismatch: expected {(H, W)}, got {mask.shape}")
        if mask.dtype != bool:
            mask = mask.astype(bool)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Debug: print depth statistics
    print(f"[DEBUG] Depth shape: {z.shape}")
    print(f"[DEBUG] Depth range: min={np.min(z):.4f}, max={np.max(z):.4f}, mean={np.mean(z):.4f}")
    print(f"[DEBUG] Non-zero depth pixels: {np.sum(z > 0)}")

    valid = (z > 0) & (z < 1)  # tweak as needed (e.g., (z > 0) & (z < max_range))
    print(f"[DEBUG] Valid depth pixels (0 < z < 1): {np.sum(valid)}")

    # Apply mask if provided
    if mask is not None:
        print(f"[DEBUG] Mask shape: {mask.shape}")
        print(f"[DEBUG] Mask True pixels: {np.sum(mask)}")
        print(f"[DEBUG] Mask dtype: {mask.dtype}")
        valid = valid & mask
        print(f"[DEBUG] Valid pixels after mask: {np.sum(valid)}")

    # Back-project
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pts  = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)      # Nx3
    cols = (color_np[valid, :3].astype(np.float32) / 255.0).astype(np.float64)      # Nx3

    print(f"[DEBUG] Final point cloud size: {pts.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    if flip_for_view:
        pcd.transform([[1, 0, 0, 0],
                       [0,-1, 0, 0],
                       [0, 0,-1, 0],
                       [0, 0, 0, 1]])

    geoms: list[o3d.geometry.Geometry] = [pcd]
    if add_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        geoms.append(axis)
        
    return geoms

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def load_detection_model(model_id="google/owlvit-base-patch32"):
    proc = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)
    model.eval()
    return proc, model

def detect_objects(image_pil, target_objects, processor, model, confidence_threshold=0.1):
    text_labels = [[f"a photo of a {obj}" for obj in target_objects]]
    inputs = processor(text=text_labels, images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image_pil.height, image_pil.width)])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )
    result = results[0]

    boxes = result["boxes"]
    scores = result["scores"]
    if "text_labels" in result:
        labels = result["text_labels"]
    elif "labels" in result:
        labels = result["labels"]
    elif "class_labels" in result:
        labels = result["class_labels"]
    else:
        labels = [f"object_{i}" for i in range(len(boxes))]

    return {"boxes": boxes, "scores": scores, "labels": labels}

def select_best_detection(detections):
    boxes, scores, labels = detections["boxes"], detections["scores"], detections["labels"]
    if boxes is None or len(boxes) == 0:
        return None
    scores_np = scores.detach().cpu().numpy()
    boxes_np = boxes.detach().cpu().numpy()
    max_idx = int(np.argmax(scores_np))
    return {
        "box": boxes_np[max_idx],
        "score": float(scores_np[max_idx]),
        "label": labels[max_idx] if isinstance(labels, list) else str(labels[max_idx]),
    }

def draw_detections_on_image(image_pil, boxes, scores, labels, save_path):
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    for box_t, score_t, label in zip(boxes, scores, labels):
        box = [round(float(x), 2) for x in box_t.detach().cpu().tolist()]
        s = float(score_t.detach().cpu().item())
        cv2.rectangle(image_cv, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(image_cv, f"{label} ({s:.2f})",
                    (int(box[0]), max(0, int(box[1]-10))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(save_path, image_cv)
    return save_path

def load_sam(model_id="facebook/sam-vit-huge", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained(model_id).to(device)
    model.eval()
    proc = SamProcessor.from_pretrained(model_id)
    return proc, model, device

def segment_with_sam(image_pil, box_xyxy_pixels, sam_proc, sam_model, device):
    inputs = sam_proc(image_pil, input_boxes=[[list(box_xyxy_pixels)]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks_list = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu()
    )
    masks = masks_list[0][0].detach().cpu().numpy()  # (K,H,W)
    scores = outputs.iou_scores.detach().cpu().numpy()[0][0]  # (K,)
    best_idx = int(np.argmax(scores))
    return {
        "best_mask": masks[best_idx].astype(bool),
        "iou_score": float(scores[best_idx]),
        "all_masks": masks.astype(bool),
        "all_scores": scores.astype(np.float32),
    }

def save_mask_and_overlay(image_pil, mask_bool_hw, label, out_dir, alpha=0.3):
    mask_path = os.path.join(out_dir, f"mask_{label}.png")
    overlay_path = os.path.join(out_dir, f"segmentation_overlay_{label}.png")

    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    mask_u8 = (mask_bool_hw.astype(np.uint8) * 255)
    cv2.imwrite(mask_path, mask_u8)

    color = np.zeros_like(image_cv)
    color[mask_bool_hw] = (0, 255, 0)
    overlay = cv2.addWeighted(image_cv, 1.0 - alpha, color, alpha, 0.0)
    cv2.imwrite(overlay_path, overlay)
    return mask_path, overlay_path

def detect_seg_pipeline(
    image_path: str,
    out_root: str,
    target_objects: List[str],
    confidence_threshold: float = 0.1,
    detection_model_id="google/owlvit-base-patch32",
    sam_model_id="facebook/sam-vit-huge",) -> Optional[Dict[str, Any]]:

    image_pil = Image.open(image_path).convert("RGB")
    print(f"Processing image: {image_path}")
    print(f"Target objects: {target_objects}")

    det_dir = ensure_dir(os.path.join(out_root, "detection"))
    seg_dir = ensure_dir(os.path.join(out_root, "segmentation"))

    # Detection
    print("\n=== Step 1: Detection ===")
    det_proc, det_model = load_detection_model(detection_model_id)
    detections = detect_objects(image_pil, target_objects, det_proc, det_model, confidence_threshold)

    boxes, scores, labels = detections["boxes"], detections["scores"], detections["labels"]
    det_vis_path = draw_detections_on_image(image_pil, boxes, scores, labels,
                                            os.path.join(det_dir, "detected_objects.png"))
    print(f"Saved detection visualization: {det_vis_path}")

    best = select_best_detection(detections)
    if best is None:
        print("No detections found. Exiting.")
        return None

    print(f"Selected detection: {best['label']} (conf {best['score']:.3f})")

    # Segmentation
    print("\n=== Step 2: Segmentation ===")
    sam_proc, sam_model, device = load_sam(sam_model_id)
    seg = segment_with_sam(image_pil, best["box"], sam_proc, sam_model, device)
    best_mask, best_iou = seg["best_mask"], seg["iou_score"]

    mask_path, overlay_path = save_mask_and_overlay(image_pil, best_mask, best["label"], seg_dir)
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")

    # Save metadata JSON
    meta = {
        "label": best["label"],
        "box": best["box"].tolist(),
        "score": best["score"],
        "iou": best_iou,
        "mask_area": int(np.sum(best_mask)),
        "paths": {
            "detection_viz": det_vis_path,
            "mask": mask_path,
            "overlay": overlay_path,
        }
    }
    meta_path = os.path.join(out_root, "results.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return meta


if __name__ == "__main__":
    root_dir = 'gen_pcds/'
    rgb_path = os.path.join(root_dir, '1_color.png')
    depth_path = os.path.join(root_dir, '1_depth_aligned_rgb.npy')
    cam_k_path = os.path.join(root_dir, '1_camerainfo.npy')
    TARGET_OBJECTS = ["spatula"]
    CONFIDENCE_THRESHOLD = 0.1

    # perform detection and segmentation
    results = detect_seg_pipeline(
            image_path=rgb_path,
            out_root= root_dir,
            target_objects=TARGET_OBJECTS,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

    mask_path = results["paths"]["mask"]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(bool)

    color_o3d = o3d.io.read_image(rgb_path)          # HxWx3 uint8
    depth_np  = np.load(depth_path)                  # HxW depth (meters or mm)
    K         = np.load(cam_k_path)                    # 3x3 intrinsics

    # load the segmentation and convert to point cloud
    geoms = pcd_from_rgbd_cpu(
        color_img_o3d=color_o3d,
        depth_np=depth_np,
        K=K,
        depth_unit="mm",        # or "mm" if your depth is millimeters
        flip_for_view=False,   # keep camera-frame alignment so gripper poses match
        add_axis=True,
        axis_size=0.1,
        mask=mask              # apply segmentation mask to filter point cloud
    )

    # for the object point cloud, filter out outliers
    pcd = geoms[0]
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    geoms[0] = pcd

    # Extract point cloud as numpy array (N, 3) for sample_grasps input
    obj_pcd = np.asarray(pcd.points)
    print(f"Object point cloud shape: {obj_pcd.shape}")

    # Save the point cloud for later use
    np.save(os.path.join(root_dir, 'obj_pcd.npy'), obj_pcd)
    print(f"Saved object point cloud to: {os.path.join(root_dir, 'obj_pcd.npy')}")

    # visualize the point cloud and camera frame
    # o3d.visualization.draw_geometries(geoms)
