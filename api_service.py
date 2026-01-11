from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn
import torch

# Import functions from existing modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_pcds.main import detect_seg_pipeline, pcd_from_rgbd_cpu
import open3d as o3d
from run_real import DexDiffuser, load_config
from scipy.spatial.transform import Rotation as R
from visualize_camera_view import visualize_grasps_interactive
import threading


# Global DexDiffuser instance (loaded once at startup)
dex_diffuser = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: Initialize DexDiffuser model
    global dex_diffuser
    try:
        config_path = "configs/sample.yaml"
        cfg = load_config(config_path)
        cfg.model = load_config("configs/model/unet_grasp_bps.yaml")
        cfg.diffuser = load_config("configs/diffuser/ddpm.yaml")
        cfg.task = load_config("configs/task/grasp_gen_ur_dexgn_slurm.yaml")
        cfg.refinement = True

        dex_diffuser = DexDiffuser(cfg)
        print("DexDiffuser model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load DexDiffuser model: {e}")
        print("Service will start but grasp generation may fail")

    yield

    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="DexDiffuser Grasp Generation API",
    lifespan=lifespan
)

# ============================================================================
# Constants
# ============================================================================

# Point cloud processing
UNIT_CONVERSION_THRESHOLD = 5.0  # If max coord > this, assume mm instead of meters
MM_TO_M_SCALE = 1000.0

# Statistical outlier removal
OUTLIER_NB_NEIGHBORS_RGBD = 100
OUTLIER_STD_RATIO_RGBD = 2.0
OUTLIER_NB_NEIGHBORS_PCD = 20
OUTLIER_STD_RATIO_PCD = 2.0

# Plane segmentation
PLANE_DISTANCE_THRESHOLD = 0.005  # meters
PLANE_RANSAC_N = 3
PLANE_NUM_ITERATIONS = 1000

# Point cloud downsampling
VOXEL_SIZE = 0.005  # meters

# Visualization
VIZ_PORT = 8050
ROBOT_TYPE = 'allegro_right'

# Debug output
DEBUG_OUTPUT_DIR = "debug_output"


# ============================================================================
# Helper Functions
# ============================================================================

def process_point_cloud(
    obj_pcd: np.ndarray,
    output_dir: Path,
    pcd_o3d: o3d.geometry.PointCloud = None
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Process point cloud: check units, center, and save debug info.

    Args:
        obj_pcd: (N, 3) point cloud array
        output_dir: Directory to save debug files
        pcd_o3d: Optional Open3D point cloud for saving colored version

    Returns:
        Tuple of (processed_pcd, original_pcd_copy, obj_center, scale_factor)
    """
    # Check and convert units if necessary (mm to meters)
    max_val = np.max(np.abs(obj_pcd))
    print(f"Point Cloud Max Coord Value: {max_val}")
    if max_val > UNIT_CONVERSION_THRESHOLD:
        print("WARNING: Point cloud values suggest unit is MM. Converting to Meters.")
        obj_pcd = obj_pcd / MM_TO_M_SCALE

    # Save original point cloud for visualization (before centering)
    obj_pcd_in_camera_frame = obj_pcd.copy()

    # Save colored point cloud if provided
    if pcd_o3d is not None:
        ply_output_path = output_dir / "object_colored.ply"
        o3d.io.write_point_cloud(str(ply_output_path), pcd_o3d)
        print(f"Saved colored point cloud to: {ply_output_path}")

    # Print statistics before centering
    print(f"Point cloud statistics BEFORE normalization:")
    print(f"  - Mean: {obj_pcd.mean(axis=0)}")

    # Center the point cloud
    obj_pcd_center_in_camera = obj_pcd.mean(axis=0)
    obj_pcd_centered = obj_pcd - obj_pcd_center_in_camera

    # No scaling applied
    scale_factor = 1.0

    print(f"Point cloud statistics AFTER centering (no scale normalization):")
    print(f"  - Mean: {obj_pcd_centered.mean(axis=0)}")
    print(f"  - Scale factor: {scale_factor}")
    print(f"  - Object center in camera frame: {obj_pcd_center_in_camera}")
    print(f"  - Max distance from center: {np.sqrt((obj_pcd_centered ** 2).sum(axis=1)).max()}")

    return obj_pcd_centered, obj_pcd_in_camera_frame, obj_pcd_center_in_camera, scale_factor


def save_debug_files(
    output_dir: Path,
    camera_extrinsics: np.ndarray,
    best_grasp: np.ndarray,
    rgb_path: Path = None,
    depth_np: np.ndarray = None,
    mask_path: Path = None,
    overlay_path: Path = None
) -> None:
    """Save debug files to output directory."""
    output_dir.mkdir(exist_ok=True)

    # Save camera extrinsics and best grasp
    np.save(output_dir / "camera_extrinsics.npy", camera_extrinsics)
    np.save(output_dir / "best_grasp_base.npy", best_grasp)

    # Save RGB image if provided
    if rgb_path is not None and rgb_path.exists():
        import shutil
        shutil.copy(rgb_path, output_dir / "original_rgb.png")

    # Save depth visualization if provided
    if depth_np is not None:
        depth_vis = (depth_np / depth_np.max() * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "original_depth.png"), depth_vis)

    # Save segmentation files if provided
    if mask_path is not None and mask_path.exists():
        import shutil
        shutil.copy(mask_path, output_dir / "segmentation_mask.png")

    if overlay_path is not None and overlay_path.exists():
        import shutil
        shutil.copy(overlay_path, output_dir / "segmentation_overlay.png")

    print(f"Saved debug files to {output_dir}/")


def generate_and_transform_grasps(
    obj_pcd_centered: np.ndarray,
    obj_center_in_camera: np.ndarray,
    scale_factor: float,
    camera_extrinsics: np.ndarray,
    num_samples: int,
    dex_diffuser_model
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate grasps and transform to base frame.

    Returns:
        Tuple of (grasps_in_base, scores, best_grasp_index)
    """
    print("Generating grasps...")
    grasp_qt, scores = dex_diffuser_model.sample_grasps(obj_pcd_centered, num_samples=num_samples)

    print("Transforming grasps to robot base frame...")
    grasp_qt_base = transform_grasps_to_base_frame(
        grasp_qt=grasp_qt,
        obj_center_in_camera=obj_center_in_camera,
        scale_factor=scale_factor,
        camera_extrinsics=camera_extrinsics
    )

    best_grasp_index = int(np.argmax(scores))

    print(f"\n=== Best Grasp Selected ===")
    print(f"Best grasp index: {best_grasp_index}")
    print(f"Best score: {scores[best_grasp_index]:.4f}")
    print(f"Best grasp (23 dims): [qw, qx, qy, qz, x, y, z, joints...]")
    print(f"  Quaternion (w,x,y,z): {grasp_qt_base[best_grasp_index, 0:4]}")
    print(f"  Position (x,y,z): {grasp_qt_base[best_grasp_index, 4:7]}")
    print(f"===========================\n")

    return grasp_qt_base, scores, best_grasp_index


def launch_visualization(
    obj_pcd_in_camera: np.ndarray,
    grasps_base: np.ndarray,
    scores: np.ndarray,
    camera_extrinsics: np.ndarray,
    robot: str = ROBOT_TYPE,
    port: int = VIZ_PORT
) -> None:
    """Launch interactive visualization in background thread."""
    def run_visualization():
        try:
            print("\n" + "="*60)
            print("Starting interactive visualization in background...")
            print("="*60)
            visualize_grasps_interactive(
                obj_pcd_in_camera=obj_pcd_in_camera,
                all_grasps_base=grasps_base,
                all_scores=scores,
                camera_extrinsics=camera_extrinsics,
                robot=robot,
                port=port
            )
        except Exception as e:
            print(f"Visualization error: {e}")

    viz_thread = threading.Thread(target=run_visualization, daemon=True)
    viz_thread.start()
    print(f"Interactive visualization launched at http://127.0.0.1:{port}")


def create_grasp_response(
    grasps_base: np.ndarray,
    scores: np.ndarray,
    best_grasp_index: int,
    ply_path: Path
) -> GraspResponse:
    """Create API response with grasp data."""
    with open(ply_path, "rb") as f:
        ply_content = f.read()

    return GraspResponse(
        grasp_qt=grasps_base.tolist(),
        scores=scores.tolist(),
        best_grasp_index=best_grasp_index,
        best_grasp=grasps_base[best_grasp_index].tolist(),
        best_score=float(scores[best_grasp_index]),
        metadata={
            "ply_file_base64": base64.b64encode(ply_content).decode('utf-8')
        }
    )


def transform_grasps_to_base_frame(
    grasp_qt: np.ndarray,
    obj_center_in_camera: np.ndarray,
    scale_factor: float,
    camera_extrinsics: np.ndarray
) -> np.ndarray:
    """
    Transform grasps from normalized object-centered frame to robot base frame.

    Args:
        grasp_qt: (N, 7+16) array of grasps in format [qw, qx, qy, qz, x, y, z, joint_angles...]
                  Poses are in normalized object-centered frame
        obj_center_in_camera: (3,) object center position in camera frame (meters)
        scale_factor: normalization scale factor applied to point cloud
        camera_extrinsics: (4, 4) transformation matrix from camera frame to robot base frame

    Returns:
        grasp_transformed: (N, 7+16) array of grasps in robot base frame
                          Format: [qw, qx, qy, qz, x, y, z, joint_angles...]
    """
    # Extract quaternion and translation
    quat_wxyz = grasp_qt[:, :4]  # (N, 4) - quaternion in wxyz format
    trans_normalized = grasp_qt[:, 4:7]  # (N, 3) - translation in normalized object frame
    joint_angles = grasp_qt[:, 7:]  # (N, 16) - joint angles (unchanged)

    print(f"\n=== Coordinate Transformation Steps (First Grasp) ===")
    print(f"Input transformation parameters:")
    print(f"  - obj_center_in_camera (m): {obj_center_in_camera}")
    print(f"  - scale_factor: {scale_factor}")
    print(f"  - camera_extrinsics:\n{camera_extrinsics}")

    print(f"\nStep 0: Original normalized grasp")
    print(f"  - trans_normalized: {trans_normalized[0]}")
    print(f"  - quat_wxyz: {quat_wxyz[0]}")

    # Step 1: Un-normalize the scale
    trans_in_obj_frame = trans_normalized / scale_factor  # Back to meters
    print(f"\nStep 1: Un-normalize scale (/ {scale_factor})")
    print(f"  - trans_in_obj_frame (m): {trans_in_obj_frame[0]}")

    # Step 2: Transform from object frame to camera frame
    # Rotation stays the same (object frame and camera frame have same orientation)
    trans_in_camera = trans_in_obj_frame + obj_center_in_camera[np.newaxis, :]  # (N, 3)
    print(f"\nStep 2: Transform to camera frame (+ obj_center)")
    print(f"  - trans_in_camera (m): {trans_in_camera[0]}")

    # Step 3: Transform from camera frame to robot base frame
    # camera_extrinsics is T_base_camera (4x4 matrix)
    R_base_camera = camera_extrinsics[:3, :3]  # (3, 3)
    t_base_camera = camera_extrinsics[:3, 3]   # (3,)

    print(f"\nStep 3: Transform to base frame")
    print(f"  - R_base_camera:\n{R_base_camera}")
    print(f"  - t_base_camera: {t_base_camera}")

    # Convert quaternion to rotation matrix
    rot_scipy = R.from_quat(np.roll(quat_wxyz, -1, axis=1))  # Convert wxyz to xyzw for scipy
    R_camera_grasp = rot_scipy.as_matrix()  # (N, 3, 3)

    # Transform rotation: R_base_grasp = R_base_camera @ R_camera_grasp
    R_base_grasp = np.einsum('ij,njk->nik', R_base_camera, R_camera_grasp)  # (N, 3, 3)

    # Transform translation: t_base_grasp = R_base_camera @ t_camera_grasp + t_base_camera
    trans_in_base = np.einsum('ij,nj->ni', R_base_camera, trans_in_camera) + t_base_camera[np.newaxis, :]

    print(f"  - R_base_camera @ trans_in_camera: {np.dot(R_base_camera, trans_in_camera[0])}")
    print(f"  - trans_in_base (m): {trans_in_base[0]}")
    print(f"\nFinal output format: [qw, qx, qy, qz, x, y, z, joint_angles...]")
    print(f"===================================================\n")

    # Convert back to quaternion
    rot_scipy_base = R.from_matrix(R_base_grasp)
    quat_xyzw_base = rot_scipy_base.as_quat()  # (N, 4) in xyzw format
    quat_wxyz_base = np.roll(quat_xyzw_base, 1, axis=1)  # Convert to wxyz

    # Assemble output in format: [qw, qx, qy, qz, x, y, z, joint_angles...]
    N = grasp_qt.shape[0]
    grasp_transformed = np.zeros((N, 23))  # 4 + 3 + 16
    grasp_transformed[:, 0:4] = quat_wxyz_base     # Quaternion (w, x, y, z)
    grasp_transformed[:, 4:7] = trans_in_base      # Position (x, y, z)
    grasp_transformed[:, 7:] = joint_angles        # Joint angles

    return grasp_transformed


# ============================================================================
# Request/Response Models
# ============================================================================

class GraspRequest(BaseModel):
    target_objects: List[str]
    confidence_threshold: float = 0.1
    num_samples: int = 32

class GraspResponse(BaseModel):
    grasp_qt: List[List[float]]
    scores: List[float]
    best_grasp_index: int
    best_grasp: List[float]
    best_score: float
    metadata: Dict[str, Any]


# ============================================================================
# FastAPI Application Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "DexDiffuser Grasp Generation API",
        "endpoints": {
            "/process_grasp": "POST - Main endpoint for grasp generation from RGB-D",
            "/process_pcd": "POST - Generate grasps directly from point cloud (.pt file)",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": dex_diffuser is not None
    }

@app.post("/process_grasp", response_model=GraspResponse)
async def process_grasp(
    rgb_image: UploadFile = File(...),
    depth_data: UploadFile = File(...),
    camera_intrinsics: UploadFile = File(...),
    camera_extrinsics: UploadFile = File(...),
    target_objects: str = Form(...),
    confidence_threshold: float = Form(0.1),
    num_samples: int = Form(32)
):
    """
    Main endpoint to process RGB-D data and generate grasps.

    Args:
        rgb_image: RGB image file (PNG/JPG)
        depth_data: Depth data as numpy array (.npy file)
        camera_intrinsics: 3x3 camera intrinsic matrix (.npy file)
        camera_extrinsics: 4x4 camera extrinsic matrix (.npy file) - transformation from camera to robot base
        target_objects: Comma-separated list of target objects
        confidence_threshold: Detection confidence threshold
        num_samples: Number of grasp samples to generate

    Returns:
        GraspResponse with grasp poses in robot base frame and scores
        Grasp format (23 dims): [qw, qx, qy, qz, x, y, z, joint_angles(16)]
    """
    if dex_diffuser is None:
        raise HTTPException(status_code=503, detail="DexDiffuser model not loaded")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_path = Path(temp_dir)
            target_objects_list = [obj.strip() for obj in target_objects.split(",")]

            # Save uploaded files
            rgb_path = temp_path / "color.png"
            depth_path = temp_path / "depth.npy"
            intrinsics_path = temp_path / "camera_intrinsics.npy"
            extrinsics_path = temp_path / "camera_extrinsics.npy"

            for file, path in [
                (rgb_image, rgb_path),
                (depth_data, depth_path),
                (camera_intrinsics, intrinsics_path),
                (camera_extrinsics, extrinsics_path)
            ]:
                content = await file.read()
                with open(path, "wb") as f:
                    f.write(content)

            # Run detection and segmentation
            print("Running detection and segmentation...")
            results = detect_seg_pipeline(
                image_path=str(rgb_path),
                out_root=str(temp_path),
                target_objects=target_objects_list,
                confidence_threshold=confidence_threshold
            )

            if results is None:
                raise HTTPException(
                    status_code=404,
                    detail="No objects detected. Try lowering confidence_threshold or checking target_objects."
                )

            # Generate point cloud from RGB-D and mask
            print("Generating point cloud...")
            mask = cv2.imread(results["paths"]["mask"], cv2.IMREAD_UNCHANGED).astype(bool)
            color_o3d = o3d.io.read_image(str(rgb_path))
            depth_np = np.load(depth_path)
            K = np.load(intrinsics_path)

            geoms = pcd_from_rgbd_cpu(
                color_img_o3d=color_o3d,
                depth_np=depth_np,
                K=K,
                depth_unit="mm",
                flip_for_view=False,
                add_axis=True,
                axis_size=0.1,
                mask=mask
            )

            pcd, _ = geoms[0].remove_statistical_outlier(
                nb_neighbors=OUTLIER_NB_NEIGHBORS_RGBD,
                std_ratio=OUTLIER_STD_RATIO_RGBD
            )
            obj_pcd = np.asarray(pcd.points)

            # Check if point cloud is empty
            if obj_pcd.shape[0] == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Generated point cloud is empty. This could be due to:\n"
                        "1. Depth values are outside the valid range\n"
                        "2. Segmentation mask doesn't overlap with valid depth pixels\n"
                        "3. Depth data is invalid or all zeros"
                    )
                )

            # Process point cloud (unit conversion, centering, save debug files)
            output_dir = Path(DEBUG_OUTPUT_DIR)
            obj_pcd_centered, obj_pcd_in_camera, obj_center, scale_factor = process_point_cloud(
                obj_pcd, output_dir, pcd
            )

            # Generate and transform grasps
            camera_extrinsics_matrix = np.load(extrinsics_path)
            grasps_base, scores, best_idx = generate_and_transform_grasps(
                obj_pcd_centered,
                obj_center,
                scale_factor,
                camera_extrinsics_matrix,
                num_samples,
                dex_diffuser
            )

            # Save debug files
            overlay_path = results["paths"].get("overlay")
            save_debug_files(
                output_dir,
                camera_extrinsics_matrix,
                grasps_base[best_idx],
                rgb_path=rgb_path,
                depth_np=depth_np,
                mask_path=Path(results["paths"]["mask"]),
                overlay_path=Path(overlay_path) if overlay_path else None
            )

            # Create response
            ply_path = output_dir / "object_colored.ply"
            response = create_grasp_response(grasps_base, scores, best_idx, ply_path)

            # Launch visualization
            print("Grasp generation completed successfully!")
            launch_visualization(obj_pcd_in_camera, grasps_base, scores, camera_extrinsics_matrix)

            return response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/process_pcd", response_model=GraspResponse)
async def process_pcd(
    point_cloud: UploadFile = File(...),
    camera_extrinsics: UploadFile = File(...),
    num_samples: int = Form(32)
):
    """
    Process point cloud directly and generate grasps.

    Args:
        point_cloud: Point cloud file (.pt format) with shape [N, 3+] (x, y, z, ...)
        camera_extrinsics: 4x4 camera extrinsic matrix (.npy file) - transformation from camera to robot base
        num_samples: Number of grasp samples to generate

    Returns:
        GraspResponse with grasp poses in robot base frame and scores
        Grasp format (23 dims): [qw, qx, qy, qz, x, y, z, joint_angles(16)]
    """
    if dex_diffuser is None:
        raise HTTPException(status_code=503, detail="DexDiffuser model not loaded")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_path = Path(temp_dir)

            # Save uploaded files
            pcd_path = temp_path / "point_cloud.pt"
            extrinsics_path = temp_path / "camera_extrinsics.npy"

            for file, path in [(point_cloud, pcd_path), (camera_extrinsics, extrinsics_path)]:
                content = await file.read()
                with open(path, "wb") as f:
                    f.write(content)

            # Load and prepare point cloud
            print("Loading point cloud...")
            pcd_data = torch.load(pcd_path)
            if isinstance(pcd_data, torch.Tensor):
                pcd_data = pcd_data.cpu().numpy()

            obj_pcd = pcd_data[:, :3]  # Extract xyz coordinates
            print(f"Point cloud loaded with shape: {obj_pcd.shape}")

            if obj_pcd.shape[0] == 0:
                raise HTTPException(status_code=400, detail="Point cloud is empty")

            # Preprocess point cloud (downsample, remove noise and plane)
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(obj_pcd)
            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=VOXEL_SIZE)
            pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(
                nb_neighbors=OUTLIER_NB_NEIGHBORS_PCD,
                std_ratio=OUTLIER_STD_RATIO_PCD
            )

            # Remove plane (table surface)
            _, inliers = pcd_o3d.segment_plane(
                distance_threshold=PLANE_DISTANCE_THRESHOLD,
                ransac_n=PLANE_RANSAC_N,
                num_iterations=PLANE_NUM_ITERATIONS
            )
            pcd_o3d = pcd_o3d.select_by_index(inliers, invert=True)
            obj_pcd = np.asarray(pcd_o3d.points)

            # Process point cloud (unit conversion, centering, save debug files)
            output_dir = Path(DEBUG_OUTPUT_DIR)
            obj_pcd_centered, obj_pcd_in_camera, obj_center, scale_factor = process_point_cloud(
                obj_pcd, output_dir, pcd_o3d
            )

            # Generate and transform grasps
            camera_extrinsics_matrix = np.load(extrinsics_path)
            grasps_base, scores, best_idx = generate_and_transform_grasps(
                obj_pcd_centered,
                obj_center,
                scale_factor,
                camera_extrinsics_matrix,
                num_samples,
                dex_diffuser
            )

            # Save debug files
            save_debug_files(output_dir, camera_extrinsics_matrix, grasps_base[best_idx])

            # Create response
            ply_path = output_dir / "object_colored.ply"
            response = create_grasp_response(grasps_base, scores, best_idx, ply_path)

            # Launch visualization
            print("Grasp generation completed successfully!")
            launch_visualization(obj_pcd_in_camera, grasps_base, scores, camera_extrinsics_matrix)

            return response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )
