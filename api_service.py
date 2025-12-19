import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn

# Import functions from existing modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_pcds.main import detect_seg_pipeline, pcd_from_rgbd_cpu
import open3d as o3d
from run_real import DexDiffuser, load_config
from scipy.spatial.transform import Rotation as R

app = FastAPI(title="DexDiffuser Grasp Generation API")

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
                          Format: [x, y, z, qw, qx, qy, qz, joint_angles...]
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
    print(f"\nFinal output format: [x, y, z, qw, qx, qy, qz, joint_angles...]")
    print(f"===================================================\n")

    # Convert back to quaternion
    rot_scipy_base = R.from_matrix(R_base_grasp)
    quat_xyzw_base = rot_scipy_base.as_quat()  # (N, 4) in xyzw format
    quat_wxyz_base = np.roll(quat_xyzw_base, 1, axis=1)  # Convert to wxyz

    # Assemble output in format: [x, y, z, qw, qx, qy, qz, joint_angles...]
    N = grasp_qt.shape[0]
    grasp_transformed = np.zeros((N, 23))  # 3 + 4 + 16
    grasp_transformed[:, 0:3] = trans_in_base      # Position (x, y, z)
    grasp_transformed[:, 3:7] = quat_wxyz_base     # Quaternion (w, x, y, z)
    grasp_transformed[:, 7:] = joint_angles        # Joint angles

    return grasp_transformed

# Global DexDiffuser instance (loaded once at startup)
dex_diffuser = None

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

@app.on_event("startup")
async def startup_event():
    """Initialize DexDiffuser model on startup"""
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

@app.get("/")
async def root():
    return {
        "message": "DexDiffuser Grasp Generation API",
        "endpoints": {
            "/process_grasp": "POST - Main endpoint for grasp generation",
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
        Grasp format (23 dims): [x, y, z, qw, qx, qy, qz, joint_angles(16)]
    """
    if dex_diffuser is None:
        raise HTTPException(status_code=503, detail="DexDiffuser model not loaded")

    # Use temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_path = Path(temp_dir)

            # Parse target objects
            target_objects_list = [obj.strip() for obj in target_objects.split(",")]

            # Save uploaded files
            rgb_path = temp_path / "color.png"
            depth_path = temp_path / "depth.npy"
            intrinsics_path = temp_path / "camera_intrinsics.npy"

            # Save RGB image
            rgb_content = await rgb_image.read()
            with open(rgb_path, "wb") as f:
                f.write(rgb_content)

            # Save depth data
            depth_content = await depth_data.read()
            with open(depth_path, "wb") as f:
                f.write(depth_content)

            # Save camera intrinsics
            intrinsics_content = await camera_intrinsics.read()
            with open(intrinsics_path, "wb") as f:
                f.write(intrinsics_content)

            # Save camera extrinsics
            extrinsics_path = temp_path / "camera_extrinsics.npy"
            extrinsics_content = await camera_extrinsics.read()
            with open(extrinsics_path, "wb") as f:
                f.write(extrinsics_content)

            # Step 1: Detection and Segmentation
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

            # Load mask and generate point cloud
            mask_path = results["paths"]["mask"]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(bool)

            # Load RGB, depth, and intrinsics
            # 注意：Open3D 读取进来的是 RGB 格式。
            # 如果后续要用 cv2 处理或保存，必须先转换成 BGR：
            # cv2_img = cv2.cvtColor(np.asarray(color_o3d), cv2.COLOR_RGB2BGR)

            color_o3d = o3d.io.read_image(str(rgb_path))
            depth_np = np.load(depth_path)
            K = np.load(intrinsics_path)

            # Generate point cloud with mask
            print("Generating point cloud...")
            geoms = pcd_from_rgbd_cpu(
                color_img_o3d=color_o3d,
                depth_np=depth_np,
                K=K,
                depth_unit="mm",  # Depth data is in millimeters
                flip_for_view=False,
                add_axis=True,
                axis_size=0.1,
                mask=mask
            )

            # Remove outliers from point cloud
            pcd = geoms[0]
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

            # Save debug files to persistent local directory
            output_dir = Path("debug_output")
            output_dir.mkdir(exist_ok=True)

            # Save colored point cloud
            ply_output_path = output_dir / "object_colored.ply"
            o3d.io.write_point_cloud(str(ply_output_path), pcd)
            print(f"Saved colored point cloud to: {ply_output_path}")

            # Save original RGB image
            import shutil
            shutil.copy(rgb_path, output_dir / "original_rgb.png")

            # Save depth as visualized image
            depth_vis = (depth_np / depth_np.max() * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "original_depth.png"), depth_vis)

            # Save segmentation mask and overlay
            shutil.copy(results["paths"]["mask"], output_dir / "segmentation_mask.png")
            if "overlay" in results["paths"]:
                shutil.copy(results["paths"]["overlay"], output_dir / "segmentation_overlay.png")

            # Read .ply file content for sending to client
            with open(ply_output_path, "rb") as f:
                ply_content = f.read()

            # Extract point cloud as numpy array
            obj_pcd = np.asarray(pcd.points)
            print(f"Object point cloud shape: {obj_pcd.shape}")

            # Debug: Print point cloud statistics
            print(f"Point cloud statistics BEFORE normalization:")
            print(f"  - Mean: {obj_pcd.mean(axis=0)}")

            # NOTE: pcd_from_rgbd_cpu already returns point cloud in meters
            # No need to convert from mm to meters

            # Center the point cloud (move centroid to origin)
            # IMPORTANT: Save this for later coordinate transformation
            obj_pcd_center_in_camera = obj_pcd.mean(axis=0)  # Object center in camera frame (meters)
            obj_pcd = obj_pcd - obj_pcd_center_in_camera

            # Normalize the scale to a reasonable range
            max_dist = np.sqrt((obj_pcd ** 2).sum(axis=1)).max()
            scale_factor = 1.0
            if max_dist > 0:
                # Scale to approximately 0.1m radius (10cm), which is typical for tabletop objects
                scale_factor = 0.1 / max_dist
                obj_pcd = obj_pcd * scale_factor

            print(f"Point cloud statistics AFTER normalization:")
            print(f"  - Mean: {obj_pcd.mean(axis=0)}")
            print(f"  - Scale factor: {scale_factor}")
            print(f"  - Object center in camera frame: {obj_pcd_center_in_camera}")

            # Check if point cloud is empty
            if obj_pcd.shape[0] == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Generated point cloud is empty. This could be due to:\n"
                        "1. Depth values are outside the valid range (0 < depth < 1 meter)\n"
                        "2. Segmentation mask doesn't overlap with valid depth pixels\n"
                        "3. Depth data is invalid or all zeros"
                    )
                )

            # Step 2: Generate grasps
            print("Generating grasps...")
            grasp_qt, scores = dex_diffuser.sample_grasps(obj_pcd, num_samples=num_samples)

            # Step 3: Transform grasps to robot base frame
            print("Transforming grasps to robot base frame...")
            camera_extrinsics_matrix = np.load(extrinsics_path)
            grasp_qt_base = transform_grasps_to_base_frame(
                grasp_qt=grasp_qt,
                obj_center_in_camera=obj_pcd_center_in_camera,
                scale_factor=scale_factor,
                camera_extrinsics=camera_extrinsics_matrix
            )

            # Find best grasp
            best_grasp_index = int(np.argmax(scores))

            print(f"\n=== Best Grasp Selected ===")
            print(f"Best grasp index: {best_grasp_index}")
            print(f"Best score: {scores[best_grasp_index]:.4f}")
            print(f"Best grasp (23 dims): [x, y, z, qw, qx, qy, qz, joints...]")
            print(f"  Position (x,y,z): {grasp_qt_base[best_grasp_index, 0:3]}")
            print(f"  Quaternion (w,x,y,z): {grasp_qt_base[best_grasp_index, 3:7]}")
            print(f"===========================\n")

            # Prepare response (using transformed grasps in robot base frame)
            response = GraspResponse(
                grasp_qt=grasp_qt_base.tolist(),
                scores=scores.tolist(),
                best_grasp_index=best_grasp_index,
                best_grasp=grasp_qt_base[best_grasp_index].tolist(),
                best_score=float(scores[best_grasp_index]),
                metadata={
                    "ply_file_base64": base64.b64encode(ply_content).decode('utf-8')
                }
            )

            print("Grasp generation completed successfully!")
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
