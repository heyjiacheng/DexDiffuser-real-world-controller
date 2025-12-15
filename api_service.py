import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
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

app = FastAPI(title="DexDiffuser Grasp Generation API")

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
    camera_extrinsics: Optional[UploadFile] = File(None),
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
        camera_extrinsics: Optional 4x4 camera extrinsic matrix (.npy file)
        target_objects: Comma-separated list of target objects
        confidence_threshold: Detection confidence threshold
        num_samples: Number of grasp samples to generate

    Returns:
        GraspResponse with grasp poses and scores
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

            # Save camera extrinsics if provided
            if camera_extrinsics:
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

            # Extract point cloud as numpy array
            obj_pcd = np.asarray(pcd.points)
            print(f"Object point cloud shape: {obj_pcd.shape}")

            # Check if point cloud is empty
            if obj_pcd.shape[0] == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Generated point cloud is empty. This could be due to:\n"
                        "1. Depth values are outside the valid range (0 < depth < 1 meter)\n"
                        "2. Segmentation mask doesn't overlap with valid depth pixels\n"
                        "3. Depth data is invalid or all zeros\n"
                        "Check the debug logs above for more details."
                    )
                )

            # Step 2: Generate grasps
            print("Generating grasps...")
            grasp_qt, scores = dex_diffuser.sample_grasps(obj_pcd, num_samples=num_samples)

            # Find best grasp
            best_grasp_index = int(np.argmax(scores))

            # Prepare response
            response = GraspResponse(
                grasp_qt=grasp_qt.tolist(),
                scores=scores.tolist(),
                best_grasp_index=best_grasp_index,
                best_grasp=grasp_qt[best_grasp_index].tolist(),
                best_score=float(scores[best_grasp_index]),
                metadata={
                    "label": results["label"],
                    "detection_score": results["score"],
                    "segmentation_iou": results["iou"],
                    "mask_area": results["mask_area"],
                    "point_cloud_size": int(obj_pcd.shape[0]),
                    "target_objects": target_objects_list
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
