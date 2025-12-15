import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import functions from existing modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_pcds.main import detect_seg_pipeline, pcd_from_rgbd_cpu
import open3d as o3d
from run_real import DexDiffuser, load_config

app = FastAPI(title="DexDiffuser Grasp Generation API")

# Store session data in memory
sessions: Dict[str, Dict[str, Any]] = {}

# Global DexDiffuser instance (loaded once at startup)
dex_diffuser = None

class GraspRequest(BaseModel):
    target_objects: List[str]
    confidence_threshold: float = 0.1
    num_samples: int = 32

class GraspResponse(BaseModel):
    session_id: str
    grasp_qt: List[List[float]]
    scores: List[float]
    best_grasp_index: int
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
            "/get_segmentation/{session_id}": "GET - Retrieve segmentation results",
            "/get_pointcloud/{session_id}": "GET - Retrieve point cloud",
            "/get_grasp/{session_id}": "GET - Retrieve grasp results",
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

    # Create unique session ID
    session_id = str(uuid.uuid4())
    session_dir = Path("sessions") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Parse target objects
        target_objects_list = [obj.strip() for obj in target_objects.split(",")]

        # Save uploaded files
        rgb_path = session_dir / "color.png"
        depth_path = session_dir / "depth.npy"
        intrinsics_path = session_dir / "camera_intrinsics.npy"

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
        extrinsics_path = None
        if camera_extrinsics:
            extrinsics_path = session_dir / "camera_extrinsics.npy"
            extrinsics_content = await camera_extrinsics.read()
            with open(extrinsics_path, "wb") as f:
                f.write(extrinsics_content)

        # Step 1: Detection and Segmentation
        print(f"Session {session_id}: Running detection and segmentation...")
        results = detect_seg_pipeline(
            image_path=str(rgb_path),
            out_root=str(session_dir),
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
        color_o3d = o3d.io.read_image(str(rgb_path))
        depth_np = np.load(depth_path)
        K = np.load(intrinsics_path)

        # Generate point cloud with mask
        print(f"Session {session_id}: Generating point cloud...")
        geoms = pcd_from_rgbd_cpu(
            color_img_o3d=color_o3d,
            depth_np=depth_np,
            K=K,
            depth_unit="m",
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
        print(f"Session {session_id}: Object point cloud shape: {obj_pcd.shape}")

        # Save point cloud
        pcd_path = session_dir / "obj_pcd.npy"
        np.save(pcd_path, obj_pcd)

        # Step 2: Generate grasps
        print(f"Session {session_id}: Generating grasps...")
        grasp_qt, scores = dex_diffuser.sample_grasps(obj_pcd, num_samples=num_samples)

        # Save grasp results
        grasp_path = session_dir / "grasp.npy"
        scores_path = session_dir / "scores.npy"
        np.save(grasp_path, grasp_qt)
        np.save(scores_path, scores)

        # Find best grasp
        best_grasp_index = int(np.argmax(scores))

        # Store session data
        sessions[session_id] = {
            "session_dir": str(session_dir),
            "results": results,
            "pcd_path": str(pcd_path),
            "grasp_path": str(grasp_path),
            "scores_path": str(scores_path),
            "target_objects": target_objects_list,
            "num_samples": num_samples
        }

        # Prepare response
        response = GraspResponse(
            session_id=session_id,
            grasp_qt=grasp_qt.tolist(),
            scores=scores.tolist(),
            best_grasp_index=best_grasp_index,
            metadata={
                "label": results["label"],
                "detection_score": results["score"],
                "segmentation_iou": results["iou"],
                "mask_area": results["mask_area"],
                "point_cloud_size": int(obj_pcd.shape[0]),
                "target_objects": target_objects_list
            }
        )

        print(f"Session {session_id}: Grasp generation completed successfully!")
        return response

    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        if session_id in sessions:
            del sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/get_segmentation/{session_id}")
async def get_segmentation(session_id: str):
    """Get segmentation results for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    results = session_data["results"]

    return {
        "session_id": session_id,
        "label": results["label"],
        "detection_score": results["score"],
        "segmentation_iou": results["iou"],
        "mask_area": results["mask_area"],
        "bounding_box": results["box"],
        "paths": results["paths"]
    }

@app.get("/get_segmentation_mask/{session_id}")
async def get_segmentation_mask(session_id: str):
    """Download segmentation mask image"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    mask_path = session_data["results"]["paths"]["mask"]

    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask file not found")

    return FileResponse(mask_path, media_type="image/png", filename="mask.png")

@app.get("/get_segmentation_overlay/{session_id}")
async def get_segmentation_overlay(session_id: str):
    """Download segmentation overlay image"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    overlay_path = session_data["results"]["paths"]["overlay"]

    if not os.path.exists(overlay_path):
        raise HTTPException(status_code=404, detail="Overlay file not found")

    return FileResponse(overlay_path, media_type="image/png", filename="overlay.png")

@app.get("/get_pointcloud/{session_id}")
async def get_pointcloud(session_id: str):
    """Get point cloud data for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    pcd_path = session_data["pcd_path"]

    if not os.path.exists(pcd_path):
        raise HTTPException(status_code=404, detail="Point cloud file not found")

    # Load and return point cloud
    obj_pcd = np.load(pcd_path)

    return {
        "session_id": session_id,
        "point_cloud_shape": obj_pcd.shape,
        "num_points": int(obj_pcd.shape[0]),
        "point_cloud": obj_pcd.tolist()
    }

@app.get("/download_pointcloud/{session_id}")
async def download_pointcloud(session_id: str):
    """Download point cloud as .npy file"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    pcd_path = session_data["pcd_path"]

    if not os.path.exists(pcd_path):
        raise HTTPException(status_code=404, detail="Point cloud file not found")

    return FileResponse(pcd_path, media_type="application/octet-stream", filename="obj_pcd.npy")

@app.get("/get_grasp/{session_id}")
async def get_grasp(session_id: str):
    """Get grasp results for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    grasp_path = session_data["grasp_path"]
    scores_path = session_data["scores_path"]

    if not os.path.exists(grasp_path) or not os.path.exists(scores_path):
        raise HTTPException(status_code=404, detail="Grasp files not found")

    # Load grasp and scores
    grasp_qt = np.load(grasp_path)
    scores = np.load(scores_path)
    best_grasp_index = int(np.argmax(scores))

    return {
        "session_id": session_id,
        "grasp_qt": grasp_qt.tolist(),
        "scores": scores.tolist(),
        "best_grasp_index": best_grasp_index,
        "best_grasp": grasp_qt[best_grasp_index].tolist(),
        "best_score": float(scores[best_grasp_index])
    }

@app.get("/download_grasp/{session_id}")
async def download_grasp(session_id: str):
    """Download grasp results as .npy file"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    grasp_path = session_data["grasp_path"]

    if not os.path.exists(grasp_path):
        raise HTTPException(status_code=404, detail="Grasp file not found")

    return FileResponse(grasp_path, media_type="application/octet-stream", filename="grasp.npy")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup files"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    session_dir = Path(session_data["session_dir"])

    # Remove directory
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # Remove from sessions dict
    del sessions[session_id]

    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(sessions),
        "sessions": list(sessions.keys())
    }

if __name__ == "__main__":
    # Create sessions directory if it doesn't exist
    Path("sessions").mkdir(exist_ok=True)

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )
