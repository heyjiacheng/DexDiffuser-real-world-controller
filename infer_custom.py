#!/usr/bin/env python3
"""
Clean inference script for DexDiffuser grasp generation.

Usage:
    # From mesh:
    python infer_custom.py --mesh_path "/path/to/mesh.obj"
    python infer_custom.py --mesh_path "/path/to/mesh.obj" --num_samples 32 --show_viz
    python infer_custom.py --mesh_path "/path/to/mesh.obj" --output_dir ./results

    # From image (RGB-D + language instruction):
    python infer_custom.py --image_path "/path/to/rgb.png" --depth_path "/path/to/depth.npy" \
                           --camera_intrinsics "/path/to/K.npy" --target_object "bear"
    python infer_custom.py --image_path "/path/to/rgb.png" --depth_path "/path/to/depth.npy" \
                           --camera_intrinsics "/path/to/K.npy" --target_object "cup" --num_samples 32
"""

import os
import argparse
import numpy as np
import torch
import cv2
import open3d as o3d
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from plotly import graph_objects as go

from models import create_ddpm, create_evaluator
from sample import load_ckpt
from bps_torch.bps import bps_torch
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.handmodel import get_handmodel, angle_denormalize
from scipy.spatial.transform import Rotation as R
from gen_pcds.main import detect_seg_pipeline, pcd_from_rgbd_cpu


def load_config(config_path: str):
    """Load YAML configuration using OmegaConf."""
    return OmegaConf.load(config_path)


def mesh_to_pointcloud(mesh_path: str, num_points: int = 2048, scale: float = 1.0) -> np.ndarray:
    """
    Convert a mesh file to a point cloud.

    Args:
        mesh_path: Path to the mesh file (.obj, .ply, .stl, etc.)
        num_points: Number of points to sample from the mesh
        scale: Scale factor to apply to the mesh

    Returns:
        Point cloud as numpy array of shape (num_points, 3)
    """
    logger.info(f"Loading mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")

    # Apply scale if specified
    if scale != 1.0:
        logger.info(f"Scaling mesh by factor: {scale}")
        mesh.scale(scale, center=mesh.get_center())

    # Sample points uniformly from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points).astype(np.float32)

    # Center the point cloud (DexDiffuser expects centered point clouds)
    pcd_center = points.mean(axis=0)
    points_centered = points - pcd_center
    logger.info(f"Centered point cloud. Original center: {pcd_center}")

    logger.info(f"Sampled {len(points_centered)} points from mesh")
    return points_centered


# Constants for image-based point cloud processing
OUTLIER_NB_NEIGHBORS = 100
OUTLIER_STD_RATIO = 2.0


def image_to_pointcloud(
    image_path: str,
    depth_path: str,
    camera_intrinsics_path: str,
    target_object: str,
    confidence_threshold: float = 0.1,
    output_dir: str = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    Convert RGB-D image with semantic segmentation to a point cloud.

    Args:
        image_path: Path to the RGB image file (.png, .jpg)
        depth_path: Path to the depth data file (.npy)
        camera_intrinsics_path: Path to the camera intrinsics matrix (.npy, 3x3)
        target_object: Language instruction for semantic segmentation (e.g., "bear", "cup")
        confidence_threshold: Detection confidence threshold
        output_dir: Directory to save intermediate results (mask, overlay, etc.)
        scale: Scale factor to apply to the point cloud (e.g., 0.001 to convert mm to meters)

    Returns:
        Point cloud as numpy array of shape (N, 3)
    """
    logger.info(f"Loading image from: {image_path}")
    logger.info(f"Loading depth from: {depth_path}")
    logger.info(f"Target object: {target_object}")

    # Determine output directory for segmentation results
    if output_dir is None:
        output_dir = str(Path(image_path).parent)

    # Run detection and segmentation
    logger.info("Running detection and segmentation...")
    target_objects_list = [target_object.strip()]
    results = detect_seg_pipeline(
        image_path=str(image_path),
        out_root=output_dir,
        target_objects=target_objects_list,
        confidence_threshold=confidence_threshold
    )

    if results is None:
        raise ValueError(
            f"No objects detected for target '{target_object}'. "
            "Try lowering confidence_threshold or checking the target_object name."
        )

    logger.info(f"Detection/segmentation completed. Result keys: {list(results.keys())}")

    # Load mask, color image, depth, and camera intrinsics
    mask_path = results["paths"]["mask"]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(bool)
    logger.info(f"Loaded mask from: {mask_path}, shape: {mask.shape}")

    color_o3d = o3d.io.read_image(str(image_path))
    depth_np = np.load(depth_path)
    K = np.load(camera_intrinsics_path)

    logger.info(f"Depth shape: {depth_np.shape}, dtype: {depth_np.dtype}")
    logger.info(f"Camera intrinsics:\n{K}")

    # Generate point cloud from RGB-D and mask
    logger.info("Generating point cloud from RGB-D...")
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

    # Apply statistical outlier removal
    pcd, _ = geoms[0].remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO
    )

    points = np.asarray(pcd.points).astype(np.float32)

    if points.shape[0] == 0:
        raise ValueError(
            "Generated point cloud is empty. This could be due to:\n"
            "1. Depth values are outside the valid range\n"
            "2. Segmentation mask doesn't overlap with valid depth pixels\n"
            "3. Depth data is invalid or all zeros"
        )

    points = points * 1000.0 * scale

    # Center the point cloud (DexDiffuser expects centered point clouds)
    pcd_center = points.mean(axis=0)
    points_centered = points - pcd_center
    logger.info(f"Centered point cloud. Original center: {pcd_center}")

    logger.info(f"Generated {len(points_centered)} points from image")
    logger.info(f"Point cloud range: min={points_centered.min(axis=0)}, max={points_centered.max(axis=0)}")

    # Save intermediate results if overlay exists
    if "overlay" in results["paths"] and results["paths"]["overlay"]:
        logger.info(f"Segmentation overlay saved to: {results['paths']['overlay']}")

    return points_centered


def convert_transformation_matrix_to_qt(transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert transformation matrices to quaternion + translation format.

    Args:
        transformation_matrix: Transformation matrices of shape (B, 4, 4)

    Returns:
        qt: Array of shape (B, 7) with format [qw, qx, qy, qz, x, y, z]
    """
    if len(transformation_matrix.shape) == 2:
        transformation_matrix = np.expand_dims(transformation_matrix, 0)

    t = transformation_matrix[:, :3, 3]
    Rs = transformation_matrix[:, :3, :3]

    R_rotmat = R.from_matrix(Rs)
    q = R_rotmat.as_quat()  # xyzw format
    q = np.roll(q, 1, axis=1)  # convert to wxyz format

    qt = np.zeros((transformation_matrix.shape[0], 7))
    qt[:, :4] = q
    qt[:, 4:] = t

    return qt


def plot_point_cloud(pts: np.ndarray, color: str = 'lightblue') -> go.Scatter3d:
    """Create a plotly scatter3d trace for point cloud visualization."""
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={'color': color, 'size': 3, 'opacity': 1}
    )


class DexDiffuserInference:
    """DexDiffuser model for grasp generation inference."""

    def __init__(self, config_dir: str = "configs", device: str = "cuda"):
        self.device = device

        # Load configurations
        cfg = load_config(os.path.join(config_dir, "sample.yaml"))
        cfg.model = load_config(os.path.join(config_dir, "model/unet_grasp_bps.yaml"))
        cfg.diffuser = load_config(os.path.join(config_dir, "diffuser/ddpm.yaml"))
        cfg.task = load_config(os.path.join(config_dir, "task/grasp_gen_ur_dexgn_slurm.yaml"))
        cfg.refinement = True

        self.cfg = cfg

        # Initialize BPS encoder
        basis_bps_set = np.load('models/basis_point_set.npy')
        self.bps = bps_torch(n_bps_points=4096, n_dims=3, custom_basis=basis_bps_set)

        # Create and load model
        logger.info("Loading DexDiffuser model...")
        self.model = create_ddpm(cfg)
        self.model.to(device=self.device)
        self.model.eval()

        # Load model checkpoint
        sampler_pth = cfg.sampler_bps_ckpt_pth
        load_ckpt(self.model, path=sampler_pth)

        # Create and load evaluator
        logger.info("Loading evaluator...")
        self.evaluator = create_evaluator(cfg, pos_enc_multires=[10, 4, -1])
        load_ckpt(self.evaluator, path=cfg.evaluator_ckpt_pth)
        self.evaluator.to(device=self.device)
        self.evaluator.eval()

        # Optional refinement
        self.refinement = cfg.refinement
        if self.refinement:
            from refine import RefineNN
            self.refineNN = RefineNN(self.evaluator)

        # Guidance parameters
        if cfg.guid_scale is not None:
            self.guid_param = {
                'evaluator': self.evaluator,
                'guid_scale': cfg.guid_scale
            }
        else:
            self.guid_param = None

        logger.info("Model initialized successfully!")

    def sample_grasps(self, obj_pcd: np.ndarray, num_samples: int = 32):
        """
        Generate grasp poses for the given point cloud.

        Args:
            obj_pcd: Object point cloud of shape (N, 3)
            num_samples: Number of grasp samples to generate

        Returns:
            grasp_qt: Grasp poses in [qw, qx, qy, qz, x, y, z, joint_angles] format
            scores: Confidence scores for each grasp
        """
        obj_pcd_torch = torch.from_numpy(obj_pcd).unsqueeze(0).repeat(num_samples, 1, 1).to(self.device)

        data = {
            'x': torch.randn(num_samples, 25, device=self.device),
            'pos': obj_pcd_torch,
        }
        data['obj_bps'] = self.bps.encode(obj_pcd_torch, feature_type=['dists'])['dists']

        logger.info("Sampling grasps with DexDiffuser...")
        outputs = self.model.sample(data, k=1, guid_param=self.guid_param).squeeze(1)[:, -1, :]

        # Optional refinement
        if self.refinement:
            logger.info("Refining grasps...")
            data['x_t'] = outputs.detach()
            outputs, _ = self.refineNN.improve_grasps_sampling_based(
                data, num_refine_steps=100, delta_translation=0.001
            )
            outputs = torch.from_numpy(outputs).to(self.device).to(torch.float32)

        # Evaluate grasps
        data['x_t'] = outputs
        scores = self.evaluator(data)['p_success'].detach().cpu().numpy().squeeze()

        # Denormalize joint angles
        outputs[:, 9:] = angle_denormalize(joint_angle=outputs[:, 9:])
        outputs = outputs.float().detach().cpu()

        # Convert to quaternion + translation format
        rot_ = robust_compute_rotation_matrix_from_ortho6d(outputs[:, 3:9])
        rot_mat = np.tile(np.eye(4), (num_samples, 1, 1))
        rot_mat[:, :3, :3] = rot_.numpy()
        rot_mat[:, :3, 3] = outputs[:, :3].numpy()

        qt = convert_transformation_matrix_to_qt(rot_mat)
        grasp_qt = np.concatenate([qt, outputs[:, 9:].numpy()], axis=1)

        return grasp_qt, scores

    def visualize(self, obj_pcd: np.ndarray, grasp_output: torch.Tensor,
                  robot: str = 'allegro_right', save_path: str = None, show: bool = True):
        """
        Visualize the grasp pose with the hand model.

        Args:
            obj_pcd: Object point cloud
            grasp_output: Grasp output tensor of shape (1, 25) or (25,)
            robot: Robot hand type
            save_path: Path to save the visualization HTML
            show: Whether to show the interactive visualization
        """
        if grasp_output.dim() == 1:
            grasp_output = grasp_output.unsqueeze(0)

        hand_model = get_handmodel(
            batch_size=1, device='cuda',
            urdf_path='data/urdf', robot=robot
        )

        hand_model.update_kinematics(q=grasp_output.to('cuda'))

        vis_data = [plot_point_cloud(obj_pcd, color='pink')]
        vis_data += hand_model.get_plotly_data(opacity=0.5, color='lightblue')

        fig = go.Figure(data=vis_data)
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='DexDiffuser Grasp Visualization'
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to: {save_path}")

        if show:
            fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="DexDiffuser Grasp Generation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From mesh:
  python infer_custom.py --mesh_path "/path/to/mesh.obj"
  python infer_custom.py --mesh_path "/path/to/mesh.obj" --num_samples 32 --show_viz

  # From image (RGB-D + language instruction):
  python infer_custom.py --image_path "/path/to/rgb.png" --depth_path "/path/to/depth.npy" \\
                         --camera_intrinsics "/path/to/K.npy" --target_object "bear"
        """
    )

    # Input source group (mutually exclusive: mesh OR image)
    input_group = parser.add_argument_group("Input source (choose one)")
    input_group.add_argument("--mesh_path", type=str, default=None,
                             help="Path to the input mesh file (.obj, .ply, .stl)")

    # Image-based input arguments
    input_group.add_argument("--image_path", type=str, default=None,
                             help="Path to the RGB image file (.png, .jpg)")
    input_group.add_argument("--depth_path", type=str, default=None,
                             help="Path to the depth data file (.npy)")
    input_group.add_argument("--camera_intrinsics", type=str, default=None,
                             help="Path to the camera intrinsics matrix (.npy, 3x3)")
    input_group.add_argument("--target_object", type=str, default=None,
                             help="Language instruction for semantic segmentation (e.g., 'bear', 'cup')")
    input_group.add_argument("--confidence_threshold", type=float, default=0.1,
                             help="Detection confidence threshold for segmentation")

    # Common arguments
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Number of grasp samples to generate")
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points to sample from the mesh (mesh input only)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save results")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory containing config files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--show_viz", action="store_true",
                        help="Show interactive visualization window")
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="Save visualization as HTML file")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor to apply to the point cloud (e.g., 0.001 to convert mm to meters)")

    args = parser.parse_args()

    # Validate input arguments
    use_mesh = args.mesh_path is not None
    use_image = args.image_path is not None

    if not use_mesh and not use_image:
        parser.error("Must provide either --mesh_path OR (--image_path, --depth_path, --camera_intrinsics, --target_object)")

    if use_mesh and use_image:
        parser.error("Cannot use both --mesh_path and --image_path. Choose one input source.")

    if use_image:
        missing = []
        if args.depth_path is None:
            missing.append("--depth_path")
        if args.camera_intrinsics is None:
            missing.append("--camera_intrinsics")
        if args.target_object is None:
            missing.append("--target_object")
        if missing:
            parser.error(f"When using --image_path, the following arguments are also required: {', '.join(missing)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate point cloud based on input type
    if args.mesh_path is not None:
        # Mesh-based input
        input_name = Path(args.mesh_path).stem
        logger.info(f"Using mesh input: {args.mesh_path}")
        obj_pcd = mesh_to_pointcloud(args.mesh_path, num_points=args.num_points, scale=args.scale)
    else:
        # Image-based input (RGB-D + semantic segmentation)
        input_name = Path(args.image_path).stem
        logger.info(f"Using image input: {args.image_path}")
        obj_pcd = image_to_pointcloud(
            image_path=args.image_path,
            depth_path=args.depth_path,
            camera_intrinsics_path=args.camera_intrinsics,
            target_object=args.target_object,
            confidence_threshold=args.confidence_threshold,
            output_dir=str(output_dir),
            scale=args.scale
        )

    # Initialize model and run inference
    model = DexDiffuserInference(config_dir=args.config_dir, device=args.device)
    grasp_qt, scores = model.sample_grasps(obj_pcd, num_samples=args.num_samples)

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    grasp_qt_sorted = grasp_qt[sorted_indices]
    scores_sorted = scores[sorted_indices]

    # Print all grasp scores
    logger.info(f"\nAll {len(scores_sorted)} grasps sorted by score:")
    for i in range(len(scores_sorted)):
        logger.info(f"  Grasp {i+1}: score = {scores_sorted[i]:.4f}")

    # Reconstruct grasp output for visualization (need the original 25-dim format)
    # Re-run to get the raw output for visualization
    obj_pcd_torch = torch.from_numpy(obj_pcd).unsqueeze(0).to(args.device)
    data = {
        'x': torch.randn(args.num_samples, 25, device=args.device),
        'pos': obj_pcd_torch.repeat(args.num_samples, 1, 1),
    }
    data['obj_bps'] = model.bps.encode(data['pos'], feature_type=['dists'])['dists']

    outputs = model.model.sample(data, k=1, guid_param=model.guid_param).squeeze(1)[:, -1, :]

    if model.refinement:
        data['x_t'] = outputs.detach()
        outputs_np, _ = model.refineNN.improve_grasps_sampling_based(
            data, num_refine_steps=100, delta_translation=0.001
        )
        outputs = torch.from_numpy(outputs_np).to(args.device).to(torch.float32)

    # Evaluate and sort
    data['x_t'] = outputs
    eval_scores = model.evaluator(data)['p_success'].detach().cpu().numpy().squeeze()
    sorted_viz_indices = np.argsort(eval_scores)[::-1]

    # Denormalize for visualization
    outputs[:, 9:] = angle_denormalize(joint_angle=outputs[:, 9:])
    outputs = outputs.float().detach().cpu()

    # Save visualization for all samples
    if args.save_viz or args.show_viz:
        viz_dir = output_dir / f"{input_name}_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        for rank, idx in enumerate(sorted_viz_indices):
            score = eval_scores[idx]
            viz_path = viz_dir / f"grasp_{rank+1:03d}_score_{score:.4f}.html" if args.save_viz else None
            # Only show interactive window for the best grasp
            show_this = args.show_viz and (rank == 0)
            model.visualize(
                obj_pcd,
                outputs[idx:idx+1],
                save_path=str(viz_path) if viz_path else None,
                show=show_this
            )
        logger.info(f"Saved {len(sorted_viz_indices)} visualizations to: {viz_dir}")

    logger.info("Inference complete!")
    return grasp_qt_sorted, scores_sorted


if __name__ == "__main__":
    main()
