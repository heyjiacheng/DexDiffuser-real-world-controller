# DexDiffuser API Service

FastAPI service for real-time dexterous grasp generation.

## Quick Start

Direct visualize dexdiffuser on own dataset:

```bash
python infer_custom.py --mesh_path "/home/supertc/repo/hamer/dexgraspnet_viz/hammer_1_pound/mesh/coacd/decomposed.obj" --scale 0.2
```

(optional) add --show_viz will pop up a window to viz

Dexdiffuser as server:

```bash
conda activate dexdiff
python api_service.py
```
Server runs on `http://0.0.0.0:8000`

## API Endpoints

**POST /process_grasp** - Generate grasps from RGB-D images
- Input: RGB image, depth (.npy), camera intrinsics (3x3), camera extrinsics (4x4), target objects
- Output: Grasp poses in robot base frame `[qw, qx, qy, qz, x, y, z, joint_angles(16)]`

**POST /process_pcd** - Generate grasps from point cloud
- Input: Point cloud (.pt), camera extrinsics (4x4)
- Output: Grasp poses in robot base frame


Use pytorch3d on RTX5080 (first comment)
```bash
https://github.com/facebookresearch/pytorch3d/issues/1962
```