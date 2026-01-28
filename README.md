# DexDiffuser API Service

FastAPI service for real-time dexterous grasp generation.

## Quick Start

load env:
```bash
conda activate dexdiff
```

Direct visualize dexdiffuser on own dataset (mesh & point cloud input):

```bash
python infer_custom.py --mesh_path "/home/supertc/repo/hamer/dexgraspnet_viz/hammer_1_pound/mesh/coacd/decomposed.obj" --scale 0.2
```

(optional) add --show_viz will pop up a window to viz

Direct visualize dexdiffuser on own dataset (image input):

```bash
python infer_custom.py --image_path "sim_test/0_color.png" --depth_path "sim_test/0_depth_aligned_rgb.npy" --camera_intrinsics "sim_test/0_camerainfo.npy" --target_object "bottle"
```

image input + grasp point close to part-level object
(grasp point = palm center + palm direction 7cm)

```bash
python infer_custom.py --image_path "sim_test/0_color.png" --depth_path "sim_test/0_depth_aligned_rgb.npy" --camera_intrinsics "sim_test/0_camerainfo.npy" --target_object "bottle" --part "cap" --num_samples 1000 --show_viz
```

Dexdiffuser as server:

```bash
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