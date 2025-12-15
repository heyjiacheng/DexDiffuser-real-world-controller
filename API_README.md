# DexDiffuser API Service

FastAPI service for robotic grasp generation from RGB-D images.

## Quick Start

### Server (100.120.117.28)

```bash
pip install -r requirements_api.txt
./start_server.sh
```

Server runs at: `http://100.120.117.28:8000`

### Client (100.85.123.57)

```bash
pip install -r requirements_client.txt
```

## Client Usage

```python
from client_example import DexDiffuserClient

# Connect to server
client = DexDiffuserClient(server_url="http://100.120.117.28:8000")

# Generate grasps
result = client.process_grasp(
    rgb_image_path="color.png",
    depth_path="depth.npy",              # HxW array in meters
    camera_intrinsics_path="K.npy",      # 3x3 intrinsic matrix
    target_objects=["spatula"],
    num_samples=32
)

# Get best grasp
best_grasp = result['grasp_qt'][result['best_grasp_index']]
# Format: [qw, qx, qy, qz, x, y, z, joint_angles...]

# Optional: Download intermediate results
client.download_segmentation_mask("mask.png")
client.download_pointcloud("pcd.npy")
client.download_grasp("grasp.npy")
```

## Data Format

- **RGB**: PNG/JPG image
- **Depth**: NumPy .npy file, shape (H,W), units in meters
- **Intrinsics**: NumPy .npy file, shape (3,3), [[fx,0,cx],[0,fy,cy],[0,0,1]]

## Main Endpoints

- `POST /process_grasp` - Generate grasps (returns session_id, grasp_qt, scores)
- `GET /get_segmentation/{session_id}` - Get segmentation info
- `GET /get_pointcloud/{session_id}` - Get point cloud
- `GET /download_grasp/{session_id}` - Download grasp .npy file

See `http://100.120.117.28:8000/docs` for full API documentation.
