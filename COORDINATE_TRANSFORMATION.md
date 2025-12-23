# Coordinate Transformation Guide

Transforms grasps from the model's normalized object frame to the robot's base frame.

## Transformation Pipeline

```
Normalized Object Frame → Object Frame → Camera Frame → Robot Base Frame
     (un-scale)            (translate)      (extrinsics)
```

**Formula:**
```python
pos_base = R_base_camera @ (pos_normalized / scale_factor + obj_center_in_camera) + t_base_camera
```

## Visual Example

```
                    Camera Frame
                         │
                         │ obj_center_in_camera
                         ↓
                    ┌─────────┐
                    │ Object  │ ← Object Frame (centered at object)
                    └─────────┘
                         │
                         │ R_base_camera, t_base_camera
                         ↓
                    Robot Base Frame
```

## Implementation

See `api_service.py` → `transform_grasps_to_base_frame()` function.
