# 相机视角可视化工具使用指南

## 目的

验证 `transform_grasps_to_base_frame` 函数的坐标转换是否正确,通过从相机视角可视化:
1. 物体点云 (obj_pcd) - 在相机坐标系中
2. 手部模型 (hand_model) - 转换后的抓取姿态
3. 机械臂基坐标系
4. 相机坐标系

## 文件说明

### 1. `visualize_camera_view.py`
新创建的可视化脚本,包含以下功能:
- `visualize_from_camera_view()`: 主可视化函数
- `load_and_visualize_from_debug_files()`: 从调试文件加载数据并可视化
- `test_with_synthetic_data()`: 使用合成数据测试可视化功能

### 2. `api_service.py` (已修改)
添加了保存调试文件的代码:
- `debug_output/camera_extrinsics.npy`: 相机外参矩阵 (4x4)
- `debug_output/best_grasp_base.npy`: 最佳抓取姿态,在基坐标系中 (23维)
- `debug_output/object_colored.ply`: 物体点云,在相机坐标系中

## 使用方法

### 方法1: 使用合成数据测试 (快速验证)

```bash
python visualize_camera_view.py --test
```

这会创建一个合成的场景:
- 在相机前方 0.5m 处创建一个立方体点云
- 在基坐标系原点附近创建一个手部抓取姿态
- 使用示例相机外参

**预期结果**: 应该能看到手部模型、点云、以及两个坐标系(相机和基座)

### 方法2: 使用真实数据 (验证实际坐标转换)

1. **先运行 API 服务并处理一次抓取请求**:
   ```bash
   python api_service.py
   ```

2. **发送抓取请求** (使用你的客户端代码),这会生成调试文件到 `debug_output/` 目录

3. **运行可视化**:
   ```bash
   python visualize_camera_view.py
   ```

**预期结果**:
- 应该能看到实际的物体点云
- 手部模型应该在合理的位置(靠近物体)
- 相机坐标系应该在原点
- 基坐标系应该在相机外参定义的位置

## 验证步骤

### 1. 检查坐标系方向
- **相机坐标系** (Camera Frame):
  - 原点在 (0, 0, 0)
  - X轴: 红色,指向右方
  - Y轴: 绿色,指向下方
  - Z轴: 蓝色,指向前方

- **基坐标系** (Base Frame):
  - 原点由相机外参定义
  - 坐标轴方向由相机外参的旋转矩阵定义

### 2. 检查物体点云位置
- 点云应该在相机前方(Z > 0)
- 点云应该在合理的距离范围内(例如 0.3-1.0m)

### 3. 检查手部模型位置
- 手部模型应该靠近物体点云
- 手部的位置和方向应该看起来能够抓取物体

### 4. 对比实际相机图像
- 将可视化结果与实际相机看到的场景对比
- 检查物体在相机中的位置是否一致
- 检查手部在相机中的位置是否符合预期

## 调试输出说明

运行可视化时,会打印详细的调试信息:

```
相机外参 T_base_camera:
[[...]]

R_camera_base (基坐标系在相机坐标系中的旋转):
[[...]]

t_camera_base (基坐标系原点在相机坐标系中的位置):
[...]

抓取姿态 (在基坐标系中):
  位置: [...]
  四元数 (wxyz): [...]

手部姿态 (在相机坐标系中):
  位置: [...]
  旋转矩阵:
[[...]]
```

## 常见问题

### Q1: 手部模型没有出现在点云附近
**可能原因**:
- 坐标转换有误
- 相机外参不正确
- 点云归一化参数有误

**解决方法**:
- 检查 `transform_grasps_to_base_frame` 函数的实现
- 验证相机外参矩阵是否正确
- 检查物体中心和缩放因子

### Q2: 坐标系方向不对
**可能原因**:
- 相机外参的旋转矩阵有误
- 坐标系约定不一致

**解决方法**:
- 确认相机和机器人的坐标系约定
- 检查相机标定结果

### Q3: 可视化窗口无法显示
**可能原因**:
- Plotly 环境配置问题
- 没有安装必要的依赖

**解决方法**:
```bash
pip install plotly
```

## 代码结构

```python
visualize_from_camera_view(
    obj_pcd_in_camera,      # 物体点云,在相机坐标系 (N, 3) 米
    grasp_in_base,          # 抓取姿态,在基坐标系 (23,) [x,y,z,qw,qx,qy,qz,joints]
    camera_extrinsics,      # 相机外参 T_base_camera (4, 4)
    robot='allegro_right'   # 机器人类型
)
```

关键步骤:
1. 从相机外参计算 `T_camera_base` (逆变换)
2. 将基坐标系中的抓取姿态转换到相机坐标系
3. 更新手部模型的运动学
4. 在相机坐标系中绘制所有元素

## 修改说明

### `api_service.py` 修改 (api_service.py:358-361)
```python
# Save additional debug files for visualization
np.save(output_dir / "camera_extrinsics.npy", camera_extrinsics_matrix)
np.save(output_dir / "best_grasp_base.npy", grasp_qt_base[best_grasp_index])
print(f"Saved camera_extrinsics.npy and best_grasp_base.npy to {output_dir}/")
```

这样每次处理抓取请求时,都会保存必要的调试文件供可视化使用。
