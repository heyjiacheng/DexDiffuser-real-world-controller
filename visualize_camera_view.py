"""
验证坐标转换的可视化工具
从相机视角可视化:
1. obj_pcd (物体点云,在相机坐标系中)
2. hand_model (手部模型,转换后的抓取姿态)
3. 机械臂基坐标系
4. 相机坐标系
"""

import numpy as np
import torch
from plotly import graph_objects as go
from scipy.spatial.transform import Rotation as R
from utils.handmodel import get_handmodel
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


def plot_point_cloud(pts, color='lightblue', mode='markers', name='Point Cloud'):
    """绘制点云"""
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        name=name,
        marker={
            'color': color,
            'size': 3,
            'opacity': 1
        }
    )


def plot_coordinate_frame(origin, rotation_matrix, scale=0.1, name_prefix='Frame'):
    """
    绘制坐标系
    Args:
        origin: (3,) 原点位置
        rotation_matrix: (3, 3) 旋转矩阵
        scale: 坐标轴长度
        name_prefix: 名称前缀
    """
    # X, Y, Z 轴的方向
    x_axis = rotation_matrix[:, 0] * scale  # 红色
    y_axis = rotation_matrix[:, 1] * scale  # 绿色
    z_axis = rotation_matrix[:, 2] * scale  # 蓝色

    traces = []

    # X轴 (红色)
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0] + x_axis[0]],
        y=[origin[1], origin[1] + x_axis[1]],
        z=[origin[2], origin[2] + x_axis[2]],
        mode='lines+text',
        name=f'{name_prefix} X',
        line=dict(color='red', width=5),
        text=['', 'X'],
        textposition='top center'
    ))

    # Y轴 (绿色)
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0] + y_axis[0]],
        y=[origin[1], origin[1] + y_axis[1]],
        z=[origin[2], origin[2] + y_axis[2]],
        mode='lines+text',
        name=f'{name_prefix} Y',
        line=dict(color='green', width=5),
        text=['', 'Y'],
        textposition='top center'
    ))

    # Z轴 (蓝色)
    traces.append(go.Scatter3d(
        x=[origin[0], origin[0] + z_axis[0]],
        y=[origin[1], origin[1] + z_axis[1]],
        z=[origin[2], origin[2] + z_axis[2]],
        mode='lines+text',
        name=f'{name_prefix} Z',
        line=dict(color='blue', width=5),
        text=['', 'Z'],
        textposition='top center'
    ))

    return traces


def visualize_from_camera_view(
    obj_pcd_in_camera,
    grasp_in_base,
    camera_extrinsics,
    robot='allegro_right'
):
    """
    从基坐标系视角可视化所有元素

    Args:
        obj_pcd_in_camera: (N, 3) 物体点云,在相机坐标系中 (米)
        grasp_in_base: (25,) 抓取姿态,在基坐标系中 [x,y,z,qw,qx,qy,qz,joints(16),rot6d(6)]
                       或者 (23,) [x,y,z,qw,qx,qy,qz,joints(16)]
        camera_extrinsics: (4, 4) 相机外参矩阵 T_base_camera
        robot: 机器人类型
    """
    print("="*60)
    print("从基坐标系视角可视化抓取")
    print("="*60)

    # 提取基坐标系到相机坐标系的变换
    R_base_camera = camera_extrinsics[:3, :3]
    t_base_camera = camera_extrinsics[:3, 3]

    # 计算相机坐标系到基坐标系的变换 (逆变换)
    R_camera_base = R_base_camera.T
    t_camera_base = -R_camera_base @ t_base_camera

    print(f"\n相机外参 T_base_camera:")
    print(camera_extrinsics)
    print(f"\n相机在基坐标系中的位置: {t_base_camera}")
    print(f"相机在基坐标系中的旋转:\n{R_base_camera}")

    # 1. 创建手部模型
    hand_model = get_handmodel(batch_size=1, device='cuda',
                               urdf_path='data/urdf',
                               robot=robot)

    # 2. 从抓取姿态中提取数据 (在基坐标系中)
    if len(grasp_in_base) == 23:
        # 格式: [qw,qx,qy,qz,x,y,z,joints(16)]
        quat_base = grasp_in_base[:4]
        trans_base = grasp_in_base[4:7]
        joint_angles = grasp_in_base[7:23]
    else:
        raise ValueError(f"Unsupported grasp format length: {len(grasp_in_base)}")

    # 将四元数转换为旋转矩阵
    rot_scipy = R.from_quat(np.roll(quat_base, -1))  # wxyz -> xyzw for scipy
    R_base_hand = rot_scipy.as_matrix()  # (3, 3)

    print(f"\n抓取姿态 (在基坐标系中):")
    print(f"  位置: {trans_base}")
    print(f"  四元数 (wxyz): {quat_base}")
    print(f"  旋转矩阵:\n{R_base_hand}")

    # 3. 将物体点云从相机坐标系转换到基坐标系
    # obj_pcd_base = R_base_camera @ obj_pcd_camera.T + t_base_camera
    obj_pcd_base = (R_base_camera @ obj_pcd_in_camera.T).T + t_base_camera

    print(f"\n物体点云转换:")
    print(f"  相机坐标系中心: {obj_pcd_in_camera.mean(axis=0)}")
    print(f"  基坐标系中心: {obj_pcd_base.mean(axis=0)}")

    # 4. 构建手部模型的输入 (q_tr 格式: [trans(3), rot6d(6), joints(16)])
    # 手部姿态已经在基坐标系中，直接使用
    # 将旋转矩阵转换为 rot6d
    rot6d_base = R_base_hand.T[:2].reshape([6])  # 转置后取前两列

    # 组合成 q_tr
    q_tr = np.concatenate([trans_base, rot6d_base, joint_angles])
    q_tr_torch = torch.from_numpy(q_tr).unsqueeze(0).float().to('cuda')

    print(f"\nq_tr (手部模型输入,在基坐标系中):")
    print(f"  trans: {trans_base}")
    print(f"  rot6d: {rot6d_base}")
    print(f"  joints: {joint_angles}")

    # 5. 更新手部运动学
    hand_model.update_kinematics(q=q_tr_torch)

    # 6. 创建可视化数据
    vis_data = []

    # 添加物体点云 (在基坐标系中)
    vis_data.append(plot_point_cloud(obj_pcd_base, color='pink', name='Object Point Cloud'))

    # 添加手部模型
    hand_plotly_data = hand_model.get_plotly_data(opacity=0.5, color='lightblue')
    vis_data.extend(hand_plotly_data)

    # 7. 添加基坐标系 (原点在 (0,0,0))
    base_frame_traces = plot_coordinate_frame(
        origin=np.array([0, 0, 0]),
        rotation_matrix=np.eye(3),
        scale=0.15,
        name_prefix='Base'
    )
    vis_data.extend(base_frame_traces)

    # 8. 添加相机坐标系 (在基坐标系中的位置)
    camera_frame_traces = plot_coordinate_frame(
        origin=t_base_camera,
        rotation_matrix=R_base_camera,
        scale=0.15,
        name_prefix='Camera'
    )
    vis_data.extend(camera_frame_traces)

    # 9. 创建并显示图形
    fig = go.Figure(data=vis_data)

    # 设置观察视角 (从一个合适的角度观察基坐标系)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # 观察位置
                center=dict(x=0, y=0, z=0.3),  # 看向基坐标系附近
                up=dict(x=0, y=0, z=1)  # Z轴朝上
            ),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title='从基坐标系视角看抓取 (Base Frame View)',
        showlegend=True
    )

    fig.show()
    print("\n可视化完成!")
    print("="*60)


def load_and_visualize_from_debug_files():
    """
    从调试文件中加载数据并可视化
    假设已经运行过 api_service.py 生成了相关文件
    """
    import open3d as o3d

    # 1. 加载物体点云 (从相机坐标系)
    # 这里我们需要重新生成物体点云,因为之前保存的是归一化后的
    # 或者从 debug_output/object_colored.ply 加载
    pcd_path = "debug_output/object_colored.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)
    obj_pcd_camera = np.asarray(pcd.points)

    print(f"加载点云: {pcd_path}")
    print(f"点云形状: {obj_pcd_camera.shape}")
    print(f"点云统计 (在相机坐标系中):")
    print(f"  均值: {obj_pcd_camera.mean(axis=0)}")
    print(f"  标准差: {obj_pcd_camera.std(axis=0)}")

    # 2. 加载相机外参
    camera_extrinsics_path = "debug_output/camera_extrinsics.npy"
    try:
        camera_extrinsics = np.load(camera_extrinsics_path)
        print(f"\n加载相机外参: {camera_extrinsics_path}")
    except FileNotFoundError:
        print(f"\n警告: 未找到 {camera_extrinsics_path}")
        print("使用默认相机外参 (单位矩阵)")
        camera_extrinsics = np.eye(4)

    # 3. 加载最佳抓取姿态 (在基坐标系中)
    # 这需要从 API 响应中获取,或者从保存的文件中加载
    # 这里我们创建一个示例抓取姿态
    grasp_base_path = "debug_output/best_grasp_base.npy"
    try:
        grasp_in_base = np.load(grasp_base_path)
        print(f"\n加载抓取姿态: {grasp_base_path}")
    except FileNotFoundError:
        print(f"\n警告: 未找到 {grasp_base_path}")
        print("请先运行 api_service.py 生成抓取姿态")
        print("使用示例抓取姿态")
        # 示例: [x,y,z,qw,qx,qy,qz,joints(16)]
        grasp_in_base = np.concatenate([
            np.array([0.5, 0.0, 0.3]),  # 位置
            np.array([1.0, 0.0, 0.0, 0.0]),  # 四元数
            np.zeros(16)  # 关节角度
        ])

    # 4. 调用可视化函数
    visualize_from_camera_view(
        obj_pcd_in_camera=obj_pcd_camera,
        grasp_in_base=grasp_in_base,
        camera_extrinsics=camera_extrinsics,
        robot='allegro_right'
    )


def test_with_synthetic_data():
    """
    使用合成数据测试可视化功能
    """
    print("\n使用合成数据测试...")

    # 1. 创建合成物体点云 (在相机坐标系中)
    # 在相机前方 0.5m 处创建一个立方体
    n_points = 1000
    cube_size = 0.05
    obj_pcd_camera = np.random.rand(n_points, 3) * cube_size
    obj_pcd_camera[:, 2] += 0.5  # Z方向偏移
    obj_pcd_camera[:, 0] -= cube_size / 2  # 居中
    obj_pcd_camera[:, 1] -= cube_size / 2

    print(f"合成点云形状: {obj_pcd_camera.shape}")
    print(f"点云中心: {obj_pcd_camera.mean(axis=0)}")

    # 2. 创建相机外参 (基坐标系相对于相机坐标系)
    # 假设相机在机器人基坐标系上方 0.3m, 前方 0.2m
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, 3] = [0.2, 0.0, 0.3]  # 基坐标系原点在相机坐标系中的位置

    print(f"\n相机外参:\n{camera_extrinsics}")

    # 3. 创建抓取姿态 (在基坐标系中)
    # 假设手在基坐标系原点附近
    grasp_in_base = np.concatenate([
        np.array([0.0, 0.0, 0.2]),  # 位置
        np.array([1.0, 0.0, 0.0, 0.0]),  # 四元数 (单位旋转)
        np.zeros(16)  # 关节角度
    ])

    print(f"\n抓取姿态 (基坐标系):")
    print(f"  位置: {grasp_in_base[:3]}")
    print(f"  四元数: {grasp_in_base[3:7]}")

    # 4. 调用可视化
    visualize_from_camera_view(
        obj_pcd_in_camera=obj_pcd_camera,
        grasp_in_base=grasp_in_base,
        camera_extrinsics=camera_extrinsics,
        robot='allegro_right'
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 使用合成数据测试
        test_with_synthetic_data()
    else:
        # 从调试文件加载数据
        load_and_visualize_from_debug_files()
