import os
import torch
import random
import numpy as np
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

# from utils.io import mkdir_if_not_exists
from models import  create_ddpm, create_evaluator
import open3d as o3d
from sample import load_ckpt
from copy import deepcopy

from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from scipy.spatial.transform import Rotation as R
from utils.handmodel import get_handmodel, angle_denormalize, trans_denormalize

from plotly import graph_objects as go
import sys
from bps_torch.bps import bps_torch
import time

def plot_point_cloud(pts, color='lightblue', mode='markers'):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker={
            'color': color,
            'size': 3,
            'opacity': 1
        }
    )

def convert_transformation_matrix_to_qt(transformation_matrix):
    """get T from qt
    Args:
        
        transformation_matrix: matrices, (B, 4, 4)

    Returns:
        qt: bacth arr, (B, 7) # wxyz, xyz
       
    """
    if len(transformation_matrix.shape) == 2:
        transformation_matrix = np.expand_dims(transformation_matrix, 0)

    t = transformation_matrix[:,:3,3]
    Rs = transformation_matrix[:,:3,:3]

    R_rotmat = R.from_matrix(Rs)
    q = R_rotmat.as_quat()
    q = np.roll(q, 1, axis=1)


    qt = np.zeros((1,7))
    qt = np.repeat(qt, transformation_matrix.shape[0], 0)
    qt[:,:4] = q
    qt[:,4:] = t

    return qt

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.
    Angles are assumed to be in radians.
    :param roll: Rotation angle around X-axis (in radians)
    :param pitch: Rotation angle around Y-axis (in radians)
    :param yaw: Rotation angle around Z-axis (in radians)
    :return: 3x3 rotation matrix
    """
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    return Rz @ Ry @ Rx

def mirror_xzplane_point_cloud(point_cloud):
    mirror_xz_pcd = point_cloud.clone()
    # mirror_x_pcd = point_cloud.copy()
    mirror_xz_pcd[:,1] = -point_cloud[:,1]

    return mirror_xz_pcd

def mirror_xzplane_grasp(grasp):
    '''
    grasp: [N, 9]
    '''
    mirror_xz_grasp = grasp.clone()
    grasp_mat = robust_compute_rotation_matrix_from_ortho6d(grasp[:,3:9])
    grasp_mat[:,2,:] = -grasp_mat[:,2,:]
    mirror_xz_grasp[:,3:9] = grasp_mat.transpose(1, 2)[:, :2].reshape([-1, 6])
    # mirror_xz_grasp[:,3:9] = grasp_mat.transpose(1, 2)[:, :, :2].reshape([-1, 6])
    mirror_xz_grasp[:,1] = -mirror_xz_grasp[:,1]
    
    return mirror_xz_grasp




class DexDiffuser:

    def __init__(self, cfg:DictConfig) -> None:
        
        self.device='cuda'


        basis_bps_set = np.load('models/basis_point_set.npy')
        self.bps  = bps_torch(n_bps_points=4096,
                            n_dims=3,
                            custom_basis=basis_bps_set)
        
        self.refinement = cfg.refinement
        self.method = cfg.model

        if cfg.model == 'ffhnet':
            sys.path.append('ffhnet_allegro')
            from FFHNet.models.ffhnet import FFHNet
            # from FFHNet.config.eval_config import EvalConfig
            # cfg1 = EvalConfig().parse()
            cfg1 = load_config("ffhnet.yaml")
            self.ffhnet = FFHNet(cfg1)
            self.ffhnet.load_ffhgenerator(epoch=50, load_path='ffhnet_allegro/checkpoints/2024-06-16T21_15_33')
            self.evaluator = create_evaluator(cfg, pos_enc_multires=None, use_bn=True)
            load_ckpt(self.evaluator, path='outputs/2024-06-13_16-31-37_Lnull/ckpts/model_20.pth')
            
        else:
            ## create model and load ckpt
            self.model = create_ddpm(cfg)
            self.model.to(device=self.device)
            self.model.eval()

            self.evaluator = create_evaluator(cfg, pos_enc_multires=[10,4,-1])
            load_ckpt(self.evaluator, path=cfg.evaluator_ckpt_pth)
            
        
            self.normalize_x = True
            self.normalize_x_trans = False

            if cfg.model.scene_model.name == 'obj_bps':
                sampler_pth = cfg.sampler_bps_ckpt_pth
            elif cfg.model.scene_model.name == 'PointNet2':
                sampler_pth = cfg.sampler_pn2_ckpt_pth
                sampler_pth = 'outputs/2024-05-27_20-40-08_dexdiff_dexgn_pn2/ckpts/model_20.pth'
            else:
                raise NotImplementedError
            ## if your models are seperately saved in each epoch, you need to change the model path manually
            load_ckpt(self.model, path=sampler_pth)
        self.evaluator.to(device=self.device)
        self.evaluator.eval()

        if self.refinement:
            from refine import RefineNN
            self.refineNN = RefineNN(self.evaluator)

            
        if cfg.guid_scale is None:
            self.guid_param = None
        else:
            self.guid_param = {'evaluator':self.evaluator,
                            'guid_scale': cfg.guid_scale}
        logger.info('Initialized!') 
    

    def visualize_func(self,q_tr,obj_pcd,robot, pts = None):
        hand_model = get_handmodel(batch_size=1, device='cuda',
                                    urdf_path='data/urdf',
                                    robot=robot)

        hand_model.update_kinematics(q=q_tr.to('cuda'))
        vis_data = [plot_point_cloud(obj_pcd, color='pink')]
        vis_data += hand_model.get_plotly_data(opacity=0.5, color='lightblue')
        
        fig = go.Figure(data=vis_data)        
        fig.show()

    def get_q_r(self,mirror,trans,rot6d):
        rot = robust_compute_rotation_matrix_from_ortho6d(torch.from_numpy(rot6d)[None,:]).numpy()
        mat = np.eye(4)
        mat[:3,:3] = rot[0]
        mat[:3, 3] = trans
        mirror_mat = mirror @ mat @ np.linalg.inv(mirror)
        mirror_rot6d = mirror_mat[:3,:3].transpose(0,1)[:2].reshape([6])
        mirror_trans = mirror_mat[:3,3]
        # print(mat)
        # print(mirror_mat)
        return mirror_trans, mirror_rot6d

    def get_q_r_anchors(self,trans,rot6d):
        homo_pts = np.array([[0,0,0],
                                [1,0,0],
                                [0,1,0],
                                [0,0,1]]) * 0.1
        
        homo_pts_r = np.array([[0,0,0],
                                [1,0,0],
                                [0,-1,0],
                                [0,0,1]]) * 0.1
        
        rot = robust_compute_rotation_matrix_from_ortho6d(torch.from_numpy(rot6d)[None,:]).numpy()[0]

        transformed_homo_pts_mirror = (rot@homo_pts.T).T
        transformed_homo_pts = deepcopy(transformed_homo_pts_mirror)
        transformed_homo_pts_mirror[:,1] = -transformed_homo_pts_mirror[:,1]


        # transform = np.linalg.pinv(transformed_homo_pts_mirror) @ homo_pts_r
        transform = np.linalg.pinv(homo_pts_r) @ transformed_homo_pts_mirror
        transformed_homo_pts_1 = (transform @ homo_pts_r.T).T
        # print()
        trans_mirror = trans.copy()
        trans_mirror[1] = -trans_mirror[1]

        return trans_mirror, transform[:2].reshape([6]), [transformed_homo_pts, transformed_homo_pts_1]

    def sample_grasps(self,obj_pcd, num_samples = 32):

        obj_pcd_torch = torch.from_numpy(obj_pcd).\
                        unsqueeze(0).repeat(num_samples,1,1).to(self.device)

        data = {'x': torch.randn(num_samples, 25, device=self.device),
                'pos': obj_pcd_torch,
                }
        # import pdb;pdb.set_trace()
        data['obj_bps'] = self.bps.encode(obj_pcd_torch,feature_type=['dists'])['dists']
        # outputs = self.model.sample(data, k=1).squeeze(1)[:, -1, :].to(torch.float64).cpu()
        if self.method == 'ffhnet':
            print("Using FFHNet!!!")
            grasps = self.ffhnet.generate_grasps(data['obj_bps'][0].detach().cpu().numpy(), n_samples=num_samples, return_arr=False)
            outputs = torch.cat([grasps['transl'], grasps['rot_6D'], grasps['joint_conf']], dim=1)

        else:
            print("Using DexDiffuser!!!")
            if self.guid_param is not None:
                print("Using guidance model!!!")
            start=time.time()
            outputs = self.model.sample(data, k=1,guid_param=self.guid_param).squeeze(1)[:, -1, :]#.to(torch.float64).cpu() # (1,)
            print("Time taken for sampling grasps: ", time.time()-start)

        if self.refinement:
            print("Refining grasps!!!")
            data['x_t'] = outputs.detach()
            start=time.time()
            outputs, score = self.refineNN.improve_grasps_sampling_based(data, num_refine_steps=100, delta_translation=0.001)
            outputs = torch.from_numpy(outputs).to(self.device).to(torch.float32)
            print("Time taken for refinement grasps: ", time.time()-start)
    
        data['x_t'] = outputs
        score = self.evaluator(data)['p_success'].detach().cpu().numpy().squeeze()

        outputs[:,9:] = angle_denormalize(joint_angle=outputs[:,9:])
        outputs = outputs.float().detach().cpu()

        rot_ = robust_compute_rotation_matrix_from_ortho6d(outputs[:,3:9])
        rot_mat = np.tile(np.eye(4),(num_samples,1,1))
        # rot_mat[:,:3,:3] = np.transpose(rot_, (0, 2, 1))          # rotation
        rot_mat[:,:3,:3] = rot_            # rotation
        rot_mat[:,:3,3] = outputs[:,:3]   # translation
        qt = convert_transformation_matrix_to_qt(rot_mat)
        # print(qt.shape)
        grasp_qt = np.concatenate([qt,outputs[:,9:].numpy()], axis=1)

        max_score_index = np.argmax(score)
        np.save('pcds/grasp.npy', outputs[max_score_index:max_score_index+1].detach().cpu().numpy())
        # np.save("/home/haofei/Desktop/MyPaper/DexDiffuser/pcds/pcd.npy", obj_pcd)

        # highest_score_grasp = grasp_qt[max_score_index]

        self.visualize_func(outputs[max_score_index:max_score_index+1],  obj_pcd_torch[0].detach().cpu(),'allegro_right')

        return grasp_qt, score # qt representation wxyz, xyz



# Define a function to load YAML configuration
def load_config(config_path):
    # Use Hydra's OmegaConf to load YAML config
    config = OmegaConf.load(config_path)
    return config

if __name__ == "__main__":
    # Specify the path to your YAML configuration file
    config_path = "configs/sample.yaml"
    cfg = load_config(config_path)
    # Access configuration parameters
    cfg.model = load_config("configs/model/unet_grasp_bps.yaml")
    cfg.diffuser = load_config("configs/diffuser/ddpm.yaml")
    cfg.task = load_config("configs/task/grasp_gen_ur_dexgn_slurm.yaml")
    cfg.refinement = True
    # print(config)
    dex = DexDiffuser(cfg)
    # pcd_np = np.random.rand(2048,3) * 0.05
    pcd_np = np.load("pcds/obj_pcd.npy")
    grasp_np = dex.sample_grasps(pcd_np,1)
    # grasp_np = dex.sample_grasps(pcd_np,1)
    # grasp_np = dex.sample_grasps(pcd_np,1)
    # grasp_np = dex.sample_grasps(pcd_np,1)
    # grasp_np = dex.sample_grasps(pcd_np,1)
    # grasp_np = dex.sample_grasps(pcd_np,1)
    print(grasp_np[0])