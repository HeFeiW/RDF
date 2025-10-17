# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
import numpy as np
from Siren import Siren, gradient, divergence
np.set_printoptions(threshold=np.inf)
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(CUR_DIR,"panda_layer"))
import glob
import trimesh
import utils
import mesh_to_sdf
import skimage
from parallel_robot_layer import ParallelRobotLayer
from serial_robot_layer import SerialRobotLayer
from ParallelSiren import ParallelSiren, load_multiple_siren_weights
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class QSDF():
    def __init__(self,robot,paths,device,used_links = None):
        # - Robot should be a serial robot,
        #   in order to initialize the serial robot layer and corresponding mesh model,
        #   and accelerate the computation of SDF
    
        self.device = device   
        self.paths = paths 
        self.robot = robot
        self.domain_min = -1.0
        self.domain_max = 1.0
        assert 'SerialRobotLayer' in str(type(robot)),\
            "robot should be an instance of SerialRobotLayer"
        if used_links is None:
            self.used_links = self.robot.all_links.copy()
        else:
            self.used_links = used_links.copy()
        for link in self.used_links:
            if self.robot.Link2Mesh[link] is None:
                print(f"Link '{link}' does not have an associated mesh, removing it from used_links.")
                self.used_links.remove(link)
        
        self.link_model_type = 'siren' # 'bp' or 'siren'
        self.link_model, self.model_info = self.load_mesh_model()
        # link_model: the neural network model for all links, receives (B*K,N,3) and outputs (B*K,N,1)
        # self.model_info: {link_name: {'mesh_name': mesh_name, 'offset': offset, 'scale': scale}}
        
        print(f'QSDF initialized with {self.link_model_type} mesh model')
        print('used_links:',self.used_links)
        
    def load_mesh_model(self):
        mesh_model_dict = {}
        model_path = self.paths['model']
        model_load = torch.load(model_path,map_location='cpu')
        model_used = {}
        model_info = {}
        if self.link_model_type == 'siren':
            for link in self.used_links:
                mesh_name = self.robot.Link2Mesh[link]
                if mesh_name is not None:
                    if mesh_name not in model_load.keys():
                        raise ValueError(f"Mesh model for link '{link}' with mesh name '{mesh_name}' not found in the loaded model.")
                    model_used[mesh_name] = model_load[mesh_name]
                    model_info[mesh_name] = {
                        'mesh_name': mesh_name,
                        'offset': model_load[mesh_name]['offset'],
                        'scale': model_load[mesh_name]['scale']
                    }
            parallel_siren = ParallelSiren(
                num_networks=len(model_used), in_features=3, out_features=1,
                hidden_features=256, hidden_layers=3, outermost_linear=True
            ).to(self.device)
            parallel_siren = load_multiple_siren_weights(model_used, parallel_siren)
            parallel_siren.eval()
        else:
            # 抛出异常
            raise ValueError("Unsupported mesh model type. Use 'siren'.")
        return parallel_siren, model_info
        
    def sdf_to_mesh(self,nbData,use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        domain = torch.linspace(self.domain_min,self.domain_max,nbData).to(self.device)
        grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain)
        grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
        p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)   
        K = len(self.model_info)
        p = p.unsqueeze(0).expand(K,-1,-1) # (K, nbData**3, 3)
        # split data to deal with memory issues
        p_split = torch.split(p, 10000, dim=1)
        d =[]
        with torch.no_grad():
            for p_s in p_split:
                d_s = self.link_model(p_s)
                d.append(d_s.reshape(K,-1,1))
            d = torch.cat(d,dim=1) # (K, nbData**3, 1)
            d = d.squeeze(-1) # (K, nbData**3)
        for k in range(K):
            mesh_name = list(self.model_info.keys())[k]
            mesh_name_list.append(mesh_name)
            verts, faces, normals, values = skimage.measure.marching_cubes(
                d[k].view(nbData,nbData,nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(self.domain_max-self.domain_min)/nbData] * 3)
            )
            verts = verts - [1,1,1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list,mesh_name_list

    def create_surface_mesh(self,nbData,vis =False):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(nbData)
        for verts, faces,mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            if vis:
                print(f'visualizing {mesh_name} mesh')
                rec_mesh.show()
    
    def get_serial_sdf_batch(self,x,pose,theta,serial_idx,use_derivative = True, used_links = None,return_index=False):
        # x: (Nx,3)  query points in world frame
        # pose: (B,4,4)  base pose in world frame
        # theta: (B,DoF)  joint angles of the serial
        # used_links: list of link names to use
        if serial_idx is None or serial_idx >= len(self.robot.serials):
            raise ValueError("Invalid serial index")
        serial = self.robot.serials[serial_idx]
        if used_links is None:
            used_links = serial.all_links
        used_links = [link for link in used_links if serial.Link2Mesh[link] is not None]
        B = len(theta)
        N = len(x)
        K = len(used_links)
        
        offset = torch.cat([self.link_model[serial.Link2Mesh[link]]['offset'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()# offset: (B*K,3)
        
        scale = torch.tensor([self.link_model[self.robot.Link2Mesh[link]]['scale'] for link in used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()# scale: (B*K)
        trans = serial.get_link_mesh_transformations(pose, theta)#trans: (K+1, B, 4, 4)
        trans = torch.stack([trans[link] for link in used_links],dim=0)
        # trans = trans[used_indices]  # trans: (K, B, 4, 4)
        trans = trans.transpose(1,0) # (B, K, 4, 4)
        trans = trans.reshape(-1,4,4).float() # trans: (B*K, 4, 4)
        x_robot_frame_batch = utils.transform_points(x.float(),torch.linalg.inv(trans).float(),device=self.device) # B*K,N,3
        # x_robot_frame_batch: (B*K,N,3); x: （N,3）
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled/scale.unsqueeze(-1).unsqueeze(-1) #B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled>1.0-1e-2,1.0-1e-2,x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded<-1.0+1e-2,-1.0+1e-2,x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded # res_x: B*K,N,3
        
        # # --- debug ---
        # theta_debug = theta[0]
        # print(theta_debug)
        # theta_debug = theta_debug.unsqueeze(0)
        # scene = trimesh.Scene()
        # mesh = serial.get_forward_robot_mesh(pose[0].unsqueeze(0), theta_debug,used_links=used_links)[0]
        # scene.add_geometry(mesh)
        # scene.show()
        # # --- debug ---
        batch_size = 10000
        sdf_values = []
        if not use_derivative:
            for i in range(0, N, batch_size):
                batch_sdf, coords = self.link_model(x_bounded[i:i+batch_size])
                sdf_values.append(batch_sdf)
            sdf = torch.cat(sdf_values, dim=0)
            # phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            # phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            # weights_near = torch.cat([self.link_model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
            # sdf
            # sdf = torch.einsum('ijk,ik->ij',phi,weights_near).reshape(K,B,N).transpose(0,1).reshape(B*K,N) # B*K,N
            # np.set_printoptions(threshold=np.inf)
            # print(sdf.reshape(B,K,N).transpose(0,1)[0].cpu().detach().numpy())
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*scale.reshape(B,K).unsqueeze(-1)#sdf  (B,K,1)
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:   
            phi,dphi = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1),dphi],dim=-1)
            phi_cat = phi_cat.reshape(B,K,N,-1,4).transpose(0,1).reshape(K,B*N,-1,4) # K,B*N,-1,4

            weights_near = torch.cat([self.link_model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)

            output = torch.einsum('ijkl,ik->ijl',phi_cat,weights_near).reshape(K,B,N,4).transpose(0,1).reshape(B*K,N,4)
            sdf = output[:,:,0]
            gradient = output[:,:,1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*(scale.reshape(B,K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # print(f'sdf_values:{sdf_value}, idx:{idx}')
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient,dim=-1)
            gradient = torch.nn.functional.normalize(gradient,dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = trans[:,:3,:3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl',fk_rotation,gradient.transpose(1,2)).transpose(1,2).reshape(B,K,N,3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)
            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B,K,N,3)
            gradient_value = torch.gather(gradient_base_frame,1,idx_grad)[:,0,:,:]
            # gradient_value = None
            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value
    def get_serial_sdf_with_joints_grad_batch(self,x,pose,theta,link_model,used_links = None, serial_idx = None):
        # theta: (B,DoF)
        # pose: (B,4,4)
        if serial_idx is None or serial_idx >= len(self.robot.serials):
            raise ValueError("Invalid serial index")
        serial = self.robot.serials[serial_idx]
        if used_links is None:
            used_links = serial.all_links
        used_links = [link for link in used_links if serial.Link2Mesh[link]is not None]
        delta = 0.001
        B = theta.shape[0]
        DoF = theta.shape[1]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,DoF,DoF)+ torch.eye(DoF,device=self.device).unsqueeze(0).expand(B,DoF,DoF)*delta).reshape(B,-1,DoF)
        theta = torch.cat([theta,d_theta],dim=1) # (B,(DoF+1),DoF)
        pose = pose.unsqueeze(1).expand(B,(DoF+1),4,4).reshape(B*(DoF+1),4,4)
        theta = theta.reshape(B*(DoF+1),DoF)
        sdf,_ = self.get_serial_sdf_batch(x,pose,theta,self.link_model,use_derivative = False, used_links = used_links, serial_idx = serial_idx)
        sdf = sdf.reshape(B,(DoF+1),-1)
        d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
        return sdf[:,0,:], d_sdf.transpose(1,2)
    def get_sdf_with_points_grad(self,x,pose,theta,use_derivative = True,return_index=False, used_links = None):
        if used_links is None:
            used_links = self.used_links
        B = len(theta) # theta: (B, DoF)
        N = len(x) # x: (N,3)
        K = len(self.used_links)
        print(x.shape)
        offset = torch.cat([self.model_info[self.robot.Link2Mesh[link]]['offset'].unsqueeze(0) for link in self.used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()# offset: (B*K,3)
        
        scale = torch.tensor([self.model_info[self.robot.Link2Mesh[link]]['scale'] for link in self.used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()# scale: (B*K)
        
        trans = self.robot.get_link_mesh_transformations(pose, theta)#trans: (K+1, B, 4, 4)
        trans = torch.stack([trans[link] for link in self.used_links],dim=0)
        # trans = trans[used_indices]  # trans: (K, B, 4, 4)
        trans = trans.transpose(1,0) # (B, K, 4, 4)
        trans = trans.reshape(-1,4,4).float() # trans: (B*K, 4, 4)
        print(trans.shape)
        x_robot_frame_batch = utils.transform_points(x.float(),torch.linalg.inv(trans).float(),device=self.device) # B*K,N,3
        # x_robot_frame_batch: (B*K,N,3); x: （N,3）
        
        
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled/scale.unsqueeze(-1).unsqueeze(-1) #B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled>1.0-1e-2,1.0-1e-2,x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded<-1.0+1e-2,-1.0+1e-2,x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded # res_x: B*K,N,3
        x_bounded = x_bounded.reshape(B,K,N,3).permute(1,0,2,3).reshape(K,B*N,3) # K,B*N,3
        
        sdf_values = []
        batch_size = 10000
        p_split = torch.split(x_bounded, batch_size, dim=1) # x_bounded: (K,B*N,3)
        if not use_derivative:
            for p in p_split:
                # p: (K, batch_size, 3)
                batch_sdf, _ = self.link_model(p)
                sdf_values.append(batch_sdf.reshape(K,-1,1))
            sdf = torch.cat(sdf_values, dim=1).squeeze(-1).reshape(K,B,N).transpose(0,1).reshape(B*K,N)# B*K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*(scale.reshape(B,K).unsqueeze(-1))# sdf: (B,K,N)
            # delete the sdf values for links that are not used
            # print('sdf shape before filtering:', sdf.shape)
            sdf = sdf[:, [self.used_links.index(link) for link in used_links], :] if used_links is not None else sdf
            # print('sdf shape after filtering:', sdf.shape)
            sdf_value, idx = sdf.min(dim=1) # (B,N)
            # remap idx to the original link indices
            idx = idx.detach().cpu()
            # print('idx before remapping:', idx)
            if used_links is not None:
                idx = torch.tensor([self.used_links.index(link) for link in used_links])[idx]
            
            # print('idx after remapping:', idx)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:
            gradient = []
            for p in p_split:
                batch_sdf, coords = self.link_model(p)
                sdf_values.append(batch_sdf.reshape(K,-1,1))
                grad = torch.autograd.grad(outputs=batch_sdf, inputs=coords,
                                           grad_outputs=torch.ones_like(batch_sdf),
                                           create_graph=False, retain_graph=False, only_inputs=True)[0]
                gradient.append(grad.reshape(K,-1,3))
            sdf = torch.cat(sdf_values, dim=1).squeeze(-1).reshape(K,B,N).transpose(0,1).reshape(B*K,N)# B*K,N
            
            # debug: for the place where res_x is not 0, replace sdf with x_bounded.norm
            # sdf = sdf + res_x.norm(dim=-1)
            mask = (res_x.norm(dim=-1) > 0.01)
            if mask.any():
                print("Warning: res_x norm is not zero for some points, replacing sdf with x_bounded norm.")
                sdf[mask] = x_bounded[mask].norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*(scale.reshape(B,K).unsqueeze(-1))# sdf: (B,K,N)
            sdf_value, idx = sdf.min(dim=1) # (B,N)

            gradient = torch.cat(gradient, dim=1).reshape(K,B,N,3).transpose(0,1).reshape(B*K,N,3)
            gradient = res_x + torch.nn.functional.normalize(gradient,dim=-1)
            gradient = torch.nn.functional.normalize(gradient,dim=-1).float()
            fk_rotation = trans[:,:3,:3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl',fk_rotation,gradient.transpose(1,2)).transpose(1,2).reshape(B,K,N,3)
            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B,K,N,3)
            gradient_value = torch.gather(gradient_base_frame,1,idx_grad)[:,0,:,:]
            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value        
        
    def get_sdf_with_joints_grad(self,x,pose,theta,used_links = None):
        # theta: (B,DoF)
        # pose: (B,4,4)
        if used_links is None:
            used_links = self.used_links.copy()
        delta = 0.001
        B = theta.shape[0]
        DoF = theta.shape[1]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,DoF,DoF)+ torch.eye(DoF,device=self.device).unsqueeze(0).expand(B,DoF,DoF)*delta).reshape(B,-1,DoF)
        theta = torch.cat([theta,d_theta],dim=1)
        # (B,(DoF+1),DoF)
        pose = pose.unsqueeze(1).expand(B,(DoF+1),4,4).reshape(B*(DoF+1),4,4)
        theta = theta.reshape(B*(DoF+1),DoF)
        sdf,_ = self.get_sdf_with_points_grad(x,pose,theta,use_derivative = False,return_index=False, used_links=used_links)
        sdf = sdf.reshape(B,(DoF+1),-1)
        d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
        return sdf[:,0,:], d_sdf.transpose(1,2)
    
# --- some functions for visualization and testing ---
def plt_sdf_shell(distance, pose,theta,qsdf,workspace):
    pts = []
    for _ in range(len(qsdf.used_links)):
        pts.append(torch.tensor([]).to(qsdf.device))
    batch_size = 10000
    pts_cnt = 0
    while pts_cnt<10000:
        p = (torch.rand(batch_size,3)*(workspace[1]-workspace[0])+workspace[0]).to(qsdf.device)
        sdf,_,idx = qsdf.get_sdf_with_points_grad(p,pose,theta,use_derivative=False,return_index=True)
        mask = (sdf.abs()<distance).squeeze(0)
        num = mask.sum().item()
        if num!=0:
            print('number of points in the shell:',num)
        pts.append(p[mask])
        pts_cnt+=num
        idx = idx.squeeze(0)
        for k in range(len(qsdf.used_links)):
            mask_k = (idx==k)&mask
            num_k = mask_k.sum().item()
            if num_k!=0:
                print(f'number of points in the shell for link {qsdf.used_links[k]}:',num_k)
            pts[k] = torch.cat([pts[k],p[mask_k]],dim=0)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import Axes3D, art3d
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(qsdf.used_links)))
    for k in range(len(qsdf.used_links)):
        if len(pts[k])>0:
            x_k = pts[k].cpu().numpy()
            ax.scatter(x_k[:,0], x_k[:,1], x_k[:,2], color=colors[k], label=qsdf.used_links[k], s=1)
            # 图例
    ax.legend()
    # 画出workspace的边界
    ax.set_xlim(workspace[0,0], workspace[1,0])
    ax.set_ylim(workspace[0,1], workspace[1,1])
    ax.set_zlim(workspace[0,2], workspace[1,2])
    plt.show()

def plt_grad_field(pose,theta,qsdf,workspace,step=0.01):
    x = torch.arange(workspace[0,0],workspace[1,0],step).to(qsdf.device)
    y = torch.arange(workspace[0,1],workspace[1,1],step).to(qsdf.device)
    z = torch.arange(workspace[0,2],workspace[1,2],step).to(qsdf.device)
    grid_x, grid_y, grid_z= torch.meshgrid(x,y,z)
    grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
    p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(qsdf.device)   
    print('p shape',p.shape)
    sdf, grad = qsdf.get_sdf_with_points_grad(p,pose,theta,use_derivative=True)
    sdf = sdf.squeeze(1)
    print('sdf shape',sdf.shape)
    print('grad shape',grad.shape)
    grad = grad.cpu().numpy()
    p = p.unsqueeze(0).expand(len(sdf),-1,-1).cpu().numpy()
    # 只画出距离小于0.1的点的梯度
    mask = (sdf.abs()<0.1).cpu().numpy()
    print('mask shape',mask.shape)
    print('p shape',p.shape)
    p = p[mask]
    grad = grad[mask]
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import Axes3D, art3d
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(p[:,0],p[:,1],p[:,2],grad[:,0],grad[:,1],grad[:,2],length=0.01)
    # 画出workspace的边界
    ax.set_xlim(workspace[0,0], workspace[1,0])
    ax.set_ylim(workspace[0,1], workspace[1,1])
    ax.set_zlim(workspace[0,2], workspace[1,2])
    plt.show()

def plt_grad_field_slice(pose,theta,qsdf,workspace,step=0.01,axis='z',slice_value=0.0):
    x = torch.arange(workspace[0,0],workspace[1,0],step).to(qsdf.device)
    y = torch.arange(workspace[0,1],workspace[1,1],step).to(qsdf.device)
    z = torch.arange(workspace[0,2],workspace[1,2],step).to(qsdf.device)
    if axis=='x':
        slice_idx = (x-slice_value).abs().argmin().item()
        x_slice = x[slice_idx].unsqueeze(0)
        grid_y, grid_z= torch.meshgrid(y,z)
        grid_y, grid_z = grid_y.reshape(-1,1), grid_z.reshape(-1,1)
        p = torch.cat([x_slice.expand(len(grid_y),-1), grid_y, grid_z],dim=1).float().to(qsdf.device)   
    elif axis=='y':
        slice_idx = (y-slice_value).abs().argmin().item()
        y_slice = y[slice_idx].unsqueeze(0)
        grid_x, grid_z= torch.meshgrid(x,z)
        grid_x, grid_z = grid_x.reshape(-1,1), grid_z.reshape(-1,1)
        p = torch.cat([grid_x, y_slice.expand(len(grid_x),-1), grid_z],dim=1).float().to(qsdf.device)   
    else:
        slice_idx = (z-slice_value).abs().argmin().item()
        z_slice = z[slice_idx].unsqueeze(0)
        grid_x, grid_y= torch.meshgrid(x,y)
        grid_x, grid_y = grid_x.reshape(-1,1), grid_y.reshape(-1,1)
        p = torch.cat([grid_x, grid_y, z_slice.expand(len(grid_x),-1)],dim=1).float().to(qsdf.device)   
    print('p shape',p.shape)
    sdf, grad = qsdf.get_sdf_with_points_grad(p,pose,theta,use_derivative=True)
    sdf = sdf.squeeze(1)
    print('sdf shape',sdf.shape)
    print('grad shape',grad.shape)
    grad = grad.cpu().numpy()
    p = p.unsqueeze(0).expand(len(sdf),-1,-1).cpu().numpy()
    # 只画出距离小于0.1的点的梯度
    mask = (sdf.abs()<0.1).cpu().numpy()
    print('mask shape',mask.shape)
    print('p shape',p.shape)
    p = p[mask]
    grad = grad[mask]
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import Axes3D, art3d
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(p[:,0],p[:,1],p[:,2],grad[:,0],grad[:,1],grad[:,2],length=0.01)
    # 画出workspace的边界
    ax.set_xlim(workspace[0,0], workspace[1,0])
    ax.set_ylim(workspace[0,1], workspace[1,1])
    ax.set_zlim(workspace[0,2], workspace[1,2])
    plt.show()

def plt_joint_slice(joint_idx1,joint_idx2, pose, points, qsdf, workspace, slice_value=0.0):
    step = 100
    theta1 = torch.linspace(qsdf.robot.theta_min_soft[joint_idx1],qsdf.robot.theta_max_soft[joint_idx1],step).to(qsdf.device)
    theta2 = torch.linspace(qsdf.robot.theta_min_soft[joint_idx2],qsdf.robot.theta_max_soft[joint_idx2],step).to(qsdf.device)
    theta = torch.zeros(step,step,qsdf.robot.dof).to(qsdf.device)
    theta[:,:,joint_idx1] = theta1.unsqueeze(1).expand(-1,step)
    theta[:,:,joint_idx2] = theta2.unsqueeze(0).expand(step,-1)
    theta = theta.reshape(-1,qsdf.robot.dof)
    pose = pose.unsqueeze(0).expand(step*step,4,4).to(qsdf.device)
    sdf,grad = qsdf.get_sdf_with_joints_grad(points,pose,theta,use_derivative=True)
    sdf = sdf.squeeze(1).reshape(step,step).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots()
    contour = ax.contourf(theta1.cpu().numpy(), theta2.cpu().numpy(), sdf, levels=100, cmap='viridis')
    ax.set_xlabel(f'Joint {joint_idx1} Angle (rad)')
    ax.set_ylabel(f'Joint {joint_idx2} Angle (rad)')
    ax.set_title('SDF Value Contour Plot')
    # Add a color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(contour, cax=cax, orientation='vertical', label='SDF Value')
    # Add a contour line at slice_value
    contour_line = ax.contour(theta1.cpu().numpy(), theta2.cpu().numpy(), sdf, levels=[slice_value], colors='red', linewidths=2)
    
    grad = grad.reshape(step,step,points.shape[0],qsdf.robot.dof).mean(dim=2).detach().cpu().numpy() # (step,step,3)
    # project the gradient onto the 2D plane
    if joint_idx1<joint_idx2:
        grad_2d = grad[:,:, [joint_idx1, joint_idx2]]
    else:
        grad_2d = grad[:,:, [joint_idx2, joint_idx1]]
    # sample every 5 points to avoid clutter
    grad_2d = grad_2d[::5,::5,:]
    theta1 = theta1[::5]
    theta2 = theta2[::5]
    # plot the gradient field
    ax.quiver(theta1.cpu().numpy(), theta2.cpu().numpy(), grad_2d[:,:,0], grad_2d[:,:,1], color='white')
    # Label the contour line
    
    plt.clabel(contour_line, fmt={slice_value: f'SDF={slice_value}'}, inline=True, fontsize=10)
    plt.show()
if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--robot', default='leaphand', type=str, choices=['panda','dexhand', 'leaphand'], help='choose the robot model to train or evaluate')
    args = parser.parse_args()
    
    # --- initialize the paths ------
    paths = {
        'urdf': os.path.join(CUR_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'data/{args.robot}/sdf_points/'),
        'model':'/workspace/RDF/siren_model.pth'
        }
    # ---- initialize the robot model -----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    robot = ParallelRobotLayer(device=device, robot=args.robot, paths=paths)
    serial_robot = robot.serials[0] # thumb
    
    used_links = serial_robot.all_links.copy()
    if 'palm_lower_left' in used_links:
        used_links.remove('palm_lower_left')
    
    siren_sdf = QSDF(robot=serial_robot,paths=paths,device=device,used_links=used_links)

    # ---- train or evaluate the SDF model -----
    if args.train:
        siren_sdf.train_siren_sdf()
        
    # ---- evaluate the SDF model -----
    if args.eval:
        
        Bt = 1024
        Bx = 128
        # randomly choose joint angles theta
        theta = torch.rand(Bt,serial_robot.dof).float().to(device)*(serial_robot.theta_max_soft-serial_robot.theta_min_soft)+serial_robot.theta_min_soft
        # theta = torch.zeros(1,serial_robot.dof).float().to(device).expand(Bt,serial_robot.dof)
        # print('theta shape:',theta.shape)
        pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),4,4).float()
        
        # # run RDF 
        x = (torch.rand(Bx,3)*(serial_robot.space_limits[1]-serial_robot.space_limits[0])+serial_robot.space_limits[0]).to(device)
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(Bt,4,4).float()
        sdf, grad,idx = siren_sdf.get_sdf_with_points_grad(x,pose,theta,use_derivative=True,return_index=True)
        print('sdf:',sdf)
        print('grad:',grad)
        print(idx)
        print('grad norm:',torch.linalg.norm(grad,dim=-1))
        
        
        # --- debug: visualize the points and the robot ---
        # theta = torch.zeros(1,serial_robot.dof).float().to(device)
        # theta[0,1] = -0.5
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).float()
        # plt_sdf_shell(0.001, pose, theta, siren_sdf, serial_robot.space_limits)
        # plt_grad_field(pose, theta, siren_sdf, serial_robot.space_limits, step=0.05)
        # plt_grad_field_slice(pose, theta, siren_sdf, serial_robot.space_limits, step=0.01, axis='z', slice_value=0.0)

        joint_idx1 = 1
        joint_idx2 = 3
        points = torch.tensor([[0.0,-0.10,0.0]]).to(device)
        pose = pose.squeeze(0)
        plt_joint_slice(joint_idx1, joint_idx2, pose, points, siren_sdf, serial_robot.space_limits, slice_value=0.0)


