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
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class SirenSDF():
    def __init__(self,robot,paths,device,domain_min=-1.0,domain_max=1.0):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device    
        self.robot = robot
        self.paths = paths
        
    def train_siren_sdf(self, epochs=500):
        mesh_dict = {}
        total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.

        
        for mesh_name in self.robot.meshes.keys():
            if mesh_name is None:
                continue
            sdf_siren = Siren(in_features=3, out_features=1, hidden_features=256, 
                    hidden_layers=3, outermost_linear=True)
            sdf_siren.cuda()
            optim = torch.optim.Adam(lr=1e-4, params=sdf_siren.parameters())
            
            mesh = trimesh.Trimesh(self.robot.meshes[mesh_name][0][:,:3].cpu().detach().numpy(),
                                   self.robot.meshes[mesh_name][1])
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices-offset, axis=1))
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
            mesh_dict[mesh_name] = {}
            mesh_dict[mesh_name]['mesh_name'] = mesh_name
            # load data
            point_path = self.paths['points'] + f'voxel_128_{mesh_name}.npy'
            if not os.path.exists(point_path):
                print(f"Data for {mesh_name} does not exist, skipping...")
                continue
            data = np.load(point_path,allow_pickle=True).item()#TODO
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            # --debug check data ---
            # mask_near = (sdf_near_data < 0)
            # plot_pts_near = point_near_data[mask_near]
            # mask_random = (sdf_random_data < 0)
            # plot_pts_random = point_random_data[mask_random]
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.scatter(plot_pts_near[:,0],plot_pts_near[:,1],plot_pts_near[:,2],c='r',s=1)
            # ax.scatter(plot_pts_random[:,0],plot_pts_random[:,1],plot_pts_random[:,2],c='b',s=1)
            # plt.show()
            # --debug check data end ---
            
            for iter in range(epochs):
                choice_near = np.random.choice(len(point_near_data),1024,replace=False)
                p_near,sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to('cuda'),torch.from_numpy(sdf_near_data[choice_near]).float().to('cuda')
                choice_random = np.random.choice(len(point_random_data),256,replace=False)
                p_random,sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to('cuda'),torch.from_numpy(sdf_random_data[choice_random]).float().to('cuda')
                model_input = torch.cat([p_near,p_random],dim=0)
                ground_truth = torch.cat([sdf_near,sdf_random],dim=0).unsqueeze(-1)
                model_output, coords = sdf_siren(model_input)    
                loss = ((model_output - ground_truth)**2).mean()
                
                if not iter % 10:
                    print("Step %d, Total loss %0.6f" % (iter, loss))
                optim.zero_grad()
                loss.backward()
                optim.step()
            mesh_dict[mesh_name] ={
                'mesh_name':     mesh_name,
                'weights':   sdf_siren.state_dict(),
                'offset':   torch.from_numpy(offset).to(self.device).float(),
                'scale':      scale,  

            }
        folder = os.path.dirname(self.paths['model'])
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        torch.save(mesh_dict,f"{self.paths['model']}") # save the robot sdf model
        print(f"{self.paths['model']} model saved!")

    def sdf_to_mesh(self, model, nbData,use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        model_dict = torch.load(self.paths['model'],map_location=self.device)
        for i, k in enumerate(model_dict.keys()):
            if 'weights' not in model_dict[k]:
                print(f"Skipping {k} as it does not contain weights.")
                continue
            mesh_dict = model_dict[k]
            mesh_name = mesh_dict['mesh_name']
            mesh_name_list.append(mesh_name)
            print(f'processing {mesh_name} mesh')
            model.load_state_dict(mesh_dict['weights'])
            domain = torch.linspace(self.domain_min,self.domain_max,nbData).to(self.device)
            grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
            p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)   

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d =[]
            for p_s in p_split:
                d_s,_ = model(p_s)
                d.append(d_s)
            d = torch.cat(d,dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData,nbData,nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(self.domain_max-self.domain_min)/nbData] * 3)
            )
            verts = verts - [1,1,1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list,mesh_name_list

    def create_surface_mesh(self,model, nbData,vis =False):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces,mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            if vis:
                print(f'visualizing {mesh_name} mesh')
                rec_mesh.show()
   
    def visualize_sdf_slice(self, model, mesh_name, z_value=0.0, nbData=100, domain_range=(-1, 1),xyz='z'):
        """
        可视化SDF场在指定Z值的切片和等高线图
        
        Parameters:
        -----------
        model : dict
            包含所有网格SDF模型的字典
        mesh_name : str
            要可视化的特定网格名称
        z_value : float
            要切片的Z坐标值
        nbData : int
            在每个维度上的采样点数
        domain_range : tuple
            采样域的范围 (min, max)
        """
        
        # 创建XY网格
        domain_min, domain_max = domain_range
        x = np.linspace(domain_min, domain_max, nbData)
        y = np.linspace(domain_min, domain_max, nbData)
        
        X, Y = np.meshgrid(x, y)
       
        # 创建固定Z值的点阵
        points = []
        for i in range(nbData):
            for j in range(nbData):
                if xyz == 'z':
                    points.append([X[i, j], Y[i, j], z_value])
                elif xyz == 'y':
                    points.append([X[i, j], z_value, Y[i, j]])
                elif xyz == 'x':
                    points.append([z_value, X[i, j], Y[i, j]])
        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # 分块计算SDF值
        sdf_values = []
        batch_size = 10000
        for i in range(0, len(points), batch_size):
            batch_sdf, _ = model(points[i:i+batch_size])
            sdf_values.append(batch_sdf.detach().cpu().numpy())
        
        sdf_values = np.concatenate(sdf_values)
        sdf_grid = sdf_values.reshape(nbData, nbData)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # sdf_grid = sdf_grid.detach().cpu().numpy()
        # 子图1：SDF值的热力图
        im1 = ax1.imshow(sdf_grid,
                        extent=[domain_min, domain_max, domain_min, domain_max],
                        origin='lower', 
                        cmap='RdBu_r',  # 红蓝配色，红色为正（外部），蓝色为负（内部）
                        vmin=-1.0, vmax=1.0)  # 限制颜色范围以便更好地观察零值附近
        
        ax1.set_title(f'SDF Values at {xyz} = {z_value:.2f}\nMesh: {mesh_name}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # 添加颜色条
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1, label='SDF Value')
        
        # 添加零等值线
        contour = ax1.contour(X, Y, sdf_grid, levels=[0], colors='black', linewidths=2)
        ax1.clabel(contour, inline=True, fontsize=10, fmt='%1.2f')
        
        # 子图2：等高线图，重点关注零值附近
        levels = np.linspace(-0.5, 0.5, 21)  # 在零值附近密集采样
        im2 = ax2.contourf(X, Y, sdf_grid, levels=levels, cmap='RdBu_r', extend='both')
        zero_contour = ax2.contour(X, Y, sdf_grid, levels=[0], colors='black', linewidths=3)
        
        # 标记所有零等值线
        for i, path in enumerate(zero_contour.collections[0].get_paths()):
            if len(path.vertices) > 0:
                # 找到路径的大致中心点来放置标签
                center = path.vertices.mean(axis=0)
                ax2.text(center[0], center[1], f'Zero {i+1}', 
                        fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax2.set_title(f'Zero Level Contours (SDF = 0)\nNumber of zero contours: {len(zero_contour.collections[0].get_paths())}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # 添加颜色条
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2, label='SDF Value')
        
        plt.tight_layout()
        # plt.show()
        
        # 保存图像
        save_path = os.path.join(CUR_DIR, "siren_sdf_check_img")
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        fig.savefig(os.path.join(save_path, f"SDF_Slice_{mesh_name}_{xyz}_{z_value:.2f}.png"))
        
        # 打印诊断信息
        print(f"=== SDF Slice Analysis at Z = {z_value:.2f} ===")
        print(f"Mesh: {mesh_name}")
        print(f"SDF value range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
        print(f"Number of distinct zero contours: {len(zero_contour.collections[0].get_paths())}")
        
        return sdf_grid, zero_contour

    def find_ghost_surface_z_values(self, model, mesh_name, nbData=50, domain_range=(-1, 1), num_slices=10):
        """
        在多个Z切片中搜索可能的幽灵面
        
        Parameters:
        -----------
        model : dict
            SDF模型字典
        mesh_name : str
            网格名称
        nbData : int
            采样点数
        domain_range : tuple
            域范围
        num_slices : int
            要检查的切片数量
        """
        
        z_values = np.linspace(domain_range[0], domain_range[1], num_slices)
        ghost_detected = False
        
        model.load_state_dict(torch.load(paths['model'],map_location=args.device)[mesh_name]['weights'])
        model.to(args.device)
        model.eval()
        
        for z in z_values:
            print(f"\n--- Checking Z = {z:.2f} ---")
            sdf_grid, contour = self.visualize_sdf_slice(model, mesh_name, z, nbData, domain_range,xyz='x')

    def check_eikonal_equation(self, model, mesh_path=None, nbData=1000):
        """
        检查SDF梯度是否满足Eikonal方程 |∇φ| = 1
        
        Parameters:
        -----------
        model : dict
            SDF模型字典
        mesh_path : str
            网格路径（用于诊断）
        nbData : int
            采样点数
        """
        for mesh_name in model.keys():
            print(f"\n--- Checking Eikonal Equation for Mesh: {mesh_name} ---")
            mesh_dict = model[mesh_name]
            weights = mesh_dict['weights'].to(self.device)
            device = self.device
            # 在立方体域内均匀采样点
            points = torch.rand((nbData, 3), device=device) * (self.domain_max - self.domain_min) + self.domain_min
            
            # 分块计算SDF值和梯度
            sdf_values = []
            gradients = []
            batch_size = 10000
            for i in range(0, len(points), batch_size):
                batch_sdf, coords = model(points[i:i+batch_size])
                batch_grad = gradient(batch_sdf, coords)
                print('shape of batch_sdf and batch_grad:',batch_sdf.shape,batch_grad.shape)
                
                sdf_values.append(batch_sdf)
                gradients.append(batch_grad)
            
            sdf_values = torch.cat(sdf_values, dim=0)
            gradients = torch.cat(gradients, dim=0)
            
            # 计算梯度的范数
            grad_norms = torch.norm(gradients, dim=-1)
            
            # 计算与1的偏差
            deviations = torch.abs(grad_norms - 1.0)
            
            # 统计偏差信息
            mean_dev = deviations.mean().item()
            max_dev = deviations.max().item()
            std_dev = deviations.std().item()
            
            print(f"SDF Value Range: [{sdf_values.min().item():.4f}, {sdf_values.max().item():.4f}]")
            print(f"Gradient Norm Deviation from 1: Mean={mean_dev:.4f}, Max={max_dev:.4f}, Std={std_dev:.4f}")
            
            # 两张子图，一张显示梯度范数直方图,并标记理想值1，另一张显示偏差直方图.
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(grad_norms.detach().cpu().numpy(), bins=50, color='blue', alpha=0.7)
            # 标记理想值1
            ax1.axvline(1.0, color='red', linestyle='--', label='Ideal |∇φ|=1')
            ax1.set_title('Histogram of Gradient Norms |∇φ|')
            ax1.set_xlabel('|∇φ|')
            ax1.set_ylabel('Frequency')
            ax1.axvline(1.0, color='red', linestyle='--', label='Ideal |∇φ|=1')
            ax1.legend()
            ax2.hist(deviations.detach().cpu().numpy(), bins=50, color='green', alpha=0.7)
            ax2.set_title('Histogram of Deviations |∇φ| - 1')
            ax2.set_xlabel('Deviation')
            ax2.set_ylabel('Frequency')
            # 添加文字，显示sdf范围和偏差统计信息
            textstr = '\n'.join((
                f'SDF Range: [{sdf_values.min().item():.4f}, {sdf_values.max().item():.4f}]',
                f'Mean Deviation: {mean_dev:.4f}',
                f'Max Deviation: {max_dev:.4f}',
                f'Std Deviation: {std_dev:.4f}'
            ))
            # 放置文字框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
            
            plt.tight_layout()
            plt.show()
            # 保存图像
            save_path = os.path.join(CUR_DIR, "siren_eikonal_check_img")
            if os.path.exists(save_path) is False:
                os.mkdir(save_path)
            fig.savefig(os.path.join(save_path, f"Eikonal_Check_{mesh_name}.png"))
            print(f"Diagnostic plots saved to {save_path}")
            
            
    def get_whole_body_sdf_batch(self,x,pose,theta,model,use_derivative = True, used_links = None,return_index=False,serial = False):
        # x: (Nx,3)  query points in world frame
        # pose: (B,4,4)  base pose in world frame
        # theta: (B,DoF)  joint angles
        # model: dict of mesh model
        # used_links: list of link index to use,！！！这里实际上并没有用到，因为除了panda，两个手都是用的全部link！！！
        if used_links is None:
            used_links = self.robot.all_links
        used_links = [link for link in used_links if self.robot.Link2Mesh[link] is not None]
        B = len(theta)
        N = len(x)
        K = len(used_links)
        
        offset = torch.cat([model[self.robot.Link2Mesh[link]]['offset'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()# offset: (B*K,3)
        
        scale = torch.tensor([model[self.robot.Link2Mesh[link]]['scale'] for link in used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()# scale: (B*K)
        trans = self.robot.get_link_mesh_transformations(pose, theta)#trans: (K+1, B, 4, 4)
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
        # mesh = self.robot.get_forward_robot_mesh(pose[0].unsqueeze(0), theta_debug,used_links=used_links)[0]
        # mesh = self.robot.get_forward_robot_mesh(pose[0].unsqueeze(0), theta_debug,used_links=used_links)[0]
        # scene.add_geometry(mesh)
        # scene.show()
        # # --- debug ---
        
        if not use_derivative:
            phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij',phi,weights_near).reshape(K,B,N).transpose(0,1).reshape(B*K,N) # B*K,N
            # np.set_printoptions(threshold=np.inf)
            # print(sdf.reshape(B,K,N).transpose(0,1)[0].cpu().detach().numpy())
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*scale.reshape(B,K).unsqueeze(-1)#sdf  (B,K,1)
            # --- unpack by serial ---
            if serial:
                sdfs = []
                for serial in self.robot.serials:
                    serial_links = serial.all_links
                    serial_links = [link for link in serial_links if self.robot.Link2Mesh[link] is not None]
                    serial_indices = [used_links.index(link) for link in serial_links if link in used_links]
                    serial_sdf = sdf[:,serial_indices,:]
                    serial_sdf, _ = torch.min(serial_sdf,dim=1)
                    sdfs.append(serial_sdf)
                    print(f'serial {serial.all_links[-1]} sdf:',serial_sdf)
                return sdfs
            # --- unpack by serial end ---
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:   
            phi,dphi = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1),dphi],dim=-1)
            phi_cat = phi_cat.reshape(B,K,N,-1,4).transpose(0,1).reshape(K,B*N,-1,4) # K,B*N,-1,4

            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)

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
    def get_serial_sdf_batch(self,x,pose,theta,model,serial_idx,use_derivative = True, used_links = None,return_index=False):
        # x: (Nx,3)  query points in world frame
        # pose: (B,4,4)  base pose in world frame
        # theta: (B,DoF)  joint angles of the serial
        # model: dict of mesh model
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
        
        offset = torch.cat([model[serial.Link2Mesh[link]]['offset'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()# offset: (B*K,3)
        
        scale = torch.tensor([model[self.robot.Link2Mesh[link]]['scale'] for link in used_links],device=self.device)
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
                batch_sdf, coords = model(x_bounded[i:i+batch_size])
                sdf_values.append(batch_sdf)
            sdf = torch.cat(sdf_values, dim=0)
            print('sdf shape:',sdf.shape)
            exit()
            # phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            # phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            # weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
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

            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)

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
    def get_whole_body_sdf_with_joints_grad_batch(self,x,pose,theta,model,used_links = None, serial = False):
        # theta: (B,DoF)
        # pose: (B,4,4)
        if used_links is None:
            used_links = self.robot.all_links
        used_links = [link for link in used_links if self.robot.Link2Mesh[link]is not None]
        delta = 0.001
        B = theta.shape[0]
        DoF = theta.shape[1]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,DoF,DoF)+ torch.eye(DoF,device=self.device).unsqueeze(0).expand(B,DoF,DoF)*delta).reshape(B,-1,DoF)
        theta = torch.cat([theta,d_theta],dim=1) # (B,(DoF+1),DoF)
        pose = pose.unsqueeze(1).expand(B,(DoF+1),4,4).reshape(B*(DoF+1),4,4)
        if serial:
            sdfs = self.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative = False, used_links = used_links,serial = True)
            for i,serial in enumerate(self.robot.serials):
                serial_links = serial.all_links
                serial_links = [link for link in serial_links if self.robot.Link2Mesh[link] is not None]
                serial_indices = [used_links.index(link) for link in serial_links if link in used_links]
                serial_theta = torch.stack([theta[:,self.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1)
                serial_sdf = sdfs[i]
                serial_sdf, d_serial_sdf = serial_sdf[:,0,:],serial_sdf[:,1:,:]-serial_sdf[:,:1,:]
                d_serial_sdf = d_serial_sdf/ delta
            return sdf
        else:
            theta = theta.reshape(B*(DoF+1),DoF)
            sdf,_ = self.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative = False, used_links = used_links)
            sdf = sdf.reshape(B,(DoF+1),-1)
            d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
            return sdf[:,0,:], d_sdf.transpose(1,2)
    def get_serial_sdf_with_joints_grad_batch(self,x,pose,theta,model,used_links = None, serial_idx = None):
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
        sdf,_ = self.get_serial_sdf_batch(x,pose,theta,model,use_derivative = False, used_links = used_links, serial_idx = serial_idx)
        sdf = sdf.reshape(B,(DoF+1),-1)
        d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
        return sdf[:,0,:], d_sdf.transpose(1,2)
    def get_whole_body_normal_with_joints_grad_batch(self,x,pose,theta,model,used_links = None):
        if used_links is None:
            used_links = self.robot.all_links
        used_links = [link for link in used_links if self.robot.Link2Mesh[link]is not None]
        normals = {}
        for t in theta.values():
            delta = 0.001
            B = t.shape[0]
            DoF = t.shape[1]
            t = t.unsqueeze(1)
            d_theta = (t.expand(B,DoF,DoF)+ torch.eye(DoF,device=self.device).unsqueeze(0).expand(B,DoF,DoF)*delta).reshape(B,-1,DoF)
            t = torch.cat([t,d_theta],dim=1).reshape(B*(DoF+1),DoF)
            pose = pose.unsqueeze(1).expand(B,(DoF+1),4,4).reshape(B*(DoF+1),4,4)
            sdf, normal = self.get_whole_body_sdf_batch(x,pose,t,model,use_derivative = True, used_links = used_links)
            normal = normal.reshape(B,(DoF+1),-1,3).transpose(1,2)
            normals[t] = normal
        return normals # normal size: (B,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints

    def get_chain_sdf_batch(self,x,pose,theta,model,chain_link,use_derivative = True, used_links = None,return_index=False):
        # x: (Nx,3)  query points in world frame
        # pose: (B,4,4)  base pose in world frame
        # theta: (B,DoF)  joint angles
        # model: dict of mesh model
        # used_links: list of link index to use,！！！这里实际上并没有用到，因为除了panda，两个手都是用的全部link！！！
        if used_links is None:
            used_links = self.robot.all_links
        used_links = [link for link in used_links if self.robot.Link2Mesh[link] is not None]
        B = len(theta)
        N = len(x)
        K = len(used_links)
        
        # print(f'Batch size: {B}, Number of links: {K}, Number of points: {N}')
        # print(f'Used links: {used_links}')
        # print(f'model keys: {model.keys()}')
        offset = torch.cat([model[self.robot.Link2Mesh[link]]['offset'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()# offset: (B*K,3)
        
        scale = torch.tensor([model[self.robot.Link2Mesh[link]]['scale'] for link in used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()# scale: (B*K)
        trans = self.robot.get_link_mesh_transformations(pose, theta)#trans: (K+1, B, 4, 4)
        used_indices = [self.robot.all_links.index(link) for link in used_links if link in self.robot.all_links]
        trans = trans[used_indices]  # trans: (K, B, 4, 4)
        trans = trans.transpose(1,0) # (B, K, 4, 4)
        trans = trans.reshape(-1,4,4).float() # trans: (B*K, 4, 4)
        x_robot_frame_batch = utils.transform_points(x.float(),torch.linalg.inv(trans).float(),device=self.device) # B*K,N,3
        # x_robot_frame_batch: (B*K,N,3); x: （N,3）
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled/scale.unsqueeze(-1).unsqueeze(-1) #B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled>1.0-1e-2,1.0-1e-2,x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded<-1.0+1e-2,-1.0+1e-2,x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded # res_x: B*K,N,3
        
        if not use_derivative:
            phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij',phi,weights_near).reshape(K,B,N).transpose(0,1).reshape(B*K,N) # B*K,N
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

            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)

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
if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--robot', default='panda', type=str, choices=['panda','dexhand', 'leaphand'], help='choose the robot model to train or evaluate')
    args = parser.parse_args()
    
    # --- initialize the paths ------
    paths = {
        'urdf': os.path.join(CUR_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'data/{args.robot}/sdf_points/'),
        'model':'/workspace/RDF/siren_model.pth'
        }
    # ---- initialize the paths depending on the robot ----
    robot = ParallelRobotLayer(device=args.device, robot=args.robot, paths=paths)
    siren_sdf = SirenSDF(args.domain_min,args.domain_max,robot=robot,paths=paths,device=args.device)
    #  train Bernstein Polynomial model 
    if args.train:
        siren_sdf.train_siren_sdf()
    if args.eval:
        # load trained model
        model = Siren(in_features=3, out_features=1, hidden_features=256, 
                  hidden_layers=3, outermost_linear=True)
        
        # evaluation: 检查梯度是否满足Eikonal方程
        # siren_sdf.check_eikonal_equation(model,mesh_path=paths['meshes'],nbData=1000)
        # --debug check ghost surface ---        
        # 选择有问题的网格名称
        for mesh_name in robot.meshes.keys():
            if mesh_name is None:
                continue
            # 方法1：在特定Z值切片查看
            # siren_sdf.visualize_sdf_slice(model, problematic_mesh_name, z_value=0.0, nbData=100)
            
            # 方法2：在多个Z值搜索幽灵面
            siren_sdf.find_ghost_surface_z_values(model, mesh_name, num_slices=20)
            
            # visualize the Bernstein Polynomial model for each robot link
        siren_sdf.create_surface_mesh(model,nbData=128,vis=True)

        exit()
        # visualize the Bernstein Polynomial model for the whole body
        B=3
        # randomly choose joint angles theta
        # theta = torch.rand(B,robot.dof).float().to(args.device)*(robot.theta_max_soft-robot.theta_min_soft)+robot.theta_min_soft
        theta = torch.zeros(1,robot.dof).float().to(args.device).expand(B,robot.dof)
        # print('theta shape:',theta.shape)
        
        # theta = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(args.device).reshape(-1,7)
        pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(theta),4,4).float()
        
        # run RDF 
        x = (torch.rand(128,3)*(robot.space_limits[1]-robot.space_limits[0])+robot.space_limits[0]).to(args.device)
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(B,4,4).float()
        used_link = robot.all_links
        for i,serial in enumerate(robot.serials):
            used_link = serial.all_links.copy()
            if 'palm_lower_left' in used_link:
                used_link.remove('palm_lower_left')
            serial_theta = torch.stack([theta[:,robot.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1)
            serial.get_link_transformations(pose, serial_theta)
            print('serial_theta:',serial_theta)
            sdf,gradient = siren_sdf.get_serial_sdf_batch(x,pose,serial_theta,model,serial_idx=i,use_derivative=True,used_links= used_link)
            print('sdf:',sdf,'gradient:',gradient)
            sdf,joint_grad = siren_sdf.get_serial_sdf_with_joints_grad_batch(x,pose,serial_theta,model,serial_idx=i,used_links= used_link)
            print(joint_grad.shape)
            print('sdf:',sdf,'joint gradient:',joint_grad[0].cpu().detach().numpy())
        used_link = robot.all_links.copy()
        if 'palm_lower_left' in used_link:
            used_link.remove('palm_lower_left')
        sdf,gradient = siren_sdf.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative=True,used_links = used_link)
        # print('sdf:',sdf,'gradient:',gradient)
        sdf,joint_grad = siren_sdf.get_whole_body_sdf_with_joints_grad_batch(x,pose,theta,model,used_links= used_link)
        print(joint_grad.shape)
        # print('sdf:',sdf,'joint gradient:',joint_grad[0].cpu().detach().numpy())
        # pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(1,4,4).float()
        # theta = torch.zeros(1,robot.dof).float().to(args.device)
        # theta_1 = torch.zeros(1,robot.dof).float().to(args.device)
        # theta_1[:,5] = 2.0
        # sdf ,grad = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(x,pose,theta,model,used_links = used_link)
        # sdf_1 ,grad_1 = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(x,pose,theta_1,model,used_links = used_link)
        # print('sdf:',(sdf-sdf_1).cpu().detach().numpy())
        # c=input()
        # print('joint gradient:',grad[0].cpu().detach().numpy())
        # print('joint gradient_1:',grad_1[0].cpu().detach().numpy())




