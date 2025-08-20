# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import glob
import trimesh
import utils
import mesh_to_sdf
import skimage
from panda_layer.panda_layer import PandaLayer
from panda_layer.robot_layer import RobotLayer
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class BPSDF():
    def __init__(self, n_func,domain_min,domain_max,robot,paths,device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device    
        self.robot = robot
        self.paths = paths
        
    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self,t, use_derivative=False):
        # t is normalized to [0,1]
        t =torch.clamp(t, min=1e-4, max=1-1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(),None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(),dphi.float()

    def build_basis_function_from_points(self,p,use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
        phi,d_phi = self.build_bernstein_t(p,use_derivative) 
        phi = phi.reshape(N,3,self.n_func)
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        phi_xy = torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk",phi_xy,phi_z).view(-1,self.n_func**3)
        if use_derivative ==False:
            return phi_xyz,None
        else:
            d_phi = d_phi.reshape(N,3,self.n_func)
            d_phi_x_1D= d_phi[:,0,:]
            d_phi_y_1D = d_phi[:,1,:]
            d_phi_z_1D = d_phi[:,2,:]
            d_phi_x = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D,phi_y).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_y = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_z = torch.einsum("ij,ik->ijk",phi_xy,d_phi_z_1D).view(-1,self.n_func**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)
            return phi_xyz,d_phi_xyz

    def train_bf_sdf(self,epoches=200,mesh_path=None,point_path=None):
        # represent SDF using basis functions
        mesh_path = self.paths['meshes']
        mesh_files = glob.glob(mesh_path)
        mesh_files = sorted(mesh_files)[:]
        mesh_dict = {}
        for i,mf in enumerate(mesh_files):
            mesh_name = mf.split('/')[-1].split('.')[0]
            if '_vis' in mesh_name:
                mesh_name = mesh_name.replace('_vis','')
            mesh = trimesh.load(mf)
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices-offset, axis=1))
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
            mesh_dict[mesh_name] = {}
            mesh_dict[mesh_name]['mesh_name'] = mesh_name
            # load data
            point_path = os.path.join(self.paths['points'],f'voxel_128_{mesh_name}.npy')
            
            data = np.load(point_path,allow_pickle=True).item()#TODO
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data <-1] = -sdf_random_data[sdf_random_data <-1]
            wb = torch.zeros(self.n_func**3).float().to(self.device)
            B = (torch.eye(self.n_func**3)/1e-4).float().to(self.device)
            # loss_list = []
            for iter in range(epoches):
                choice_near = np.random.choice(len(point_near_data),1024,replace=False)
                p_near,sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(self.device),torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)
                choice_random = np.random.choice(len(point_random_data),256,replace=False)
                p_random,sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(self.device),torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)
                p = torch.cat([p_near,p_random],dim=0)
                sdf = torch.cat([sdf_near,sdf_random],dim=0)
                phi_xyz, _ = self.build_basis_function_from_points(p.float().to(self.device),use_derivative=False)

                K = torch.matmul(B,phi_xyz.T).matmul(torch.linalg.inv((torch.eye(len(p)).float().to(self.device)+torch.matmul(torch.matmul(phi_xyz,B),phi_xyz.T))))
                B -= torch.matmul(K,phi_xyz).matmul(B)
                delta_wb = torch.matmul(K,(sdf - torch.matmul(phi_xyz,wb)).squeeze())
                # loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()
                # loss_list.append(loss)
                wb += delta_wb

            # print(f'mesh name {mesh_name} finished!')
            mesh_dict[mesh_name] ={
                'mesh_name':     mesh_name,
                'weights':  wb,
                'offset':   torch.from_numpy(offset).to(self.device).float(),
                'scale':      scale,  

            }
        if os.path.exists(self.paths['model']) is False:
            os.mkdir(self.paths['model'])
        torch.save(mesh_dict,f"{self.paths['model']}") # save the robot sdf model
        print(f"{self.paths['model']} model saved!")

    def sdf_to_mesh(self, model, nbData,use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            # print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min,self.domain_max,nbData).to(self.device)
            grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
            p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)   

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d =[]
            for p_s in p_split:
                phi_p,d_phi_p = self.build_basis_function_from_points(p_s,use_derivative)
                d_s = torch.matmul(phi_p,weights)
                d.append(d_s)
            d = torch.cat(d,dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData,nbData,nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(self.domain_max-self.domain_min)/nbData] * 3)
            )
            verts = verts - [1,1,1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list,mesh_name_list

    def create_surface_mesh(self,model, nbData,vis =False, save_mesh_name=None):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces,mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            if vis:
                rec_mesh.show()
            if save_mesh_name != None:
                save_path = os.path.join(CUR_DIR,"output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path,f"{save_mesh_name}_{mesh_name}.stl"))

    def get_whole_body_sdf_batch(self,x,pose,theta,model,use_derivative = True, used_links = None,return_index=False):
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
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()
        scale = torch.tensor([model[self.robot.Link2Mesh[link]]['scale'] for link in used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()
        trans = self.robot.get_link_transformations(pose, theta)
        # print(f'trans shape:{trans.shape}')
        # print(f'x shape:{x.shape}')
        # 在trans (links,B,4*4)中只保留used_links的transformation(变为（K，4，4）)
        # mask = torch.tensor([],dtype=torch.bool)
        # for ee_link in self.robot.ee_links:
        #     for name in self.robot.chain[ee_link].get_link_names():
        #         if name not in used_links:
        #             mask.append(False)
        #         else:
        #             mask.append(True)
        # # x: (N,3)  query points in world frame
        # # trans: (B,K,4,4)  -> (B*K,4,4)
        # trans=torch.gather()
        used_indices = [self.robot.all_links.index(link) for link in used_links if link in self.robot.all_links]
        # print(f'used_indices: {used_indices}')
        trans = trans[used_indices]  # (K, B, 4, 4)
        # print(f'trans shape after gather:{trans.shape}')
        trans = trans.reshape(-1,4,4).float()
        # # --- check if the transformation is correct ---
        # print(f'pose shape: {pose.shape}, theta shape: {theta.shape}')
        # print(f'trans shape: {trans.shape}')
        # if torch.isnan(trans).any() or torch.isinf(trans).any():
        #     print('Warning: trans contains NaN or Inf!')
        #     trans = torch.where(torch.isnan(trans), torch.zeros_like(trans), trans)
        #     trans = torch.where(torch.isinf(trans), torch.zeros_like(trans), trans)
        # dets = torch.det(trans)
        # print('Min det:', dets.min().item(), 'Max det:', dets.max().item())
        # print('Any det==0:', (dets==0).any().item())
        # 可选：只对可逆的矩阵做逆运算
        x_robot_frame_batch = utils.transform_points(x.float(),torch.linalg.inv(trans).float(),device=self.device) # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled/scale.unsqueeze(-1).unsqueeze(-1) #B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled>1.0-1e-2,1.0-1e-2,x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded<-1.0+1e-2,-1.0+1e-2,x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            weights_near = torch.cat([model[self.robot.Link2Mesh[link]]['weights'].unsqueeze(0) for link in used_links],dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij',phi,weights_near).reshape(K,B,N).transpose(0,1).reshape(B*K,N) # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*scale.reshape(B,K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            print(f'sdf_min: {sdf_value.min()}, sdf_max: {sdf_value.max()},sdf_mean: {sdf_value.mean()}')
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
            print(f'sdf_min: {sdf_value.min()}, sdf_max: {sdf_value.max()},sdf_mean: {sdf_value.mean()}')
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient,dim=-1)
            gradient = torch.nn.functional.normalize(gradient,dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = trans[:,:3,:3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl',fk_rotation,gradient.transpose(1,2)).transpose(1,2).reshape(B,K,N,3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B,K,N,3)
            gradient_value = torch.gather(gradient_base_frame,1,idx_grad)[:,0,:,:]
            # gradient_value = None
            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self,x,pose,theta,model,used_links = None):
        if used_links is None:
            used_links = self.robot.all_links
        used_links = [link for link in used_links if self.robot.Link2Mesh[link]is not None]
        delta = 0.001
        B = theta.shape[0]
        DoF = theta.shape[1]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,DoF,DoF)+ torch.eye(DoF,device=self.device).unsqueeze(0).expand(B,DoF,DoF)*delta).reshape(B,-1,DoF)
        theta = torch.cat([theta,d_theta],dim=1).reshape(B*(DoF+1),DoF)
        pose = pose.unsqueeze(1).expand(B,(DoF+1),4,4).reshape(B*(DoF+1),4,4)
        sdf,_ = self.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative = False, used_links = used_links)
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

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--robot', default='panda', type=str, choices=['panda','dexhand', 'leaphand'], help='choose the robot model to train or evaluate')
    args = parser.parse_args()
    
    # --- initialize the paths ------
    paths = {
        'urdf': os.path.join(CUR_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'data/{args.robot}/sdf_points/'),
        'model':os.path.join(CUR_DIR, f'models/{args.robot}/BP_{args.n_func}.pt')
        }
    # ---- initialize the paths depending on the robot ----
    robot = RobotLayer(device=args.device, robot=args.robot, paths=paths)
    bp_sdf = BPSDF(args.n_func,args.domain_min,args.domain_max,robot=robot,paths=paths,device=args.device)
    #  train Bernstein Polynomial model   
    if args.train:
        bp_sdf.train_bf_sdf(mesh_path=paths['meshes'], epoches=200)
    if args.eval:
        # load trained model
        model = torch.load(paths['model'])
        # print('model loaded!',model.keys())
        # visualize the Bernstein Polynomial model for each robot link
        bp_sdf.create_surface_mesh(model,nbData=128,vis=False,save_mesh_name=f'BP_{args.n_func}')

        # visualize the Bernstein Polynomial model for the whole body
        B=11
        theta = torch.cat([torch.zeros(1,robot.DoFs[ee_link]).float() for ee_link in robot.ee_links],dim=-1).to(args.device)
        theta = theta.expand(B,-1)
        # print('theta shape:',theta.shape)
        
        # theta = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(args.device).reshape(-1,7)
        pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(theta),4,4).float()
        # trans_list = robot.get_transformations_each_link(pose,theta)
        # print(trans_list)
        
        # print('------------------------------------------------------------')
        trans_list = robot.get_link_transformations(pose, theta)
        # print(trans_list)
        
        # run RDF 
        x = torch.rand(128,3).to(args.device)*2.0 - 1.0
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(B,4,4).float()
        used_link = robot.all_links
        # print('used link:',used_link)
        sdf,gradient = bp_sdf.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative=True,used_links = used_link)
        # print('sdf:',sdf,'gradient:',gradient)
        sdf,joint_grad = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(x,pose,theta,model,used_links= used_link)
        # print('sdf:',sdf,'joint gradient:',joint_grad)





