

# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import torch
import os
from panda_layer.panda_layer import PandaLayer
from panda_layer.robot_layer import RobotLayer
import bf_sdf
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import utils
import argparse

def plot_2D_panda_sdf(pose,theta,bp_sdf,nbData,model,device):
    domain_0 = torch.linspace(-1.0,1.0,nbData).to(device)
    domain_1 = torch.linspace(-1.0,1.0,nbData).to(device)
    grid_x, grid_y= torch.meshgrid(domain_0,domain_1)
    p1 = torch.stack([grid_x.reshape(-1),grid_y.reshape(-1),torch.zeros_like(grid_x.reshape(-1))],dim=1)
    p2 = torch.stack([torch.zeros_like(grid_x.reshape(-1)),grid_x.reshape(-1)*0.4, grid_y.reshape(-1)*0.4+0.375],dim=1)
    p3 = torch.stack([grid_x.reshape(-1)*0.4 + 0.2,torch.zeros_like(grid_x.reshape(-1)),grid_y.reshape(-1)*0.4+0.375],dim=1)
    grid_x, grid_y= grid_x.detach().cpu().numpy(), grid_y.detach().cpu().numpy()

    plt.figure(figsize=(10,10))
    plt.rc('font', size=25)
    p2_split = torch.split(p2,1000,dim=0)
    sdf,ana_grad = [],[]
    for p_2 in p2_split:
        sdf_split,ana_grad_split = bp_sdf.get_whole_body_sdf_batch(p_2,pose,theta,model,use_derivative=True)
        sdf_split,ana_grad_split = sdf_split.squeeze(),ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf,dim=0)
    ana_grad = torch.cat(ana_grad,dim=0)
    p2 = p2.detach().cpu().numpy()
    sdf =sdf.squeeze().reshape(nbData,nbData).detach().cpu().numpy()
    ct1 = plt.contour(grid_x*0.4,grid_y*0.4+0.375,sdf,levels=12)
    plt.clabel(ct1, inline=False, fontsize=10)
    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:,[1,2]],dim=-1)*0.01
    p2_3d = p2.reshape(nbData,nbData,3)
    ana_grad_3d = ana_grad_2d.reshape(nbData,nbData,2) 
    plt.quiver(p2_3d[0:-1:4,0:-1:4,1],p2_3d[0:-1:4,0:-1:4,2],ana_grad_3d[0:-1:4,0:-1:4,0].detach().cpu().numpy(),ana_grad_3d[0:-1:4,0:-1:4,1].detach().cpu().numpy(),scale=0.5,color = [0.1,0.1,0.1])
    plt.title('YoZ')
    plt.show()

    # plt.subplot(1,3,3)
    plt.figure(figsize=(10,10))
    plt.rc('font', size=25)
    p3_split = torch.split(p3,1000,dim=0)
    sdf,ana_grad = [],[]
    for p_3 in p3_split:
        sdf_split,ana_grad_split = bp_sdf.get_whole_body_sdf_batch(p_3,pose,theta,model,use_derivative=True)
        sdf_split,ana_grad_split = sdf_split.squeeze(),ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf,dim=0)
    ana_grad = torch.cat(ana_grad,dim=0)
    p3 = p3.detach().cpu().numpy()
    sdf =sdf.squeeze().reshape(nbData,nbData).detach().cpu().numpy()
    ct1 = plt.contour(grid_x*0.4+0.2,grid_y*0.4+0.375,sdf,levels=12)
    plt.clabel(ct1, inline=False, fontsize=10)
    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:,[0,2]],dim=-1)*0.01
    p3_3d = p3.reshape(nbData,nbData,3)
    ana_grad_3d = ana_grad_2d.reshape(nbData,nbData,2) 
    plt.quiver(p3_3d[0:-1:4,0:-1:4,0],p3_3d[0:-1:4,0:-1:4,2],ana_grad_3d[0:-1:4,0:-1:4,0].detach().cpu().numpy(),ana_grad_3d[0:-1:4,0:-1:4,1].detach().cpu().numpy(),scale=0.5,color = [0.1,0.1,0.1])
    plt.title('XoZ')
    plt.show()

def plot_3D_panda_with_gradient(pose,theta,bp_sdf,model,device):  
    robot_mesh = robot.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)
    surface_points = robot_mesh.vertices
    scene = trimesh.Scene() 

    # robot mesh
    scene.add_geometry(robot_mesh)
    scene.show()
    choice = np.random.choice(len(surface_points), 1024, replace=False)
    surface_points = surface_points[choice]
    p =torch.from_numpy(surface_points).float().to(device)
    ball_query = trimesh.creation.uv_sphere(1).vertices
    choice_ball = np.random.choice(len(ball_query), 1024, replace=False)
    ball_query = ball_query[choice_ball]
    p = p + torch.from_numpy(ball_query).float().to(device)*0.5
    sdf,ana_grad = bp_sdf.get_whole_body_sdf_batch(p,pose,theta,model,use_derivative=True)
    sdf,ana_grad = sdf.squeeze().detach().cpu().numpy(),ana_grad.squeeze().detach().cpu().numpy()
    # points
    pts = p.detach().cpu().numpy()
    colors = np.zeros_like(pts,dtype=object)
    colors[:,0] = np.abs(sdf)*400
    # pc =trimesh.PointCloud(pts,colors)
    # scene.add_geometry(pc)
    # for i in range(len(pts)):
    #     dg = ana_grad[i]
    #     if dg.sum() ==0:
    #         continue
    #     c = colors[i]
    #     print(c)
    #     m = utils.create_arrow(-dg,pts[i],vec_length = 0.05,color=c)
    #     scene.add_geometry(m)
    # scene.show()   
    space_limits = np.array([[-0.5,-0.5,0.0],[0.5,0.5,1.0]])
    N = 20
    points = np.random.rand(N,3) * (space_limits[1]-space_limits[0]) + space_limits[0]
    points = torch.from_numpy(points).float().to(device)
    sdf,ana_grad = bp_sdf.get_whole_body_sdf_batch(points,pose,theta,model,use_derivative=True)
    sdf,ana_grad = sdf.squeeze().detach().cpu().numpy(),ana_grad.squeeze().detach().cpu().numpy()
    # 在的空间中画一个0.1*0.1*0.1的立方体
    scene.add_geometry(trimesh.creation.box(extents=[100,100,100],transform=trimesh.transformations.translation_matrix([0,0,0.05]),color=[255,0,0,100]))
    print('here is the cube')
    for i in range(len(points)):
        dg = ana_grad[i]
        if dg.sum() ==0:
            continue
        c = [0,0,255]
        m = utils.create_arrow(-dg,points[i].detach().cpu().numpy(),vec_length = 0.05,color=c)
        scene.add_geometry(m)
        print(f'{i} point: {points[i]}, sdf: {sdf[i]}')
    scene.show()
        # c = input()
    # gradients
    

def generate_panda_mesh_sdf_points(max_dist =0.10):
    # represent SDF using basis functions
    import glob
    import mesh_to_sdf
    mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*"
    mesh_files = glob.glob(mesh_path)
    mesh_files = sorted(mesh_files)[1:] #except finger
    mesh_dict = {}

    for i,mf in enumerate(mesh_files):
        mesh_name = mf.split('/')[-1].split('.')[0]
        print(mesh_name)
        mesh = trimesh.load(mf)
        mesh_dict[i] = {}
        mesh_dict[i]['mesh_name'] = mesh_name

        vert = mesh.vertices
        points = vert + np.random.uniform(-max_dist,max_dist,size=vert.shape)
        sdf = random_sdf = mesh_to_sdf.mesh_to_sdf(mesh, 
                                     points, 
                                     surface_point_method='scan', 
                                     sign_method='normal', 
                                     bounding_radius=None, 
                                     scan_count=100, 
                                     scan_resolution=400, 
                                     sample_point_count=10000000, 
                                     normal_sample_count=100) 
        mesh_dict[i]['points'] = points
        mesh_dict[i]['sdf'] = sdf
    np.save('data/panda_mesh_sdf.npy',mesh_dict)

def vis_panda_sdf(pose, theta,device):
    data = np.load('data/panda_mesh_sdf.npy',allow_pickle=True).item()
    # trans = panda.get_transformations_each_link(pose,theta)
    trans = robot.get_link_transformations(pose,theta)
    pts = []
    for i,k in enumerate(data.keys()):
        points = data[k]['points']
        sdf = data[k]['sdf']
        print(points.shape, sdf.shape)
        choice = (sdf <0.05) * (sdf>0.045)
        points = points[choice]
        sdf = sdf[choice]

        sample = np.random.choice(len(points), 128, replace=True)
        points,sdf = points[sample], sdf[sample]

        points = torch.from_numpy(points).float().to(device)
        ones = torch.ones([len(points), 1],device =device).float()
        points = torch.cat([points, ones], dim=-1)
        t = trans[i].squeeze()
        print(points.shape,t.shape)

        trans_points = torch.matmul(t,points.t()).t()[:,:3]
        pts.append(trans_points)
    pts = torch.cat(pts,dim=0).detach().cpu().numpy()
    print(pts.shape)
    scene = trimesh.Scene()
    robot_mesh = robot.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)
    scene.add_geometry(robot_mesh)
    pc =trimesh.PointCloud(pts,colors = [255,0,0])
    scene.add_geometry(pc)
    scene.show()
def plot_sdf_shell(robot,bp_sdf,pose,theta,model,device,distance=0.0):
    space_limits = robot.space_limits.cpu().numpy()
    N = 100000
    points = np.random.rand(N,3) * (space_limits[1]-space_limits[0]) + space_limits[0]
    points = torch.from_numpy(points).float().to(device)
    sdf,_ = bp_sdf.get_whole_body_sdf_batch(points,pose,theta,model,use_derivative=False)
    sdf = sdf.squeeze().detach().cpu().numpy()
    th = (space_limits.max() - space_limits.min()).mean() / 50.0
    print(th)
    choice = ((sdf-distance)<th) * ((sdf-distance)>-th)
    points = points[choice]
    sdf = sdf[choice]
    print(points.shape,sdf.shape)

    scene = trimesh.Scene()
    robot_mesh = robot.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)
    scene.add_geometry(robot_mesh)
    pc =trimesh.PointCloud(points.detach().cpu().numpy(),colors = [255,0,0,150])
    scene.add_geometry(pc)
    scene.show()

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--robot', default='panda', type=str,choices=['panda','dexhand','leaphand'])
    args = parser.parse_args()
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'data/{args.robot}/sdf_points/'),
        'model':os.path.join(CUR_DIR, f'models/{args.robot}/BP_{args.n_func}.pt')
        }
    robot = RobotLayer(device=args.device,paths=paths,robot=args.robot)
    bp_sdf = bf_sdf.BPSDF(args.n_func,args.domain_min,args.domain_max,robot,paths,args.device)

    #  load  model
    model = torch.load(f'models/{args.robot}/BP_{args.n_func}.pt')
    
    # --- initialize theta ---
    device = args.device
    theta = torch.randn([1,robot.dof]).to(device) * (robot.theta_max_soft - robot.theta_min_soft) + robot.theta_min_soft
    # theta = torch.zeros([1,robot.dof]).to(device)
    theta = theta.to(args.device).reshape(-1,robot.dof)
    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(len(theta),4,4).float()
    # # vis 2D SDF with gradient
    # plot_2D_panda_sdf(pose,theta,bp_sdf,nbData=80,model=model,device=args.device)

    # vis 3D SDF with gradient
    # plot_3D_panda_with_gradient(pose,theta,bp_sdf,model=model,device=args.device)
    plot_sdf_shell(robot,bp_sdf,pose,theta,model,device=args.device)
