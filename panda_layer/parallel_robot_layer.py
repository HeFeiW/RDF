        # self.leap_dof_lower = torch.from_numpy(np.array([
        #     -1.5716, -0.4416, -1.2216, -1.3416,  1.0192,  0.0716,  0.2516, -1.3416,
        #     -1.5716, -0.4416, -1.2216, -1.3416, -1.5716, -0.4416, -1.2216, -1.3416
        # ])).to(self.device)
        # self.leap_dof_upper = torch.from_numpy(np.array([
        #     1.5584, 1.8584, 1.8584, 1.8584, 1.7408, 1.0684, 1.8584, 1.8584, 1.5584,
        #     1.8584, 1.8584, 1.8584, 1.5584, 1.8584, 1.8584, 1.8584
        # ])).to(self.device)
        
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import trimesh
import glob
import os
import numpy as np
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'..'))
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import math
import argparse
import utils
from serial_robot_layer import SerialRobotLayer

def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class ParallelRobotLayer(torch.nn.Module):
    def __init__(self, device,paths,robot='panda'):
        # The forward kinematics equations implemented here are     robot_mesh.show()from
        super().__init__()
        self.device = device
        self.robot = robot
        # --- initialize mesh path and urdf path ---
        self.paths = paths
        self.mesh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), paths['meshes'])
        urdf_glob_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), paths['urdf'])
        urdf_files = glob.glob(urdf_glob_path)
        if len(urdf_files) == 0:
            raise FileNotFoundError(f"No URDF file found at {urdf_glob_path}")
        self.paths['urdf'] = urdf_files[0]  # 取第一个匹配的URDF文件
        
        # --- initialize link chain ---
        if robot == 'panda':
            self.ee_links = ['panda_leftfinger', 'panda_rightfinger']
            self.space_limits = torch.tensor([[-0.5,-0.5,0],[0.5,0.5,1.0]])
        elif robot == 'dexhand':
            self.ee_links = ['ring_tip_1','pinky_tip_1','middle_tip_1','index_tip_1','thumb_tip_1','hand-cover_1']
            self.space_limits = torch.tensor([[-0.3,-0.2,0.2],[0.3,0.2,0.4]])
        elif robot == 'leaphand':
            self.ee_links = ['fingertip','fingertip_2','fingertip_3','thumb_fingertip']
            self.space_limits = torch.tensor([[-0.3,-0.3,-0.15],[0.1,0.1,0.2]])
            
        # --- initialize meshes ---
        self.meshes = self.load_meshes()
        self.LinkMeshTrans = self.parse_urdf_origins(self.paths['urdf'])
        self.Link2Mesh = self.parse_link2mesh(self.paths['urdf'])
        
       
        self.serials = []
        for ee_link in self.ee_links:
            chain = pk.build_serial_chain_from_urdf(open(self.paths['urdf']).read().encode(),ee_link).to(dtype=torch.float32, device=self.device)
            self.serials.append(SerialRobotLayer(chain=chain,
                                                 meshes=self.meshes,
                                                 Link2Mesh=self.Link2Mesh,
                                                 LinkMeshTrans=self.LinkMeshTrans,
                                                 scale=self.scale,
                                                 device=self.device
                                                ))
        # --- initialize joints ---
        self.chain = pk.build_chain_from_urdf(open(self.paths['urdf']).read().encode()).to(dtype=torch.float32, device=self.device)
        joints_info_dict = {}
        for serial in self.serials:
            joints_info_dict.update(serial.joint_limits)
        self.dof = len(joints_info_dict)
        self.joint_limits = {joint: joints_info_dict[joint] for joint in joints_info_dict.keys()}
        self.Joint2Idx = {joint: idx for idx, joint in enumerate(joints_info_dict.keys())}
        
        # # --- initialize joint limits ---
        self.theta_min = torch.tensor([limit[0] for limit in self.joint_limits.values()]).to(self.device)
        self.theta_max = torch.tensor([limit[1] for limit in self.joint_limits.values()]).to(self.device)
        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min-self.theta_mid)*0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max-self.theta_mid)*0.8 + self.theta_mid
        self.dof = len(self.theta_min)

    def load_meshes(self):
        check_normal = False
        # mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        self.scale = self.parse_scale(self.paths['urdf'])
        for mesh_file in mesh_files:
            if 'panda_' in os.path.basename(mesh_file):
                name = os.path.basename(mesh_file).split('panda_')[-1][:-4]
            if "_vis" in os.path.basename(mesh_file):
                name = os.path.basename(mesh_file).split('_vis')[0]
            else:
                name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            # print(f'Loading mesh for {name}')
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            if name in self.scale.keys():
                mesh.vertices *= self.scale[name]
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                mesh.faces,
                torch.cat((torch.FloatTensor(np.array(mesh.vertex_normals)), temp), dim=-1).to(self.device).to(torch.float)
                ]
        return meshes
    def parse_scale(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        scale = {}
        for link in root.findall('link'):
            link_name = link.attrib['name']
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None and 'scale' in mesh.attrib:
                        scale_values = mesh.attrib['scale'].split()
                        scale[link_name] = [float(value) for value in scale_values]
                    else:
                        scale[link_name] = [1.0, 1.0, 1.0]
            else:
                scale[link_name] = [1.0, 1.0, 1.0]
        return scale
    def parse_link2mesh(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        Link2Mesh = {}
        self.all_links = []
        for link in root.findall('link'):
            link_name = link.attrib['name']
            self.all_links.append(link_name)
            mesh_name = None
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None and 'filename' in mesh.attrib:
                        mesh_file = mesh.attrib['filename']
                        # 只保留文件名，不含路径和扩展名
                        mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
            Link2Mesh[link_name] = mesh_name
        return Link2Mesh
    def parse_urdf_origins(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        LinkMeshTrans = {}
        for link in root.findall('link'):
            name = link.attrib['name']
            origin_elem = None
            # 优先visual的origin
            visual = link.find('visual')
            if visual is not None:
                origin_elem = visual.find('origin')
            # 其次collision的origin
            if origin_elem is None:
                collision = link.find('collision')
                if collision is not None:
                    origin_elem = collision.find('origin')
            # 默认无变换
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin_elem is not None:
                xyz = [float(x) for x in origin_elem.attrib.get('xyz', '0 0 0').split()]
                rpy = [float(r) for r in origin_elem.attrib.get('rpy', '0 0 0').split()]
            # 欧拉角转旋转矩阵
            rot_mat = torch.from_numpy(utils.euler_to_matrix(np.array(rpy)))
            xyz_mat = torch.tensor(xyz).view(3, 1)
            # rot_mat xyz_mat 变成 4x4
            rot_mat = torch.cat((rot_mat, torch.zeros(1, 3)), dim=0)
            rot_mat = torch.cat((rot_mat, torch.tensor([[0], [0], [0], [1]])), dim=1)
            xyz_mat = torch.cat((xyz_mat, torch.tensor([[0]])), dim=0)
            xyz_mat = torch.cat((torch.zeros(4,3), xyz_mat), dim=1)
            trans_mat = rot_mat + xyz_mat
            LinkMeshTrans[name] = trans_mat.to(self.device).float()
        return LinkMeshTrans
    def get_link_mesh_transformations(self, base_pose, theta):
        # theta: (B, dof)
        # base_pose: (B, 4, 4)
        trans = {}
        for serial in self.serials:
            serial_theta = torch.stack([theta[:,serial.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1)
            serial_trans = serial.get_link_mesh_transformations(base_pose, serial_theta)
            for link in serial.all_links:
                if link in self.Link2Mesh.keys() and self.Link2Mesh[link] is not None:
                    trans[link] = serial_trans[link]
        print(len(trans))
        return trans
    def forward(self, pose, theta):
        batch_size = theta.shape[0]
        vertices ={k: v[0].repeat(batch_size, 1, 1) for k,v in self.meshes.items()}# {mesh_name,(B, Nv, 4)}
        normals = {k: v[-1].repeat(batch_size, 1, 1) for k,v in self.meshes.items()}# {mesh_name,(B, Nv, 3)}
        trans = self.get_link_mesh_transformations(pose, theta)
        # trans : (Nl, B, 4, 4)) Nl=number of links(including those not in self.Link2Mesh)        
        # the keys of vertices and normals are the same, and are mesh names(instead of link names)
        transformed_vertices = {}
        transformed_normals = {}
        # transeformed_vertices : (link, (B, Nv, 3))
        # transeformed_normals : (link, (B, Nv, 3))
        for link in self.all_links:
            if link in self.Link2Mesh.keys() and self.Link2Mesh[link] is not None:
                transformed_vertices[link] = torch.matmul(trans[link], vertices[self.Link2Mesh[link]].transpose(2, 1)).transpose(1, 2)
                transformed_vertices[link] = transformed_vertices[link][:, :, :3]  # remove the homogeneous coordinate
                transformed_normals[link] = torch.matmul(trans[link], normals[self.Link2Mesh[link]].transpose(2, 1)).transpose(1, 2)
                transformed_normals[link] = transformed_normals[link][:, :, :3]        
        return transformed_vertices, transformed_normals

    def get_robot_mesh(self, vertices_list, faces):
        # vertices_list : {link, (Nv, 3)} # Nl=number of links that have meshes,
        # faces : (Nm) Nm=number of meshes
        meshes = [trimesh.Trimesh(vertices_list[link].detach().cpu().numpy(), faces[self.Link2Mesh[link]]) for link in vertices_list.keys()]
        return meshes

    def get_forward_robot_mesh(self, pose, theta):
        vertices, _ = self.forward(pose, theta)
        # vertices : (link, (B, Nv, 3))
        # normals : (link, (B, Nv, 3))
        # print(f'vertices keys: {vertices.keys()}')
        # print(f'self.Link2Mesh keys: {self.Link2Mesh.keys()}')
        for k in vertices.keys():
            B = vertices[k].shape[0]
            break
        temp_vertices_list = {link:vertices[link] for link in self.Link2Mesh.keys() if self.Link2Mesh[link] is not None}
        # verices: (Nlm, B, Nv, 3)->(B, Nlm, Nv, 3) Nlm=number of links that have meshes,
        # if the robot have repeated links (because of multiple ee_links)
        # it will hit only once
        vertices_list = [{k:temp_vertices_list[k][b] for k in temp_vertices_list.keys()} for b in range(B)]
        # print(f'vertices_list len: {len(vertices_list)}, vertices_list[0] len: {len(vertices_list[0])}')
        # vertices_list : (B, {link,(Nv, 3)}) there are Nlm links （the links that have meshes) 
        robot_faces =  {link:self.meshes[link][1] for link in self.meshes.keys()} # (Nm) Nm=number of meshes
        mesh = [self.get_robot_mesh(vertices, robot_faces) for vertices in vertices_list]
        # mesh: (B, Nl) Nl=number of links that have meshes, 
        # may have repeated links if the robot has multiple ee_links
        return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='panda',choices=['panda', 'dexhand','leaphand'],
                        help='Robot type to sample SDF points from')
    args = parser.parse_args()
    device = 'cuda'
    paths = {
        'urdf': f'../descriptions/{args.robot}/*.urdf',
        'meshes': f'../descriptions/{args.robot}/meshes/*.stl'
    }
    robot = ParallelRobotLayer(device,paths=paths,robot=args.robot).to(device)
    scene = trimesh.Scene()
    mesh = robot.get_forward_robot_mesh(
        torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(1,-1,-1).float(),
        torch.zeros(1,robot.dof).float().to(device)
    )[0]
    scene.add_geometry(mesh)
    scene.show()
    # # show robot
    # # theta = panda.theta_min + (panda.theta_max-panda.theta_min)*0.5
    # # theta = torch.tensor([0, 0.8, -0.0, -2.3, -2.8, 1.5, np.pi/4.0]).float().to(device).reshape(-1,7)
    # theta = torch.rand(1,7).float().to(device)
    # pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),-1,-1).float()
    # robot_mesh = panda.get_forward_robot_mesh(pose, theta)
    # robot_mesh = np.sum(robot_mesh)
    # trimesh.exchange.export.export_mesh(robot_mesh, os.path.join('output_meshes',f"whole_body_levelset_0.stl"))
    # robot_mesh.show()