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
import pytorch_kinematics as pk
import math
import argparse

def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class RobotLayer(torch.nn.Module):
    def __init__(self, device,paths,robot='panda'):
        # The forward kinematics equations implemented here are     robot_mesh.show()from
        super().__init__()
        self.device = device
        self.robot = robot
        # --- initialize mesh path and urdf path ---
        self.mesh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), paths['meshes'])
        urdf_glob_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), paths['urdf'])
        urdf_files = glob.glob(urdf_glob_path)
        if len(urdf_files) == 0:
            raise FileNotFoundError(f"No URDF file found at {urdf_glob_path}")
        self.urdf_path = urdf_files[0]  # 取第一个匹配的URDF文件
        
        # --- initialize link chain ---
        if robot == 'panda':
            self.ee_links = ['panda_hand']
        elif robot == 'dexhand':
            self.ee_links = ['ring_tip_1','pinky_tip_1','middle_tip_1','index_tip_1','thumb_tip_1']
        elif robot == 'leaphand':
            self.ee_links = ['fingertip','fingertip_2','fingertip_3','thumb_fingertip']
            self.Link2Mesh = {
                'palm_base': 'palm_base',
                'palm_lower_left': 'palm_lower_left',
                'mcp_joint': 'mcp_joint',
                'pip': 'pip',
                'dip': 'dip',
                'fingertip': 'fingertip',
                'mcp_joint_2': 'mcp_joint',
                'pip_2': 'pip',
                'dip_2': 'dip',
                'fingertip_2': 'fingertip',
                'mcp_joint_3': 'mcp_joint',
                'pip_3': 'pip',
                'dip_3': 'dip',
                'fingertip_3': 'fingertip',
                'thumb_left_temp_base': 'thumb_left_temp_base',
                'thumb_pip': 'thumb_pip',
                'thumb_dip': 'thumb_dip',
                'thumb_fingertip': 'thumb_fingertip'
            }
        self.chain = {}
        self.transformations = {}
        self.DoFs = {}
        self.all_links = []
        for ee_link in self.ee_links:
            self.chain[ee_link] = pk.build_serial_chain_from_urdf(open(self.urdf_path).read().encode(),ee_link).to(dtype=torch.float32, device=self.device)
            self.DoFs[ee_link] = len(self.chain[ee_link].get_joint_parameter_names())
            self.all_links += self.chain[ee_link].get_link_names()
            print(f'Chain for {ee_link} initialized with DoF: {self.DoFs[ee_link]}')
            self.transformations[ee_link] = self.chain[ee_link].forward_kinematics(torch.zeros(1,self.DoFs[ee_link]).to(self.device))
            print(self.chain[ee_link])
        # --- initialize meshes ---
        self.meshes = self.load_meshes()
        self.links = list(self.meshes.keys())
        
        # --- initialize joint limits ---
        if robot == 'panda':
            self.theta_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,  -2.8973]).to(self.device)
            self.theta_max = torch.tensor([ 2.8973,	1.7628,	 2.8973,  -0.0698,  2.8973,	3.7525,	 2.8973]).to(self.device)
        elif robot == 'dexhand':
            #TODO
            pass
        elif robot == 'leaphand':
            self.theta_min =  torch.from_numpy(np.array([\
                    -1.5716, -0.4416, -1.2216, -1.3416,  1.0192,  0.0716,  0.2516, -1.3416,\
                    -1.5716, -0.4416, -1.2216, -1.3416, -1.5716, -0.4416, -1.2216, -1.3416\
                ])).to(self.device)
            self.theta_max = torch.from_numpy(np.array([\
                    1.5584, 1.8584, 1.8584, 1.8584, 1.7408, 1.0684, 1.8584, 1.8584, 1.5584,\
                    1.8584, 1.8584, 1.8584, 1.5584, 1.8584, 1.8584, 1.8584\
                ])).to(self.device)
        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min-self.theta_mid)*0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max-self.theta_mid)*0.8 + self.theta_mid
        self.dof = len(self.theta_min)

        # --- mesh faces/vertices/normals ---
        self.num_vertices_per_part = [self.meshes[link][0].shape[0] for link in self.links]
        self.robot_faces = [self.meshes[link][1] for link in self.links]
        self.link_vertices = {link: self.meshes[link][0] for link in self.links}
        self.link_normals = {link: self.meshes[link][-1] for link in self.links}

    # def check_normal(self,verterices, normals):
    #     center = np.mean(verterices,axis=0)
    #     verts = torch.from_numpy(verterices-center).float()
    #     normals = torch.from_numpy(normals).float()
    #     cosine = torch.cosine_similarity(verts,normals).float()
    #     normals[cosine<0] = -normals[cosine<0]
    #     return normals
    def get_link_transformations(self,base_pose, theta):
        transformations = {}
        
        for ee_link in self.ee_links:
            print(f'Computing transformations for {ee_link}')
            print(theta[ee_link].shape)
            transformations[ee_link] = self.chain[ee_link].forward_kinematics(theta[ee_link], end_only=False)
        for ee_link in self.ee_links:
            for k in transformations[ee_link].keys():
                matrix = transformations[ee_link][k].get_matrix()
                transformations[ee_link][k] = torch.matmul(base_pose, matrix)
        return transformations
    def load_meshes(self):
        check_normal = False
        # mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}

        for mesh_file in mesh_files:
            if self.mesh_path.split('/')[-2]=='visual':
                name = os.path.basename(mesh_file)[:-4].split('_')[0]
            else:
                name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)

            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                mesh.faces,
                torch.cat((torch.FloatTensor(np.array(mesh.vertex_normals)), temp), dim=-1).to(self.device).to(torch.float)
                ]
        return meshes

    def forward(self, pose, theta):
        batch_size = theta.shape[0]
        current_pose = pose.view(batch_size, 4, 4)
        link0_vertices = self.link0.repeat(batch_size, 1, 1)
        # print(link0_vertices.shape)
        link0_vertices = torch.matmul(pose,
                                      link0_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link0_normals = self.link0_normals.repeat(batch_size, 1, 1)
        link0_normals = torch.matmul(pose,
                                      link0_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        link1_vertices = self.link1.repeat(batch_size, 1, 1)
        T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.333, theta[:, 0], batch_size).float()


        link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.forward_kinematics(self.A1, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 1], batch_size).float()
        link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23 = self.forward_kinematics(self.A2, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0.316, theta[:, 2], batch_size).float()
        link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34 = self.forward_kinematics(self.A3, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 3], batch_size).float()
        link5_vertices = self.link5.repeat(batch_size, 1, 1)
        T45 = self.forward_kinematics(self.A4, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
                                      0.384, theta[:, 4], batch_size).float()
        link6_vertices = self.link6.repeat(batch_size, 1, 1)
        T56 = self.forward_kinematics(self.A5, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 5], batch_size).float()
        link7_vertices = self.link7.repeat(batch_size, 1, 1)
        T67 = self.forward_kinematics(self.A6, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 6], batch_size).float()
        link8_vertices = self.link8.repeat(batch_size, 1, 1)
        T78 = self.forward_kinematics(self.A7, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.107, -np.pi/4*torch.ones_like(theta[:,0],device=self.device), batch_size).float()
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)
        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        link1_vertices= torch.matmul(
            pose_to_T01,
            link1_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link1_normals = self.link1_normals.repeat(batch_size, 1, 1)
        link1_normals = torch.matmul(pose_to_T01,
                                     link1_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        
        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        link2_vertices= torch.matmul(
            pose_to_T12,
            link2_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link2_normals = self.link2_normals.repeat(batch_size, 1, 1)
        link2_normals = torch.matmul(pose_to_T12,
                                     link2_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
    
        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        link3_vertices= torch.matmul(
            pose_to_T23,
        link3_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link3_normals = self.link3_normals.repeat(batch_size, 1, 1)
        link3_normals = torch.matmul(pose_to_T23,
                                     link3_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        link4_vertices= torch.matmul(
            pose_to_T34,
            link4_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link4_normals = self.link4_normals.repeat(batch_size, 1, 1)
        link4_normals = torch.matmul(pose_to_T34,
                                     link4_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        link5_vertices= torch.matmul(
            pose_to_T45,
            link5_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link5_normals = self.link5_normals.repeat(batch_size, 1, 1)
        link5_normals = torch.matmul(pose_to_T45,
                                 link5_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        link6_vertices= torch.matmul(
            pose_to_T56,
            link6_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link6_normals = self.link6_normals.repeat(batch_size, 1, 1)
        link6_normals = torch.matmul(pose_to_T56,
                                     link6_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T67 = torch.matmul(pose_to_T56, T67)
        link7_vertices= torch.matmul(
            pose_to_T67,
        link7_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link7_normals = self.link7_normals.repeat(batch_size, 1, 1)
        link7_normals = torch.matmul(pose_to_T67,
                                     link7_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        

        pose_to_T78 = torch.matmul(pose_to_T67, T78)
        link8_vertices= torch.matmul(
            pose_to_T78,
        link8_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link8_normals = self.link8_normals.repeat(batch_size, 1, 1)
        link8_normals = torch.matmul(pose_to_T78,
                                    link8_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return [link0_vertices, link1_vertices, link2_vertices, \
                link3_vertices, link4_vertices, link5_vertices, \
                link6_vertices, link7_vertices, link8_vertices, \
                link0_normals, link1_normals, link2_normals, \
                link3_normals, link4_normals, link5_normals, \
                link6_normals, link7_normals, link8_normals]

    def get_transformations_each_link(self,pose, theta):
        batch_size = theta.shape[0]
        T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.333, theta[:, 0], batch_size).float()

        # link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.forward_kinematics(self.A1, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 1], batch_size).float()
        # link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23 = self.forward_kinematics(self.A2, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0.316, theta[:, 2], batch_size).float()
        # link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34 = self.forward_kinematics(self.A3, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 3], batch_size).float()
        # link5_vertices = self.link5.repeat(batch_size, 1, 1)
        T45 = self.forward_kinematics(self.A4, torch.tensor(-np.pi/2., dtype=torch.float32, device=self.device),
                                      0.384, theta[:, 4], batch_size).float()
        # link6_vertices = self.link6.repeat(batch_size, 1, 1)
        T56 = self.forward_kinematics(self.A5, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 5], batch_size).float()
        # link7_vertices = self.link7.repeat(batch_size, 1, 1)
        T67 = self.forward_kinematics(self.A6, torch.tensor(np.pi/2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 6], batch_size).float()
        # link8_vertices = self.link8.repeat(batch_size, 1, 1)
        T78 = self.forward_kinematics(self.A7, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.107, -np.pi/4*torch.ones_like(theta[:,0],device=self.device), batch_size).float()
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)
        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        pose_to_T67 = torch.matmul(pose_to_T56, T67)
        pose_to_T78 = torch.matmul(pose_to_T67, T78)

        return [pose_to_Tw0,pose_to_T01,pose_to_T12,pose_to_T23,pose_to_T34,pose_to_T45,pose_to_T56,pose_to_T67,pose_to_T78]

    def get_eef(self,pose, theta,link=-1):
        poses = self.get_transformations_each_link(pose, theta)
        pos = poses[link][:, :3, 3]
        rot = poses[link][:, :3, :3]
        return  pos, rot

    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        theta = theta.view(batch_size, -1)
        alpha = alpha*torch.ones_like(theta)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)

        l_1_to_l = torch.cat([c_theta, -s_theta, torch.zeros_like(s_theta), A * torch.ones_like(c_theta),
                                s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * D,
                                s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * D,
                                torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.ones_like(s_theta)], dim=1).reshape(batch_size, 4, 4)
        # l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        # l_1_to_l[:, 0, 0] = c_theta
        # l_1_to_l[:, 0, 1] = -s_theta
        # l_1_to_l[:, 0, 3] = A
        # l_1_to_l[:, 1, 0] = s_theta * c_alpha
        # l_1_to_l[:, 1, 1] = c_theta * c_alpha
        # l_1_to_l[:, 1, 2] = -s_alpha
        # l_1_to_l[:, 1, 3] = -s_alpha * D
        # l_1_to_l[:, 2, 0] = s_theta * s_alpha
        # l_1_to_l[:, 2, 1] = c_theta * s_alpha
        # l_1_to_l[:, 2, 2] = c_alpha
        # l_1_to_l[:, 2, 3] = c_alpha * D
        # l_1_to_l[:, 3, 3] = 1
        print(f'l_1_to_l shape: {l_1_to_l}')
        return l_1_to_l

    def get_robot_mesh(self, vertices_list, faces):

        link0_verts = vertices_list[0]
        link0_faces = faces[0]

        link1_verts = vertices_list[1]
        link1_faces = faces[1]

        link2_verts = vertices_list[2]
        link2_faces = faces[2]

        link3_verts = vertices_list[3]
        link3_faces = faces[3]

        link4_verts = vertices_list[4]
        link4_faces = faces[4]

        link5_verts = vertices_list[5]
        link5_faces = faces[5]

        link6_verts = vertices_list[6]
        link6_faces = faces[6]

        link7_verts = vertices_list[7]
        link7_faces = faces[7]

        link8_verts = vertices_list[8]
        link8_faces = faces[8]

        link0_mesh = trimesh.Trimesh(link0_verts, link0_faces)
        # link0_mesh.visual.face_colors = [150,150,150]
        link1_mesh = trimesh.Trimesh(link1_verts, link1_faces)
        # link1_mesh.visual.face_colors = [150,150,150]
        link2_mesh = trimesh.Trimesh(link2_verts, link2_faces)
        # link2_mesh.visual.face_colors = [150,150,150]
        link3_mesh = trimesh.Trimesh(link3_verts, link3_faces)
        # link3_mesh.visual.face_colors = [150,150,150]
        link4_mesh = trimesh.Trimesh(link4_verts, link4_faces)
        # link4_mesh.visual.face_colors = [150,150,150]
        link5_mesh = trimesh.Trimesh(link5_verts, link5_faces)
        # link5_mesh.visual.face_colors = [250,150,150]
        link6_mesh = trimesh.Trimesh(link6_verts, link6_faces)
        # link6_mesh.visual.face_colors = [250,150,150]
        link7_mesh = trimesh.Trimesh(link7_verts, link7_faces)
        # link7_mesh.visual.face_colors = [250,150,150]
        link8_mesh = trimesh.Trimesh(link8_verts, link8_faces)
        # link8_mesh.visual.face_colors = [250,150,150]

        robot_mesh = [
                       link0_mesh,
                       link1_mesh,
                       link2_mesh,
                       link3_mesh,
                       link4_mesh,
                       link5_mesh,
                       link6_mesh,
                       link7_mesh,
                       link8_mesh
        ]
        # robot_mesh = np.sum(robot_mesh)
        return robot_mesh

    def get_forward_robot_mesh(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        vertices_list = [[
                          outputs[0][i].detach().cpu().numpy(),
                          outputs[1][i].detach().cpu().numpy(),
                          outputs[2][i].detach().cpu().numpy(),
                          outputs[3][i].detach().cpu().numpy(),
                          outputs[4][i].detach().cpu().numpy(),
                          outputs[5][i].detach().cpu().numpy(),
                          outputs[6][i].detach().cpu().numpy(),
                          outputs[7][i].detach().cpu().numpy(),
                          outputs[8][i].detach().cpu().numpy()] for i in range(batch_size)]
        
        mesh = [self.get_robot_mesh(vertices, self.robot_faces) for vertices in vertices_list]
        return mesh

    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        robot_vertices = torch.cat((
                                   outputs[0].view(batch_size, -1, 3),
                                   outputs[1].view(batch_size, -1, 3),
                                   outputs[2].view(batch_size, -1, 3),
                                   outputs[3].view(batch_size, -1, 3),
                                   outputs[4].view(batch_size, -1, 3),
                                   outputs[5].view(batch_size, -1, 3),
                                   outputs[6].view(batch_size, -1, 3),
                                   outputs[7].view(batch_size, -1, 3),
                                   outputs[8].view(batch_size, -1, 3)), 1)  # .squeeze()

        robot_vertices_normal = torch.cat((
                                   outputs[9].view(batch_size, -1, 3),
                                   outputs[10].view(batch_size, -1, 3),
                                   outputs[11].view(batch_size, -1, 3),
                                   outputs[12].view(batch_size, -1, 3),
                                   outputs[13].view(batch_size, -1, 3),
                                   outputs[14].view(batch_size, -1, 3),
                                   outputs[15].view(batch_size, -1, 3),
                                   outputs[16].view(batch_size, -1, 3),
                                   outputs[17].view(batch_size, -1, 3)), 1)  # .squeeze()

        return robot_vertices,robot_vertices_normal



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
    panda = RobotLayer(device,paths=paths,robot=args.robot).to(device)
    # scene = trimesh.Scene()

    # # show robot
    # # theta = panda.theta_min + (panda.theta_max-panda.theta_min)*0.5
    # # theta = torch.tensor([0, 0.8, -0.0, -2.3, -2.8, 1.5, np.pi/4.0]).float().to(device).reshape(-1,7)
    # theta = torch.rand(1,7).float().to(device)
    # pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),-1,-1).float()
    # robot_mesh = panda.get_forward_robot_mesh(pose, theta)
    # robot_mesh = np.sum(robot_mesh)
    # trimesh.exchange.export.export_mesh(robot_mesh, os.path.join('output_meshes',f"whole_body_levelset_0.stl"))
    # robot_mesh.show()