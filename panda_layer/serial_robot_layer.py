from robot_layer import RobotLayer
import argparse
import numpy as np
import os
import torch
import trimesh
class SerialRobotLayer(torch.nn.Module):
    def __init__(self,chain,meshes,Link2Mesh,LinkMeshTrans,scale,device):
        self.device = device
        self.chain = chain
        self.all_links = self.chain.get_link_names()
        self.dof = len(self.chain.get_joint_parameter_names())
        self.Link2Mesh = {link: Link2Mesh[link] for link in self.all_links}
        self.LinkMeshTrans = {link: LinkMeshTrans[link] for link in self.all_links}
        self.scale = {link: scale[link] for link in self.all_links}
        joints = self.chain.get_joint_parameter_names()
        self.Joint2Idx = {joint:idx for idx,joint in enumerate(joints)}
        low, high = self.chain.get_joint_limits()
        self.joint_limits = {joint: [low[i], high[i]] for i,joint in enumerate(joints)}
        self.theta_min = torch.tensor([self.joint_limits[joint][0] for joint in joints]).to(self.device)
        self.theta_max = torch.tensor([self.joint_limits[joint][1] for joint in joints]).to(self.device)
        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min-self.theta_mid)*0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max-self.theta_mid)*0.8 + self.theta_mid
        self.meshes = {Link2Mesh[link]: meshes[Link2Mesh[link]] for link in self.all_links if Link2Mesh[link] is not None}

    def get_link_transformations(self,base_pose, theta):
        # theta: (B, dof)
        # base_pose: (B, 4, 4)
        # return transformation: (Nl, B, 4, 4)
        B = theta.shape[0]
        transformation = self.chain.forward_kinematics(theta,end_only=False)
        transformation = torch.stack([transformation[k].get_matrix() for k in transformation.keys()],dim=0)# temp_trans: (Nl, B, 4, 4)
        transformation = torch.matmul(base_pose.unsqueeze(0), transformation) # temp_trans: (Nl, B, 4, 4)
        return transformation
    def get_link_mesh_transformations(self, base_pose, theta):
        # theta: (B, dof)
        # base_pose: (B, 4, 4)
        batch_size = theta.shape[0]
        trans = self.get_link_transformations(base_pose, theta)
        trans_idx = 0
        trans_full = {}
        for link in self.all_links:
            if link in self.Link2Mesh.keys() and self.Link2Mesh[link] is not None:
                trans_full[link] = torch.matmul(trans[trans_idx,:,:,:], self.LinkMeshTrans[link].to(self.device).unsqueeze(0).expand(batch_size,4,4))          
            trans_idx += 1
        return trans_full
    def forward(self, pose, theta):
        batch_size = theta.shape[0]
        vertices ={k: v[0].repeat(batch_size, 1, 1) for k,v in self.meshes.items()}# {mesh_name,(B, Nv, 4)}
        normals = {k: v[-1].repeat(batch_size, 1, 1) for k,v in self.meshes.items()}# {mesh_name,(B, Nv, 3)}
        trans = self.get_link_transformations(pose, theta)
        # trans : (Nl, B, 4, 4)) Nl=number of links(including those not in self.Link2Mesh)        
        # the keys of vertices and normals are the same, and are mesh names(instead of link names)
        transformed_vertices = {}
        transformed_normals = {}
        # transeformed_vertices : (link, (B, Nv, 3))
        # transeformed_normals : (link, (B, Nv, 3))
        trans_idx = 0
        for link in self.all_links:
            if link in self.Link2Mesh.keys() and self.Link2Mesh[link] is not None:
                trans_full = torch.matmul(trans[trans_idx,:,:,:], self.LinkMeshTrans[link].to(self.device).unsqueeze(0).expand(batch_size,4,4))
                transformed_vertices[link] = torch.matmul(trans_full, vertices[self.Link2Mesh[link]].transpose(2, 1)).transpose(1, 2)
                transformed_vertices[link] = transformed_vertices[link][:, :, :3]  # remove the homogeneous coordinate
                transformed_normals[link] = torch.matmul(trans_full, normals[self.Link2Mesh[link]].transpose(2, 1)).transpose(1, 2)
                transformed_normals[link] = transformed_normals[link][:, :, :3]                
            trans_idx += 1
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
    robot = RobotLayer(device=device,paths=paths,robot=args.robot).to(device)
    serial = {}
    
    for ee_link in robot.ee_links:
        serial[ee_link] = SerialRobotLayer(device=device,paths=paths,robot=args.robot,ee_link=ee_link).to(device)
        for s_ee_link in serial[ee_link].ee_links:
            print(f"ee_link: {s_ee_link}, dof: {serial[ee_link].dof}")
            print(f"chain: {serial[ee_link].chain}")
            print(f"joints: {serial[ee_link].Joint2Idx.keys()}")
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