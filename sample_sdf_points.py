
# -----------------------------------------------------------------------------
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
import skimage
import pyrender
import torch
import argparse
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Sample SDF points from mesh')
parser.add_argument('--robot', type=str, default='panda',choices=['panda', 'dexhand','leaphand'],
                    help='Robot type to sample SDF points from')
args = parser.parse_args()

# --- initialize mesh path depending on robot type ---
if args.robot == 'panda':
    mesh_path = os.path.join(CUR_PATH, 'panda_layer/meshes/voxel_128/*.stl')
elif args.robot == 'leaphand':
    mesh_path = os.path.join(CUR_PATH, 'descriptions/leaphand/meshes/*.stl')
elif args.robot == 'dexhand':
    mesh_path = os.path.join(CUR_PATH, 'descriptions/dexhand/right/*.stl')
   
    
mesh_files = glob.glob(mesh_path)
mesh_files = sorted(mesh_files)[:] #except finger

for mf in mesh_files:
    # 先检查一下，如果已经存在数据，就跳过
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/{args.robot}/sdf_points')
    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)
    
    mesh_name = mf.split('/')[-1].split('.')[0]
    print(mesh_name)
    data_path = os.path.join(save_path,f'voxel_128_{mesh_name}.npy')
    if os.path.exists(data_path):
        print(f"Data for {mf} already exists, skipping...")
        continue
    mesh = trimesh.load(mf)
    mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

    center = mesh.bounding_box.centroid
    scale = np.max(np.linalg.norm(mesh.vertices-center, axis=1))

    # sample points near surface (as same as deepSDF)
    near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh, 
                                                      number_of_points = 500000, 
                                                      surface_point_method='scan', 
                                                      sign_method='normal', 
                                                      scan_count=100, 
                                                      scan_resolution=400, 
                                                      sample_point_count=10000000, 
                                                      normal_sample_count=100, 
                                                      min_size=0.015, 
                                                      return_gradients=False)
    # # sample points randomly within the bounding box [-1,1]
    random_points = np.random.rand(500000,3)*2.0-1.0
    random_sdf = mesh_to_sdf.mesh_to_sdf(mesh, 
                                     random_points, 
                                     surface_point_method='scan', 
                                     sign_method='normal', 
                                     bounding_radius=None, 
                                     scan_count=100, 
                                     scan_resolution=400, 
                                     sample_point_count=10000000, 
                                     normal_sample_count=100) 
    
    # save data
    data = {
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    
    np.save(data_path, data)

    # # # for visualization
    # data = np.load(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points/voxel_128_{mesh_name}.npy')), allow_pickle=True).item()
    # random_points = data['random_points']
    # random_sdf = data['random_sdf']
    # near_points = data['near_points']
    # near_sdf = data['near_sdf']
    # colors = np.zeros(random_points.shape)
    # colors[random_sdf < 0, 2] = 1
    # colors[random_sdf > 0, 0] = 1
    # cloud = pyrender.Mesh.from_points(random_points, colors=colors)
    # scene = pyrender.Scene()
    # scene.add(cloud)
    # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
