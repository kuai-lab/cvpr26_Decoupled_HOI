import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np 
import joblib 
import trimesh 
import json 
import time 
from pathlib import Path

import torch 


DEFAULT_DATA_ROOT = os.environ.get("DECHOI_BEHAVE_DATA_ROOT", "./datasets")
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_ROOT = Path(os.environ.get("DECHOI_PROCESSED_DATA", REPO_ROOT / "data" / "processed_data"))

def generate_sdf(mesh, dest_json_path, dest_sdf_path, dest_voxel_mesh_path="", grid_dim=256, print_time=False):
    from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
    # Save centroid and extents data used for transforming vertices to [-1,1] while query
    # vertices = mesh.vertices - mesh.bounding_box.centroid
    # vertices *= 2 / np.max(mesh.bounding_box.extents)
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents
    # Save centroid and extents as SDF
    json_dict = {}
    json_dict['centroid'] = centroid.tolist()
    json_dict['extents'] = extents.tolist()
    json_dict['grid_dim'] = grid_dim
    json.dump(json_dict, open(dest_json_path, 'w'))
    
    if print_time:
        start_time = time.time() 
    
    sdf = mesh_to_voxels(mesh, voxel_resolution=grid_dim)
    
    if dest_voxel_mesh_path != "":
        import skimage.measure
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxel_mesh.export(open(dest_voxel_mesh_path, 'w'), file_type='obj')
    
    if print_time:
        print("Generating SDF took {0} seconds".format(time.time()-start_time))
    
    np.save(dest_sdf_path, sdf)
    
    centroid = np.copy(centroid)
    extents = np.copy(extents)
    
    return centroid, extents, sdf

def get_rest_obj_sdf():
    data_root_folder = str(PROCESSED_DATA_ROOT)

    # Load rest object geometry 
    object_geo_folder = os.path.join(data_root_folder, "rest_object_geo")
    dest_obj_sdf_folder = os.path.join(data_root_folder, "rest_object_sdf_256_npy_files")
    if not os.path.exists(dest_obj_sdf_folder):
        os.makedirs(dest_obj_sdf_folder)
  
    object_files = os.listdir(object_geo_folder)
    for object_name in object_files:
        if ".ply" in object_name:
            object_ply_path = os.path.join(object_geo_folder, object_name) 
            obj_mesh = trimesh.load_mesh(object_ply_path)
                
            dest_json_path = os.path.join(dest_obj_sdf_folder, object_name+".json")
            dest_sdf_path = os.path.join(dest_obj_sdf_folder, object_name+".npy")
            dest_voxel_mesh_path = os.path.join(dest_obj_sdf_folder, object_name+".obj")

            generate_sdf(obj_mesh, dest_json_path, dest_sdf_path, \
            dest_voxel_mesh_path, grid_dim=256, print_time=False)
    
def get_behave_obj_sdf():
    data_root_folder = DEFAULT_DATA_ROOT

    # Load rest object geometry 
    object_geo_folder = os.path.join(data_root_folder, "objects")
    dest_obj_sdf_folder = os.path.join(data_root_folder, "behave_object_sdf_256_npy_files")
    if not os.path.exists(dest_obj_sdf_folder):
        os.makedirs(dest_obj_sdf_folder)
  
    object_files = os.listdir(object_geo_folder)
    for object_name in object_files:
        if "yoga" in object_name:
            continue 

        if "basketball" in object_name:
            continue

        if "backpack" in object_name:
            continue 

        if "keyboard" in object_name:
            continue 

        object_ply_path = os.path.join(object_geo_folder, object_name, object_name+".obj") 
        obj_mesh = trimesh.load_mesh(object_ply_path)
            
        dest_json_path = os.path.join(dest_obj_sdf_folder, object_name+".json")
        dest_sdf_path = os.path.join(dest_obj_sdf_folder, object_name+".npy")
        dest_voxel_mesh_path = os.path.join(dest_obj_sdf_folder, object_name+".obj")

        generate_sdf(obj_mesh, dest_json_path, dest_sdf_path, \
        dest_voxel_mesh_path, grid_dim=256, print_time=False)

if __name__ == "__main__":
    # get_rest_obj_sdf()

    get_behave_obj_sdf() 
