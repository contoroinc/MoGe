import open3d as o3d
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

def print_device_info():
    # Print Open3D version
    print("Open3D version:", o3d.__version__)

    try:
        # Try to check CUDA availability
        cuda_available = hasattr(o3d, 'cuda') and o3d.cuda.is_available()
        print("\tCUDA available:", cuda_available)
    except:
        print("\tCUDA support not built into Open3D")
        cuda_available = False

    # Print device information
    try:
        print("Device information:")
        if hasattr(o3d, 'core'):
            print("\tDefault device:", o3d.core.Device.get_default())
        else:
            print("\tCore API not available in this Open3D version")
    except:
        print("Could not get device information")

import numpy as np
from scipy.spatial import cKDTree

def roe_solver(pred_points, gt_points, truncation=1.0):
    """ROE solver for optimal scale and z-translation."""
    pred_z = pred_points[:, 2]
    gt_z = gt_points[:, 2]
    
    # Sort points by z-coordinate for efficient search
    sort_idx = np.argsort(pred_z)
    pred_z_sorted = pred_z[sort_idx]
    
    best_error = float('inf')
    best_s = 1.0
    best_tz = 0.0
    
    # Try each point as anchor for z-alignment
    for k in range(len(pred_z)):
        # Compute scale and translation
        s = gt_z[k] / pred_z[k]
        tz = gt_z[k] - s * pred_z[k]
        
        # Compute errors
        errors = np.abs(s * pred_z + tz - gt_z)
        errors = np.minimum(errors, truncation)
        total_error = np.sum(errors)
        
        if total_error < best_error:
            best_error = total_error
            best_s = s
            best_tz = tz
            
    return best_s, best_tz

def get_actual_3d(affine_invariant, point_cloud, max_distance=0.1, truncation=1.0):
    """
    Convert affine-invariant predictions to actual 3D coordinates.
    
    Args:
        affine_invariant: Nx3 array of predicted points (x,y,z)
        point_cloud: Mx3 array of real 3D points from RGB-D/LiDAR
        max_distance: Maximum distance for point matching
        truncation: Truncation threshold for ROE solver
        
    Returns:
        aligned_points: Nx3 array of aligned predictions in real-world coordinates
    """
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(point_cloud)
    
    # Find corresponding points
    distances, indices = tree.query(affine_invariant, distance_upper_bound=max_distance)
    valid_matches = distances != np.inf
    
    pred_points = affine_invariant[valid_matches]
    gt_points = point_cloud[indices[valid_matches]]
    
    # Solve for optimal scale and translation
    scale, t_z = roe_solver(pred_points, gt_points, truncation)
    
    # Apply transformation
    aligned_points = affine_invariant.copy()
    aligned_points *= scale
    aligned_points[:, 2] += t_z
    
    return aligned_points
    
def transform_point_cloud(pcd, translation, quat_xyzw):
    """Transform point cloud using translation and quaternion rotation (xyzw format)"""
    # Create transformation matrix
    rotation_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    # Apply transformation
    return pcd.transform(transform), transform

def visualize_point_clouds(point_clouds, transforms=None):
    """Visualize multiple point clouds with their transforms and color tints"""
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    if not vis.create_window():
        print("Failed to create window, trying without window...")
        return
    
    # Set render options if available
    try:
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.asarray([1, 1, 1])
            opt.point_size = 1.0
    except:
        print("Could not set render options, continuing with defaults...")
    
    # Color tints
    tints = [
        [0.2, 0, 0],     # slight red tint
        [0, 0.2, 0],     # slight green tint
        [0, 0, 0.2],     # slight blue tint
        [0.15, 0.15, 0]  # slight yellow tint
    ]
    
    # Add each point cloud
    for i, pcd in enumerate(point_clouds):
        # Apply color tint
        colors = np.asarray(pcd.colors)
        if len(colors) == 0:
            colors = np.ones((len(np.asarray(pcd.points)), 3)) * 0.5
        tint = tints[i % len(tints)]
        colors = np.clip(colors + tint, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Add coordinate frame for this point cloud's transform
        if transforms is not None and i < len(transforms):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            frame.transform(transforms[i])
            vis.add_geometry(frame)
    
    # Add world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(world_frame)
    
    try:
        # Try to set a default viewpoint
        ctr = vis.get_view_control()
        if ctr is not None:
            ctr.set_zoom(0.8)
    except:
        print("Could not set default viewpoint...")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def load_and_visualize_from_yaml(yaml_file):
    """Load point clouds and their transforms from YAML and visualize"""
    # Read YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    point_clouds = []
    transforms = []
    
    # Process each point cloud in the YAML file
    for pc_name in ['rgbd_point_cloud', 'affine_invariant']:
        if pc_name in data:
            pc_data = data[pc_name]
            # Extract path, translation, and rotation from the list of dictionaries
            path = next(item['path'] for item in pc_data if 'path' in item)
            translation = next(item['Translation'] for item in pc_data if 'Translation' in item)
            rotation = next(item['Rotation'] for item in pc_data if 'Rotation' in item)
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(path)
            
            # Transform point cloud
            transformed_pcd, transform = transform_point_cloud(pcd, translation, rotation)
            
            point_clouds.append(transformed_pcd)
            transforms.append(transform)
    
    # Visualize all point clouds
    visualize_point_clouds(point_clouds, transforms)

if __name__ == "__main__":
    print_device_info()
    load_and_visualize_from_yaml('ply_info.yaml')