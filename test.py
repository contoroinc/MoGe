import open3d as o3d
import numpy as np
import copy
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

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

def apply_s_and_t(affine_invariant, scale: float, t: np.array):
    """Apply scale and translation to points."""
    assert t.shape == (3,)
    
    # Create new point cloud without copying the original
    aligned_points = o3d.geometry.PointCloud()
    
    # Get points, apply transformation, and set directly
    points = np.asarray(affine_invariant.points)
    points = points * scale  # Scale first
    points += t  # Then translate
    aligned_points.points = o3d.utility.Vector3dVector(points)
    
    # Copy other attributes if they exist
    if affine_invariant.has_colors():
        aligned_points.colors = affine_invariant.colors
    if affine_invariant.has_normals():
        aligned_points.normals = affine_invariant.normals
        
    return aligned_points

def transform_point_cloud(pcd, translation, quat_xyzw=[0, 0, 0, 1]):
    """Transform point cloud using translation and quaternion rotation (xyzw format)"""
    # Create transformation matrix
    rotation_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    # Apply transformation
    return pcd.transform(transform), transform

def crop_point_cloud_to_cube(pcd, center=[0, 0, 0], size=3.0):
    """
    Crop point cloud to a cube centered at 'center' with sides of length 'size'.
    
    Args:
        pcd: open3d.geometry.PointCloud
        center: Center coordinates of the cube [x, y, z]
        size: Length of cube sides
    
    Returns:
        open3d.geometry.PointCloud: Cropped point cloud
    """
    # Convert points to numpy array
    points = np.asarray(pcd.points)
    
    # Calculate bounds
    half_size = size / 2
    min_bound = np.array(center) - half_size
    max_bound = np.array(center) + half_size
    
    # Find points within bounds
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    # Create new point cloud with cropped points
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])
    
    # Copy colors if they exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    # Copy normals if they exist
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    return cropped_pcd
    
def add_tint_to_colors(pcd, tint, tint_strength=0.3):
    """Add tint to point cloud while preserving original colors"""
    # Get original colors or create default if none exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones((len(np.asarray(pcd.points)), 3)) * 0.7  # Default gray
    
    # Add tint while preserving original colors
    tinted_colors = np.clip(colors * (1 - tint_strength) + np.array(tint) * tint_strength, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(tinted_colors)
    return pcd

def interactive_visualization(point_clouds, transforms=None):
    """Interactive visualization with real-time control of scale and translation"""
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # Store original point cloud and initial parameters
    global original_pc, current_s, current_t, transformed_pc, point_size
    original_pc = copy.deepcopy(point_clouds[1])  # Store affine invariant point cloud
    current_t = np.array([-0.10, 0.00, -0.44])
    current_s = 1.22
    point_size = 5.0  # Smaller point size
    tint_strength = 0.3

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size  # Smaller point size
    opt.background_color = np.asarray([1, 1, 1])  # White background
    
    # Add reference point cloud with red tint
    ref_cloud = copy.deepcopy(point_clouds[0])
    ref_cloud = add_tint_to_colors(ref_cloud, [0.8, 0.2, 0.2], tint_strength=tint_strength)
    vis.add_geometry(ref_cloud)
    
    # Add transformed point cloud with blue tint
    transformed_pc = apply_s_and_t(original_pc, scale=current_s, t=current_t)
    transformed_pc = add_tint_to_colors(transformed_pc, [0.2, 0.2, 0.8], tint_strength=tint_strength)
    vis.add_geometry(transformed_pc)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coordinate_frame)
    
    def update_visualization():
        # Update transformed point cloud
        global transformed_pc, point_size
        # Get the transformed points
        new_points = np.asarray(original_pc.points) * current_s + current_t
        
        # Update the points directly
        transformed_pc.points = o3d.utility.Vector3dVector(new_points)

        opt = vis.get_render_option()
        opt.point_size = point_size  # Smaller point size

        # Update visualization
        vis.update_geometry(transformed_pc)
        vis.poll_events()
        vis.update_renderer()
        print(f"Scale: {current_s:.2f}, Translation: [{current_t[0]:.2f}, {current_t[1]:.2f}, {current_t[2]:.2f}], point_size: {point_size:.2f}")
    
    # Key callback functions
    def increase_scale(vis):
        global current_s
        current_s += 0.01
        update_visualization()
        return False
    
    def decrease_scale(vis):
        global current_s
        current_s -= 0.01
        update_visualization()
        return False
    
    def move_x_pos(vis):
        global current_t
        current_t[0] += 0.01
        update_visualization()
        return False
    
    def move_x_neg(vis):
        global current_t
        current_t[0] -= 0.01
        update_visualization()
        return False
    
    def move_y_pos(vis):
        global current_t
        current_t[1] += 0.01
        update_visualization()
        return False
    
    def move_y_neg(vis):
        global current_t
        current_t[1] -= 0.01
        update_visualization()
        return False
    
    def move_z_pos(vis):
        global current_t
        current_t[2] += 0.01
        update_visualization()
        return False
    
    def move_z_neg(vis):
        global current_t
        current_t[2] -= 0.01
        update_visualization()
        return False

    def smaller_pixel(vis):
        global point_size
        point_size -= 0.1
        if point_size <= 0.5:
            # warning msg
            point_size = 0.5
            print(f"Minimum pixel size reached: point size={point_size:.2f}")
            return
        update_visualization()
        return False
    
    def bigger_pixel(vis):
        global point_size
        point_size += 0.1
        if point_size >= 10.0:
            # warning msg
            point_size = 10.0
            print(f"Maximum pixel size reached: point size={point_size:.2f}")
            return
        update_visualization()
        return False
    
    # Register key callbacks
    vis.register_key_callback(ord('T'), increase_scale)  # t to increase scale
    vis.register_key_callback(ord('G'), decrease_scale)  # g to decrease scale
    vis.register_key_callback(ord('D'), move_x_pos)     # d to move right
    vis.register_key_callback(ord('A'), move_x_neg)     # a to move left
    vis.register_key_callback(ord('S'), move_y_pos)     # s to move up
    vis.register_key_callback(ord('W'), move_y_neg)     # w to move down
    vis.register_key_callback(ord('R'), move_z_pos)     # r to move forward
    vis.register_key_callback(ord('F'), move_z_neg)     # f to move backward
    vis.register_key_callback(ord('Z'), smaller_pixel)     # r to move forward
    vis.register_key_callback(ord('C'), bigger_pixel)     # f to move backward
    
    print("\nInteractive Controls:")
    print("t/g: Increase/Decrease Scale")
    print("d/a: Move Right/Left (X-axis)")
    print("s/w: Move Up/Down (Y-axis)")
    print("r/f: Move Forward/Backward (Z-axis)")
    print("Current values will be printed when you make changes")
    print("Hold Left mouse button to rotate view")
    print("Hold Right mouse button to pan")
    print("Scroll wheel to zoom")
    
    # Run visualization
    vis.run()
    vis.destroy_window()
    
    # Return final values
    return current_s, current_t

def load_and_visualize_from_yaml(yaml_file):
    """Load point clouds and their transforms from YAML and visualize"""
    # Load YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    point_clouds = []
    transforms = []
    # Process each point cloud in the YAML file
    for pc_name in ['rgbd_point_cloud', 'affine_invariant']:
        if pc_name in data:
            # Extract values from the list of dictionaries
            pc_data = data[pc_name]
            path = next(item['path'] for item in pc_data if 'path' in item)
            translation = next(item['Translation'] for item in pc_data if 'Translation' in item)
            rotation = next(item['Rotation'] for item in pc_data if 'Rotation' in item)
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(path)
            
            # Crop
            pcd = crop_point_cloud_to_cube(pcd, center=[0, 0, 0], size=5.0)

            # Transform point cloud
            transformed_pcd, transform = transform_point_cloud(pcd, translation, rotation)

            point_clouds.append(transformed_pcd)
            transforms.append(transform)
    
    # Visualize with interactive controls
    interactive_visualization(point_clouds, transforms)

if __name__ == "__main__":
    print_device_info()
    load_and_visualize_from_yaml('ply_info.yaml')