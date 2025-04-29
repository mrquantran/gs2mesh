# prune_gaussians_dbscan.py

import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import os
from tqdm import tqdm
import time # For timing

import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN
CUML_AVAILABLE = True
print("cuML found. GPU DBSCAN enabled.")

def construct_list_of_attributes(max_sh_degree):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3): # DC features are always 3 (RGB)
        l.append(f'f_dc_{i}')
    num_rest_features = 3 * (max_sh_degree + 1) ** 2 - 3
    for i in range(num_rest_features):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(3): # Scaling features are always 3 (x,y,z)
        l.append(f'scale_{i}')
    for i in range(4): # Rotation features are always 4 (quaternion)
        l.append(f'rot_{i}')
    return l

def read_ply_data(ply_path, max_sh_degree):
    """Reads all necessary attributes from the Gaussian Splatting PLY file."""
    try:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        print(f"Available keys in vertices: {vertices.properties}")
    except FileNotFoundError:
        print(f"Error: Input PLY file not found at {ply_path}")
        return None
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return None

    num_points = len(vertices)
    print(f"Reading {num_points} Gaussians from {ply_path}")

    # --- Extract XYZ ---
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32) # Ensure float32 for GPU

    # --- Extract Normals (often zeros) ---
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T.astype(np.float32)

    # --- Extract Features (DC and Rest) ---
    f_dc = np.zeros((num_points, 3), dtype=np.float32)
    f_dc[:, 0] = vertices['f_dc_0']
    f_dc[:, 1] = vertices['f_dc_1']
    f_dc[:, 2] = vertices['f_dc_2']

    num_rest_features = 3 * (max_sh_degree + 1) ** 2 - 3
    f_rest = np.zeros((num_points, num_rest_features), dtype=np.float32)
    available_keys = {prop.name for prop in vertices.properties}

    loaded_rest_count = 0
    for i in range(num_rest_features):
        key_name = f'f_rest_{i}'
        if key_name in available_keys:
            try:
                f_rest[:, i] = vertices[key_name]
                loaded_rest_count += 1
            except ValueError as e:
                 print(f"Warning: Could not read {key_name}. Setting column to zero. Error: {e}")

    if loaded_rest_count != num_rest_features:
         print(f"Warning: Expected {num_rest_features} f_rest features based on max_sh_degree={max_sh_degree}, but only found/loaded {loaded_rest_count} in the PLY file.")

    # --- Extract Opacity ---
    opacities = vertices['opacity'][:, np.newaxis].astype(np.float32)

    # --- Extract Scale ---
    scales = np.zeros((num_points, 3), dtype=np.float32)
    scales[:, 0] = vertices['scale_0']
    scales[:, 1] = vertices['scale_1']
    scales[:, 2] = vertices['scale_2']

    # --- Extract Rotation ---
    rotations = np.zeros((num_points, 4), dtype=np.float32)
    rotations[:, 0] = vertices['rot_0']
    rotations[:, 1] = vertices['rot_1']
    rotations[:, 2] = vertices['rot_2']
    rotations[:, 3] = vertices['rot_3']

    return {
        "xyz": xyz,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }


def save_ply_data(output_path, data_dict, max_sh_degree):
    """Saves the filtered Gaussian data back into a PLY file."""
    xyz = data_dict["xyz"]
    num_points = xyz.shape[0]

    print(f"Saving {num_points} pruned Gaussians to {output_path}")
    if num_points == 0:
        print("Warning: No points left after pruning. Saving empty PLY file.")

    attribute_names = construct_list_of_attributes(max_sh_degree)
    dtype_full = [(name, 'f4') for name in attribute_names] # Use 'f4' (float32)

    elements = np.empty(num_points, dtype=dtype_full)

    # Assign attributes safely, checking if they exist in the filtered dict
    # Order must match construct_list_of_attributes implicitly via the loop
    for name in attribute_names:
        if name == 'x': elements['x'] = data_dict["xyz"][:, 0]
        elif name == 'y': elements['y'] = data_dict["xyz"][:, 1]
        elif name == 'z': elements['z'] = data_dict["xyz"][:, 2]
        elif name == 'nx': elements['nx'] = data_dict["normals"][:, 0]
        elif name == 'ny': elements['ny'] = data_dict["normals"][:, 1]
        elif name == 'nz': elements['nz'] = data_dict["normals"][:, 2]
        elif name.startswith('f_dc_'):
            idx = int(name.split('_')[-1])
            if idx < data_dict["f_dc"].shape[1]:
                elements[name] = data_dict["f_dc"][:, idx]
            # else: default to 0 (already handled by np.empty if dtype includes it)
        elif name.startswith('f_rest_'):
            idx = int(name.split('_')[-1])
            if idx < data_dict["f_rest"].shape[1]:
                 elements[name] = data_dict["f_rest"][:, idx]
            # else: default to 0
        elif name == 'opacity': elements[name] = data_dict["opacities"][:, 0] # Opacity was (N,1)
        elif name.startswith('scale_'):
            idx = int(name.split('_')[-1])
            elements[name] = data_dict["scales"][:, idx]
        elif name.startswith('rot_'):
            idx = int(name.split('_')[-1])
            elements[name] = data_dict["rotations"][:, idx]
        else:
             print(f"Warning: Unhandled attribute name during saving: {name}")


    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element], text=False).write(output_path) # Use binary format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune Gaussian Splatting PLY file using DBSCAN.")
    parser.add_argument("--input_ply", type=str, required=True, help="Path to the input PLY file (e.g., iteration_30000/point_cloud.ply)")
    parser.add_argument("--output_ply", type=str, required=True, help="Path to save the pruned PLY file.")
    parser.add_argument("--eps", type=float, required=True, help="DBSCAN eps parameter (maximum distance between samples). Needs tuning per scene!")
    parser.add_argument("--min_samples", type=int, default=10, help="DBSCAN min_samples parameter (number of samples in a neighborhood).")
    parser.add_argument("--max_sh_degree", type=int, default=3, help="Maximum SH degree used when training the input PLY.")
    parser.add_argument("--keep_cluster_ids", nargs='+', type=int, default=None, help="Optional: List specific cluster IDs to keep (besides non-noise). Default keeps all non-noise clusters.")
    parser.add_argument("--keep_largest_cluster_only", action='store_true', help="If set, only keep the largest cluster identified (ignores --keep_cluster_ids).")
    parser.add_argument("--use_gpu", action='store_true', help="Attempt to use GPU DBSCAN (cuML). Falls back to CPU if unavailable.")

    args = parser.parse_args()

    # --- 1. Load PLY Data ---
    print(f"Loading PLY data from: {args.input_ply}")
    start_load = time.time()
    loaded_data = read_ply_data(args.input_ply, args.max_sh_degree)
    if loaded_data is None:
        exit(1)
    load_time = time.time() - start_load
    print(f"PLY data loaded in {load_time:.2f} seconds.")


    xyz_centers = loaded_data["xyz"]
    num_original_points = xyz_centers.shape[0]

    if num_original_points == 0:
        print("Input PLY file contains no points. Exiting.")
        exit(0)

    print(f"Loaded {num_original_points} Gaussians.")

    # --- 2. Run DBSCAN ---
    labels = None
    dbscan_time = 0.0

    # Try GPU DBSCAN first if requested and available
    if args.use_gpu and CUML_AVAILABLE:
        print(f"Running GPU DBSCAN (cuML) with eps={args.eps}, min_samples={args.min_samples}...")
        try:
            start_dbscan = time.time()
            # Create GPU arrays
            xyz_gpu = cp.asarray(xyz_centers)

            # Run cuML DBSCAN
            cudbscan = cuDBSCAN(eps=args.eps, min_samples=args.min_samples, output_type='cupy')
            labels_gpu = cudbscan.fit_predict(xyz_gpu)

            # --- IMPORTANT FIX: Use correct API for synchronization ---
            cp.cuda.get_current_stream().synchronize() # Ensure GPU work is done

            # Copy results back to CPU
            labels = cp.asnumpy(labels_gpu)
            dbscan_time = time.time() - start_dbscan
            print(f"GPU DBSCAN completed in {dbscan_time:.2f} seconds.")

        except Exception as e:
            print(f"\nError during GPU DBSCAN execution: {e}")
            print("Ensure RAPIDS cuML is installed correctly and compatible with your drivers/CUDA.")
            labels = None

    # --- 3. Identify Outliers and Clusters ---
    noise_mask = (labels == -1)
    num_noise = np.sum(noise_mask)
    num_clustered = num_original_points - num_noise
    non_noise_labels = labels[~noise_mask]
    unique_cluster_labels = np.unique(non_noise_labels)
    num_clusters = len(unique_cluster_labels)

    print(f"DBSCAN found {num_clusters} cluster(s) and {num_noise} noise points (potential floaters).")

    # --- 4. Determine which points to keep ---
    keep_mask = np.zeros(num_original_points, dtype=bool) # Initialize keep_mask

    if args.keep_largest_cluster_only:
        if num_clusters > 0:
             cluster_sizes = np.bincount(non_noise_labels) # Sizes only for non-noise labels
             largest_cluster_label = unique_cluster_labels[np.argmax(cluster_sizes)] # Get label corresponding to max size
             keep_mask = (labels == largest_cluster_label)
             print(f"Keeping only the largest cluster (ID: {largest_cluster_label}) with {np.sum(keep_mask)} points.")
        else:
             print("Warning: No clusters found, only noise. Keeping nothing.")
             # keep_mask remains all False
    elif args.keep_cluster_ids is not None:
         # Validate provided IDs against actual cluster labels found
         valid_keep_ids = [l for l in args.keep_cluster_ids if l in unique_cluster_labels]
         invalid_keep_ids = [l for l in args.keep_cluster_ids if l not in unique_cluster_labels]
         if invalid_keep_ids:
             print(f"Warning: Specified cluster IDs {invalid_keep_ids} were not found by DBSCAN.")

         if valid_keep_ids:
              # Create mask for specified clusters
              for cluster_id in valid_keep_ids:
                  keep_mask |= (labels == cluster_id) # Use OR to add clusters
              print(f"Keeping specified cluster IDs {valid_keep_ids} resulting in {np.sum(keep_mask)} points.")
         else:
              print("Warning: None of the specified cluster IDs were valid. Keeping nothing.")
              # keep_mask remains all False
    else:
        # Default: Keep all non-noise points
        keep_mask = ~noise_mask
        print(f"Keeping all {num_clustered} points belonging to any cluster (removing noise).")


    num_kept_points = np.sum(keep_mask)

    # --- 5. Filter Data ---
    filtered_data = {}
    print("Filtering attributes...")
    start_filter = time.time()
    for key in tqdm(loaded_data.keys()):
        if loaded_data[key] is not None and loaded_data[key].shape[0] == num_original_points : # Basic check
            filtered_data[key] = loaded_data[key][keep_mask]
        else:
            # Handle cases where an attribute might be missing or malformed (though read_ply_data should handle most)
             print(f"Warning: Skipping attribute '{key}' during filtering due to potential issues.")
             filtered_data[key] = None # Or handle appropriately

    filter_time = time.time() - start_filter
    print(f"Attribute filtering completed in {filter_time:.2f} seconds.")

    # --- 6. Save Pruned PLY ---
    output_dir = os.path.dirname(args.output_ply)
    if output_dir and not os.path.exists(output_dir): # Ensure output directory exists
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving pruned PLY to: {args.output_ply}")
    start_save = time.time()
    try:
        save_ply_data(args.output_ply, filtered_data, args.max_sh_degree)
    except Exception as e:
        print(f"\nError during PLY saving: {e}")
        print("Please check file permissions and data consistency.")
        exit(1)
    save_time = time.time() - start_save
    print(f"PLY saving completed in {save_time:.2f} seconds.")

    print("-" * 30)
    print("Pruning Summary:")
    print(f"  Input Gaussians: {num_original_points}")
    print(f"  DBSCAN Time: {dbscan_time:.2f} seconds")
    print(f"  Identified as Noise: {num_noise}")
    print(f"  Number of Clusters Found: {num_clusters}")
    print(f"  Gaussians Kept: {num_kept_points}")
    print(f"  Pruned Gaussians Saved To: {args.output_ply}")
    print("-" * 30)