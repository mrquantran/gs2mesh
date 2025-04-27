# filter_colmap_model.py

import os
import numpy as np
from collections import defaultdict, namedtuple
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import shutil
import re # For parsing image names

# --- Assume read_write_model.py is in the same directory or accessible ---
try:
    from read_write_model import (
        read_model, write_model, Camera, Image, Point3D, qvec2rotmat,
        read_cameras_binary, read_images_binary, read_points3D_binary,
        write_cameras_text, write_images_text, write_points3D_text
    )
except ImportError:
    print("Error: Could not import read_write_model.py.")
    print("Ensure read_write_model.py is in the same directory or your PYTHONPATH.")
    sys.exit(1)
# ------------------------------------------------------------------------

# Updated PLY writer to include normal format (writing placeholders)
def write_points3D_ply(points3D, filename):
    """Writes a Point3D dictionary to a PLY file including normal placeholders."""
    if not points3D:
        print("Warning: No points to write to PLY file.")
        return

    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float error
property float nx
property float ny
property float nz
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(points3D)))
        # Sort points by ID for consistent output
        ordered_points = sorted(points3D.values(), key=lambda p: p.id)
        for point in ordered_points:
            # Clamp RGB values to valid range
            r = np.clip(point.rgb[0], 0, 255)
            g = np.clip(point.rgb[1], 0, 255)
            b = np.clip(point.rgb[2], 0, 255)
            f.write(f"{point.xyz[0]:.6f} {point.xyz[1]:.6f} {point.xyz[2]:.6f} ")
            f.write(f"{r} {g} {b} ")
            f.write(f"{point.error:.6f} ")
            # Write normal placeholders (0, 0, 0) as they aren't computed here
            f.write("0.0 0.0 0.0\n")
    print(f"Saved {len(points3D)} points (with normal placeholders) to {filename}")

def parse_image_index(image_name):
    """Extracts the integer index from image names like '0000.png', 'image_008.jpg' etc."""
    match = re.search(r'(\d+)', image_name)
    if match:
        return int(match.group(1))
    else:
        print(f"Warning: Could not parse index from image name: {image_name}")
        return None # Indicate failure


def filter_colmap_model(input_project_path, output_project_path, selected_views_file, input_npz_path=None):
    """
    Filters a COLMAP project based on a list of selected image names,
    creating a new self-contained project directory. Optionally processes cameras.npz.

    Args:
        input_project_path (str or Path): Path to the original COLMAP project directory
                                          (containing sparse/0, images, etc.).
        output_project_path (str or Path): Path to the directory where the filtered
                                           project will be saved.
        selected_views_file (str or Path): Path to the text file containing the names
                                          of the selected images, one per line.
        input_npz_path (str or Path, optional): Path to the input cameras.npz file.
                                                Defaults to None.
    """
    input_project_path = Path(input_project_path)
    output_project_path = Path(output_project_path)
    selected_views_file = Path(selected_views_file)
    if input_npz_path:
        input_npz_path = Path(input_npz_path)


    # --- Define Paths ---
    input_sparse_dir = input_project_path / "sparse" / "0"
    input_images_dir = input_project_path / "images" # Assuming standard 'images' dir

    output_sparse_dir = output_project_path / "sparse" / "0"
    output_images_dir = output_project_path / "images"
    output_npz_file = output_project_path / "cameras.npz" # Define output NPZ path


    # --- Check Inputs ---
    # (Input sparse model check remains the same)
    if not input_sparse_dir.exists() or \
       not (input_sparse_dir / "cameras.bin").exists() or \
       not (input_sparse_dir / "images.bin").exists() or \
       not (input_sparse_dir / "points3D.bin").exists():
        print(f"Error: Input sparse model not found or incomplete in {input_sparse_dir}")
        sys.exit(1)

    if not input_images_dir.exists() or not input_images_dir.is_dir():
         print(f"Warning: Input images directory '{input_images_dir}' not found. Cannot copy images.")

    if not selected_views_file.exists():
        print(f"Error: Selected views file not found: {selected_views_file}")
        sys.exit(1)

    if input_npz_path and not input_npz_path.exists():
        print(f"Error: Input cameras.npz file specified but not found: {input_npz_path}")
        sys.exit(1)

    # --- Create Output Directories ---
    print(f"Creating output directories...")
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True) # Create images dir too

    print(f"Reading original model from: {input_sparse_dir}")
    try:
        cameras, images, points3D = read_model(path=str(input_sparse_dir), ext=".bin")
        print(f"Original model loaded:")
        print(f"  Cameras: {len(cameras)}")
        print(f"  Images: {len(images)}")
        print(f"  Points3D: {len(points3D)}")
    except Exception as e:
        print(f"Error reading model: {e}")
        sys.exit(1)

    print(f"Reading selected view names from: {selected_views_file}")
    with open(selected_views_file, 'r') as f:
        selected_image_names = {line.strip() for line in f if line.strip()}
    print(f"Found {len(selected_image_names)} selected view names.")
    if not selected_image_names:
        print("Error: No selected views found in the file.")
        sys.exit(1)

    # --- Load NPZ data if provided ---
    original_npz_data = None
    if input_npz_path:
        print(f"Loading camera data from: {input_npz_path}")
        try:
            original_npz_data = np.load(input_npz_path)
            print(f"  Loaded {len(original_npz_data.files)} items from npz.")
        except Exception as e:
            print(f"Error loading {input_npz_path}: {e}")
            original_npz_data = None # Proceed without npz


    # --- 1. Filter Images and Identify Used Cameras/IDs ---
    print("Filtering images...")
    filtered_images_dict = {}
    selected_image_ids = set()
    used_camera_ids = set()

    for image_id, img in tqdm(images.items(), desc="Processing Images"):
        if img.name in selected_image_names:
            filtered_images_dict[image_id] = img
            selected_image_ids.add(image_id)
            used_camera_ids.add(img.camera_id)

    print(f"Kept {len(filtered_images_dict)} images out of {len(images)}.")
    if len(filtered_images_dict) != len(selected_image_names):
        print("Warning: Some selected image names were not found in the original model's images.bin!")
        found_names = {img.name for img in filtered_images_dict.values()}
        missing_names = selected_image_names - found_names
        print(f"Missing names: {missing_names}")

    if not filtered_images_dict:
        print("Error: No images kept after filtering. Check image names in the list file.")
        sys.exit(1)

    # --- 2. Filter Cameras ---
    print("Filtering cameras...")
    filtered_cameras = {cam_id: cameras[cam_id] for cam_id in used_camera_ids if cam_id in cameras}
    print(f"Kept {len(filtered_cameras)} cameras out of {len(cameras)}.")
    if not filtered_cameras:
        print("Error: No cameras kept. This shouldn't happen if images were kept.")
        sys.exit(1)

    # --- 3. Filter Points3D ---
    print("Filtering Points3D...")
    filtered_points3D = {}
    kept_point_ids = set()

    for point_id, point in tqdm(points3D.items(), desc="Processing Points3D"):
        observing_selected_image_ids = []
        original_indices_in_track = []
        for i, img_id in enumerate(point.image_ids):
            if img_id in selected_image_ids:
                observing_selected_image_ids.append(img_id)
                original_indices_in_track.append(i)

        if len(observing_selected_image_ids) >= 2:
            kept_point_ids.add(point_id)
            new_image_ids = point.image_ids[original_indices_in_track]
            new_point2D_idxs = point.point2D_idxs[original_indices_in_track]
            filtered_points3D[point_id] = Point3D(id=point.id, xyz=point.xyz, rgb=point.rgb,
                                                  error=point.error, image_ids=new_image_ids,
                                                  point2D_idxs=new_point2D_idxs)

    print(f"Kept {len(filtered_points3D)} Points3D out of {len(points3D)} (observed by >= 2 selected views).")
    if not filtered_points3D:
        print("Warning: No 3D points kept. The selected views might have poor overlap or the threshold (>=2) is too high.")

    # --- 4. Update Image Point References & Prepare for NPZ ---
    print("Updating image point references...")
    final_filtered_images = {}
    image_id_to_original_index = {} # Map filtered image_id to original parsed index for NPZ
    ordered_final_image_ids = sorted(filtered_images_dict.keys()) # Ensure consistent order

    for image_id in tqdm(ordered_final_image_ids, desc="Updating Images"):
        original_image = filtered_images_dict[image_id]
        if not hasattr(original_image, 'point3D_ids'):
             print(f"Warning: Image {image_id} missing 'point3D_ids'. Skipping.")
             continue

        # --- Parse original index for NPZ mapping ---
        original_index = parse_image_index(original_image.name)
        if original_index is not None:
            image_id_to_original_index[image_id] = original_index
        # -------------------------------------------

        valid_point_mask = np.isin(original_image.point3D_ids, list(kept_point_ids))
        valid_point_mask &= (original_image.point3D_ids != -1)

        new_point3D_ids = original_image.point3D_ids[valid_point_mask]
        new_xys = original_image.xys[valid_point_mask]

        final_filtered_images[image_id] = Image(id=original_image.id, qvec=original_image.qvec,
                                                tvec=original_image.tvec, camera_id=original_image.camera_id,
                                                name=original_image.name, xys=new_xys,
                                                point3D_ids=new_point3D_ids)

    print(f"Updated point references in {len(final_filtered_images)} images.")


    # --- 5. Write Filtered Binary Model ---
    print(f"Writing filtered binary model to: {output_sparse_dir}")
    try:
        write_model(filtered_cameras, final_filtered_images, filtered_points3D,
                    path=str(output_sparse_dir), ext=".bin")
    except Exception as e:
        print(f"Error writing filtered binary model: {e}")
        sys.exit(1)

    # --- 6. Write Filtered Text Model ---
    print(f"Writing filtered text model to: {output_sparse_dir}")
    try:
        write_model(filtered_cameras, final_filtered_images, filtered_points3D,
                    path=str(output_sparse_dir), ext=".txt")
    except Exception as e:
        print(f"Error writing filtered text model: {e}")

    # --- 7. Generate Filtered PLY ---
    ply_filename = output_sparse_dir / "points3D.ply"
    print(f"Writing filtered PLY file (with normal placeholders) to: {ply_filename}")
    try:
        write_points3D_ply(filtered_points3D, ply_filename)
    except Exception as e:
        print(f"Error writing PLY file: {e}")

    # --- 8. Copy Selected Image Files ---
    print(f"Copying selected image files to: {output_images_dir}")
    copied_count = 0
    skipped_count = 0
    if not input_images_dir.exists():
         print(f"Skipping image copy because input directory '{input_images_dir}' does not exist.")
    else:
        for img in tqdm(final_filtered_images.values(), desc="Copying Images"):
            src_path = input_images_dir / img.name
            dest_path = output_images_dir / img.name
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dest_path) # copy2 preserves metadata
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {src_path} to {dest_path}: {e}")
                    skipped_count += 1
            else:
                print(f"Warning: Source image file not found, cannot copy: {src_path}")
                skipped_count += 1
        print(f"Copied {copied_count} image files. Skipped/Errors: {skipped_count}.")

    # --- 8b. Copy Selected Mask Files ---
    # Assumes masks are in 'masks' directory under input_project_path, with same filenames as images
    input_masks_dir = input_project_path / "mask"
    output_masks_dir = output_project_path / "mask"
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    copied_mask_count = 0
    skipped_mask_count = 0
    if not input_masks_dir.exists():
        print(f"Warning: Input masks directory '{input_masks_dir}' not found. Skipping mask copy.")
    else:
        for img in tqdm(final_filtered_images.values(), desc="Copying Masks"):
            mask_name = img.name
            # split the name to get the mask name
            base = os.path.splitext(mask_name)[0]
            # Extract number, pad to 3 digits, keep prefix if any
            match = re.match(r'(.*?)(\d+)$', base)
            if match:
                prefix, num = match.groups()
                mask_name = f"{prefix}{int(num):03d}.png"
            else:
                mask_name = base + ".png"
            src_mask_path = input_masks_dir / mask_name
            dest_mask_path = output_masks_dir / mask_name
            if src_mask_path.exists():
                try:
                    shutil.copy2(src_mask_path, dest_mask_path)
                    copied_mask_count += 1
                except Exception as e:
                    print(f"Error copying mask {src_mask_path} to {dest_mask_path}: {e}")
                    skipped_mask_count += 1
            else:
                print(f"Warning: Source mask file not found, cannot copy: {src_mask_path}")
                skipped_mask_count += 1
        print(f"Copied {copied_mask_count} mask files. Skipped/Errors: {skipped_mask_count}.")

    # --- 9. Create Filtered cameras.npz ---
    if original_npz_data is not None:
        print(f"Creating filtered cameras.npz file at: {output_npz_file}")
        filtered_npz_data = {}
        missing_matrices = 0
        # Iterate through the *final sorted* list of selected image IDs
        for new_idx, image_id in enumerate(ordered_final_image_ids):
            if image_id not in image_id_to_original_index:
                print(f"Warning: Could not parse original index for image ID {image_id} (name: {final_filtered_images[image_id].name}). Skipping for NPZ.")
                missing_matrices += 1
                continue

            original_idx = image_id_to_original_index[image_id]
            world_mat_key = f'world_mat_{original_idx}'
            scale_mat_key = f'scale_mat_{original_idx}'

            if world_mat_key in original_npz_data and scale_mat_key in original_npz_data:
                # Add to new dict with re-indexed keys
                filtered_npz_data[f'world_mat_{new_idx}'] = original_npz_data[world_mat_key]
                filtered_npz_data[f'scale_mat_{new_idx}'] = original_npz_data[scale_mat_key]
            else:
                print(f"Warning: Matrices '{world_mat_key}' or '{scale_mat_key}' not found in input npz for image ID {image_id} (name: {final_filtered_images[image_id].name}). Skipping for NPZ.")
                missing_matrices += 1

        if filtered_npz_data:
            try:
                np.savez(output_npz_file, **filtered_npz_data)
                print(f"Saved {len(filtered_npz_data) // 2} camera entries to {output_npz_file}.")
                if missing_matrices > 0:
                     print(f"  ({missing_matrices} entries could not be processed due to missing indices or matrices)")
            except Exception as e:
                print(f"Error saving filtered cameras.npz: {e}")
        else:
             print("Warning: No data available to save for filtered cameras.npz.")


# =============================================================================
#  Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a COLMAP sparse model based on a list of selected image names, creating a new project structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input_model", "-i", required=True, type=str,
                        help="Path to the root input COLMAP project directory (containing sparse/0, images).")
    parser.add_argument("--output_model", "-o", required=True, type=str,
                        help="Path to the root output directory where the filtered project will be saved.")
    parser.add_argument("--selected_views", "-s", required=True, type=str,
                        help="Path to the text file containing selected image names (one per line).")

    args = parser.parse_args()

    # inputnpz is
    input_npz = Path(args.input_model) / "cameras.npz"

    filter_colmap_model(
        input_project_path=Path(args.input_model),
        output_project_path=Path(args.output_model),
        selected_views_file=Path(args.selected_views),
        input_npz_path=input_npz # Pass the NPZ path
    )

    print("\nFiltering complete.")
    print(f"Filtered project saved to: {args.output_model}")