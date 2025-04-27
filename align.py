# align_mesh.py (Revised)
import open3d as o3d
import numpy as np
import argparse
import copy
import os

def get_average_extent(pcd):
    """Calculates the average extent of the point cloud's AABB."""
    try:
        aabb = pcd.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        if np.any(extent <= 0):
            return 1.0 # Avoid division by zero or negative extents
        return np.mean(extent)
    except Exception:
        # Handle cases where bounding box calculation fails (e.g., empty pcd)
        return 1.0


def align_mesh_to_gt(gs_mesh_path, gt_pcd_path, output_aligned_mesh_path,
                     max_correspondence_dist=0.1, # Keep this tunable!
                     max_iterations=200, verbose=True):
    """
    Aligns a source mesh to a target point cloud using ICP with scale estimation,
    using bounding box extents for better initial scale guess.

    Args:
        gs_mesh_path (str): Path to the Gaussian Splatting mesh (.ply).
        gt_pcd_path (str): Path to the Ground Truth point cloud (.ply, e.g., stlXXX_total.ply).
        output_aligned_mesh_path (str): Path to save the aligned mesh (.ply).
        max_correspondence_dist (float): Max distance for ICP correspondences. Adjust based on scene scale.
        max_iterations (int): Maximum ICP iterations.
        verbose (bool): Print progress and results.
    """
    if verbose:
        print(f"Loading source mesh from: {gs_mesh_path}")
    try:
        gs_mesh = o3d.io.read_triangle_mesh(gs_mesh_path)
        if not gs_mesh.has_vertices():
            print(f"ERROR: Source mesh '{gs_mesh_path}' is empty.")
            return False
    except Exception as e:
        print(f"ERROR: Failed to load source mesh '{gs_mesh_path}': {e}")
        return False

    if verbose:
        print(f"Loading target ground truth point cloud from: {gt_pcd_path}")
    try:
        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
        if not gt_pcd.has_points():
            print(f"ERROR: Target point cloud '{gt_pcd_path}' is empty.")
            return False
    except Exception as e:
        print(f"ERROR: Failed to load target point cloud '{gt_pcd_path}': {e}")
        return False

    # Use vertices of the source mesh as the source point cloud for alignment
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = gs_mesh.vertices
    if len(source_pcd.points) == 0:
        print(f"ERROR: Extracted 0 points from source mesh '{gs_mesh_path}'. Cannot align.")
        return False

    if verbose:
        print(f"Source mesh vertices: {len(source_pcd.points)}")
        print(f"Target GT points: {len(gt_pcd.points)}")

    # --- Improved Initial Alignment ---
    center_source = source_pcd.get_center()
    center_target = gt_pcd.get_center()

    # Estimate initial scale based on average bounding box extent
    avg_extent_source = get_average_extent(source_pcd)
    avg_extent_target = get_average_extent(gt_pcd)

    if avg_extent_source < 1e-6:
        print("ERROR: Source point cloud has zero or near-zero extent. Cannot estimate scale.")
        return False

    init_scale = avg_extent_target / avg_extent_source

    if verbose:
        print(f"Estimated initial scale (Target Extent / Source Extent): {init_scale:.4f}")

    # Apply initial scaling *before* calculating translation needed for centering
    initial_translation = center_target - center_source * init_scale

    # Create initial transformation matrix: Scale -> Translate
    # (Rotation is identity initially, ICP will find it)
    init_transform = np.identity(4)
    init_transform[:3, :3] = np.identity(3) * init_scale
    init_transform[:3, 3] = initial_translation

    if verbose:
        print("Calculated initial transformation (Centering + Estimated Scale):")
        print(init_transform)

    # --- ICP Registration with Scale Estimation ---
    if verbose:
        print(f"Performing ICP registration (max_correspondence_dist={max_correspondence_dist}, max_iter={max_iterations})...")

    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7, # Slightly tighter criteria
        relative_rmse=1e-7,
        max_iteration=max_iterations
    )

    try:
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            gt_pcd,
            max_correspondence_dist,
            init_transform,
            estimation_method,
            criteria
        )
    except Exception as e:
        print(f"\nERROR during ICP: {e}")
        print("ICP failed. This might be due to:")
        print(f"  - Poor initial alignment (check estimated scale: {init_scale:.4f})")
        print(f"  - Incorrect max_correspondence_dist ({max_correspondence_dist}). Try increasing/decreasing.")
        print("  - Very different point cloud structures.")
        return False


    # --- Apply Result ---
    transformation = reg_result.transformation
    if verbose:
        print("ICP registration completed.")
        print(f"  Fitness: {reg_result.fitness:.4f}") # Higher is better (fraction of points that have correspondences)
        print(f"  Inlier RMSE: {reg_result.inlier_rmse:.4f}") # Lower is better (error for corresponding points)
        print("  Resulting transformation (Source -> Target):")
        print(transformation)

    # Check if ICP seems reasonable
    # Fitness < 0.5 or RMSE > max_correspondence_dist often indicates poor alignment
    if reg_result.fitness < 0.5 or reg_result.inlier_rmse > max_correspondence_dist * 1.5 :
         print("\nWARNING: ICP results indicate potentially poor alignment.")
         print("  Low fitness suggests poor overlap or wrong scale/orientation.")
         print("  High RMSE suggests correspondences are far apart.")
         print(f"  Consider adjusting --max_dist (currently {max_correspondence_dist}). Try a larger value if fitness is low, maybe smaller if RMSE is high but fitness reasonable.")
         print("  Also double-check input files and paths.")
         # Decide if you want to proceed despite the warning
         # proceed = input("Proceed saving aligned mesh despite warning? (y/N): ")
         # if proceed.lower() != 'y':
         #     return False

    # Apply the final transformation to the original source mesh
    gs_mesh_aligned = copy.deepcopy(gs_mesh)
    gs_mesh_aligned.transform(transformation)

    # --- Save Aligned Mesh ---
    if verbose:
        print(f"\nSaving aligned mesh to: {output_aligned_mesh_path}")
    try:
        os.makedirs(os.path.dirname(output_aligned_mesh_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_aligned_mesh_path, gs_mesh_aligned)
        print("Aligned mesh saved successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save aligned mesh '{output_aligned_mesh_path}': {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align a source mesh to a target ground truth point cloud using ICP.")
    parser.add_argument("--gs_mesh", type=str, required=True, help="Path to the input mesh generated by Gaussian Splatting.")
    parser.add_argument("--gt_pcd", type=str, required=True, help="Path to the ground truth point cloud (e.g., DTU's stlXXX_total.ply).")
    parser.add_argument("--output_mesh", type=str, required=True, help="Path to save the aligned output mesh.")
    # --- IMPORTANT PARAMETER ---
    parser.add_argument("--max_dist", type=float, default=0.2, help="Maximum correspondence distance for ICP (!! TUNE THIS !! based on scene scale/units and initial error, e.g., 0.1, 0.05, 0.5).")
    # --- Max Iterations ---
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum ICP iterations.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")

    args = parser.parse_args()

    align_mesh_to_gt(
        gs_mesh_path=args.gs_mesh,
        gt_pcd_path=args.gt_pcd,
        output_aligned_mesh_path=args.output_mesh,
        max_correspondence_dist=args.max_dist,
        max_iterations=args.max_iter,
        verbose=not args.quiet
    )