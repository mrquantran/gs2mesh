# File: select_views_graph_beamsearch.py

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict, namedtuple
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
import heapq
import cv2
import random
import pygad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # Often pygad uses plt indirectly
# --------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
gs_path = os.path.join(script_dir, 'third_party', 'gaussian-splatting')
if gs_path not in sys.path:
    sys.path.append(gs_path)
# ---------------------------------------------

try:
    from scene.colmap_loader import (
        read_extrinsics_binary, qvec2rotmat,
        read_next_bytes
    )
    Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

except ImportError as e:
    print(f"Error importing from gaussian-splatting: {e}")
    print("Please ensure the script can find the 'third_party/gaussian-splatting' directory.")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)

def read_points3D_binary_with_data(path_to_model_file):
    """Reads points3D.bin, returns dict mapping ID to Point3D tuple."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id, xyz, rgb, error = props[0], np.array(props[1:4]), np.array(props[4:7]), props[7]
            track_len = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track = read_next_bytes(fid, num_bytes=8*track_len, format_char_sequence="ii"*track_len)
            img_ids = np.array(tuple(map(int, track[0::2])))
            p2D_idxs = np.array(tuple(map(int, track[1::2])))
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=img_ids, point2D_idxs=p2D_idxs)
    return points3D

def calculate_geodesic_distance(pose1, pose2, w_translation=0.1):
    """Calculates the geodesic distance between two camera poses."""
    R1 = qvec2rotmat(pose1.qvec)
    R2 = qvec2rotmat(pose2.qvec)
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    # Handle potential numerical inaccuracies leading to trace > 3.0 or < -1.0
    trace = max(-1.0, min(3.0, trace))
    rotation_angle = np.arccos((trace - 1.0) / 2.0)
    translation_distance = np.linalg.norm(pose1.tvec - pose2.tvec)
    return rotation_angle + w_translation * translation_distance

def reprojection_error_weight(error, factor=1.0, epsilon=1e-6):
    """ Calculates weight based on reprojection error. Lower error -> higher weight. """
    weight = np.exp(-factor * error**2)
    return np.clip(weight, 0.0, 1.0) # Ensure weight is between 0 and 1

def calculate_weighted_matchability_matrix(images, points3D_data, error_weight_factor=1.0):
    """
    Calculates a matrix where M[i, j] is the sum of weights of 3D points
    observed by both image i and image j. Weight is based on reprojection error.
    """
    num_images = len(images)
    if num_images == 0: return np.array([]), [], {}
    image_ids_sorted = sorted(images.keys())
    id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids_sorted)}
    match_matrix = np.zeros((num_images, num_images), dtype=float)

    point3D_observers = defaultdict(list)
    valid_point3D_ids = set(points3D_data.keys())

    # Map point3D IDs to the images that observe them
    for img_id, img_data in images.items():
        if not hasattr(img_data, 'point3D_ids'): continue # Skip if image has no points info
        valid_obs_mask = (img_data.point3D_ids != -1) & np.isin(img_data.point3D_ids, list(valid_point3D_ids))
        observed_valid_ids = img_data.point3D_ids[valid_obs_mask]
        for pt3D_id in observed_valid_ids:
             if pt3D_id in valid_point3D_ids: # Double check point exists
                point3D_observers[pt3D_id].append(img_id)

    # Fill the matrix based on shared observations, weighted by point error
    for pt3D_id, observer_ids in point3D_observers.items():
        if pt3D_id not in points3D_data: continue
        point_error = points3D_data[pt3D_id].error
        weight = reprojection_error_weight(point_error, error_weight_factor)

        # Get indices for observers present in our sorted list
        observer_indices = [id_to_idx[img_id] for img_id in observer_ids if img_id in id_to_idx]

        # Add weight to pairwise entries in the matrix
        for i in range(len(observer_indices)):
            for j in range(i + 1, len(observer_indices)):
                idx1, idx2 = observer_indices[i], observer_indices[j]
                match_matrix[idx1, idx2] += weight
                match_matrix[idx2, idx1] += weight # Symmetric matrix

    # Normalize the matrix to [0, 1] range for consistent scoring
    max_weighted_matches = np.max(match_matrix) if num_images > 1 else 0
    match_matrix = match_matrix / max_weighted_matches

    print(f"Calculated weighted matchability matrix. Max summed weight (normalized): {np.max(match_matrix):.4f}")
    return match_matrix, image_ids_sorted, id_to_idx


def calculate_pose_distance_matrix(images, image_ids_sorted, w_translation):
    """Calculates the pairwise geodesic distance matrix."""
    num_images = len(image_ids_sorted)
    if num_images == 0: return np.array([])
    distance_matrix = np.zeros((num_images, num_images), dtype=float)

    for i in tqdm(range(num_images), desc="Calculating Pose Distances"):
        for j in range(i + 1, num_images):
            img_id1, img_id2 = image_ids_sorted[i], image_ids_sorted[j]
            if img_id1 in images and img_id2 in images:
                dist = calculate_geodesic_distance(images[img_id1], images[img_id2], w_translation)
                distance_matrix[i, j] = distance_matrix[j, i] = dist

            # save to txt
            with open("pose_distances.txt", "a") as f:
                f.write(f"{img_id1} {img_id2} {distance_matrix[i, j]:.4f}\n")

    # Normalize the distance matrix to [0, 1] range
    max_dist = np.max(distance_matrix) if num_images > 1 else 0.0
    distance_matrix = distance_matrix / max_dist

    print(f"Calculated pose distance matrix. Max distance (normalized): {np.max(distance_matrix):.4f}")
    return distance_matrix

def calculate_image_sharpness(image_path):
    """Calculates sharpness score (variance of Laplacian) for an image."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        # ksize=3 is common for Laplacian

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F, ksize=3).var()
        # Handle cases where variance might be extremely small or negative (unlikely but possible)
        return max(0.0, laplacian_var)
    except Exception as e:
        print(f"Error processing {image_path} for sharpness: {e}. Returning 0.")
        return 0.0

def calculate_all_image_qualities(images_data, image_ids_sorted, images_dir):
    """Calculates sharpness for all images and normalizes."""
    num_images = len(image_ids_sorted)
    if num_images == 0: return np.array([])

    quality_scores = np.zeros(num_images, dtype=float)
    id_to_name = {img_id: img.name for img_id, img in images_data.items()}

    print(f"Calculating image sharpness scores (using Laplacian variance)...")
    not_found_count = 0
    for i, img_id in enumerate(tqdm(image_ids_sorted, desc="Calculating Sharpness")):
        if img_id not in id_to_name:
             print(f"Warning: Image ID {img_id} not found in images_data.")
             continue
        img_name = id_to_name[img_id]
        img_path = images_dir / img_name
        quality_scores[i] = calculate_image_sharpness(img_path)

    if not_found_count > 0:
        print(f"Warning: {not_found_count}/{num_images} image files were not found in {images_dir}.")

    # Normalize scores to [0, 1]
    max_score = np.max(quality_scores)
    quality_scores /= max_score

    print(f"Calculated and normalized sharpness scores. Max raw score: {max_score:.2f}")
    return quality_scores

def calculate_subset_score(subset_indices,
                           pose_distance_matrix,
                           weighted_matchability_matrix,
                           image_quality_scores,
                           w_cov, w_match, w_qual):
    """Calculates the combined score for a subset including image quality."""
    if len(subset_indices) == 0: return 0.0, 0.0, 0.0, 0.0 # Score, Cov, Match, Qual

    subset_size = len(subset_indices)
    subset_indices_arr = np.array(subset_indices)

    # Handle single view case: Coverage/Matchability are 0
    if subset_size == 1:
        idx = subset_indices_arr[0]
        if idx < len(image_quality_scores):
            quality_score = image_quality_scores[idx]
        combined_score = w_qual * quality_score
        return combined_score, 0.0, 0.0, quality_score

    # --- Calculations for subset_size > 1 ---
    num_pairs = subset_size * (subset_size - 1) / 2.0

    # Coverage Score: Average pairwise geodesic distance
    sub_dist_matrix = pose_distance_matrix[np.ix_(subset_indices_arr, subset_indices_arr)]
    coverage_score = np.sum(np.triu(sub_dist_matrix, k=1)) / num_pairs

    # Matchability Score: Average pairwise WEIGHTED matchability
    sub_match_matrix = weighted_matchability_matrix[np.ix_(subset_indices_arr, subset_indices_arr)]
    matchability_score = np.sum(np.triu(sub_match_matrix, k=1)) / num_pairs

    # Quality Score: Average quality score of views in the subset
    subset_qualities = image_quality_scores[subset_indices_arr]
    quality_score = np.mean(subset_qualities)

    # Combined Score
    combined_score = (w_cov * coverage_score +
                      w_match * matchability_score +
                      w_qual * quality_score)

    return combined_score, coverage_score, matchability_score, quality_score

fitness_data = {}

def fitness_func_wrapper(ga_instance, solution, solution_idx):
    """
    Wrapper function to calculate fitness for PyGAD.
    Uses the pre-calculated matrices and scores stored in the global 'fitness_data'.
    """
    global fitness_data
    # The 'solution' is a numpy array of selected view indices (genes)
    subset_indices = solution.astype(int) # Ensure indices are integers

    # Retrieve necessary data
    pose_distance_matrix = fitness_data['pose_distance_matrix']
    weighted_matchability_matrix = fitness_data['weighted_matchability_matrix']
    image_quality_scores = fitness_data['image_quality_scores']
    w_cov = fitness_data['w_cov']
    w_match = fitness_data['w_match']
    w_qual = fitness_data['w_qual']

    # Calculate the score using the existing function
    # calculate_subset_score returns: score, cov, match, qual
    combined_score, _, _, _ = calculate_subset_score(
        subset_indices,
        pose_distance_matrix,
        weighted_matchability_matrix,
        image_quality_scores,
        w_cov, w_match, w_qual
    )

    # Handle potential NaN/inf scores (though calculate_subset_score should be robust now)
    if not np.isfinite(combined_score):
        # print(f"Warning: Non-finite score ({combined_score}) for solution {subset_indices}. Returning low fitness.")
        return -1e9 # Return a very low fitness for invalid solutions

    return combined_score


def ga_view_selection(num_views_to_select, # M
                      num_total_views,     # N
                      pose_distance_matrix,
                      weighted_matchability_matrix,
                      image_quality_scores,
                      image_ids_sorted, # Needed for potential logging/debug inside GA
                      w_cov, w_match, w_qual,
                      # --- GA Specific Parameters ---
                      num_generations=100,
                      population_size=50,
                      mutation_percent=10,
                      parent_selection_type="tournament", # tournament vs. roulette wheel vs. sss
                      crossover_type="scattered", # Good for combinatorial problems
                      mutation_type="random", # inversion vs swap vs random vs scramble
                      keep_elitism=2 # Keep the best 2 individuals
                      ):
    """
    Performs view selection using a Genetic Algorithm (PyGAD).
    """
    global fitness_data
    # Store data needed by the fitness function in the global dict
    fitness_data = {
        'pose_distance_matrix': pose_distance_matrix,
        'weighted_matchability_matrix': weighted_matchability_matrix,
        'image_quality_scores': image_quality_scores,
        'w_cov': w_cov,
        'w_match': w_match,
        'w_qual': w_qual
    }

    print(f"\nStarting Genetic Algorithm Selection...")
    print(f"  Num Views to Select (M): {num_views_to_select}")
    print(f"  Total Views Available (N): {num_total_views}")
    print(f"  Generations: {num_generations}, Population Size: {population_size}")
    print(f"  Mutation %: {mutation_percent}, Keep Elitism: {keep_elitism}")
    print(f"  Selection: {parent_selection_type}, Crossover: {crossover_type}, Mutation: {mutation_type}")

    # --- Configure PyGAD Instance ---
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 2, # Number of parents to mate
        fitness_func=fitness_func_wrapper,
        sol_per_pop=population_size,
        num_genes=num_views_to_select, # Each chromosome has M genes (indices)

        gene_space=list(range(num_total_views)), # Possible values for each gene (0 to N-1)
        gene_type=int, # Ensure genes are integers

        parent_selection_type=parent_selection_type,
        keep_elitism=keep_elitism, # Keep the N best parents in the next gen
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent,
        K_tournament=5,
        allow_duplicate_genes=False, # CRITICAL: Ensure selected indices are unique

        # Optional: Stop criteria (e.g., stop if no improvement for X generations)
        # stop_criteria=["saturate_10"] # Stop after 10 generations with no improvement

        # Optional: Parallel processing (if useful and dependencies installed)
        # parallel_processing=['thread', 4] # Use 4 threads

        # Optional: Suppress internal PyGAD logging if too verbose
        # logger=None
    )

    # --- Run the GA ---
    # Use tqdm for progress bar visualization
    print("Running GA generations...")
    ga_instance.run()

    # --- Get Results ---
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

    print(f"\nGenetic Algorithm finished after {ga_instance.generations_completed} generations.")
    print(f"Best solution fitness: {solution_fitness:.4f}")

    # The 'solution' is the numpy array of selected indices
    best_indices = sorted(solution.astype(int).tolist()) # Convert to sorted list of ints

    # Optional: Plot progress
    try:
        ga_instance.plot_fitness(title="GA Fitness Progress", save_dir="ga_fitness_plot.png")
    except Exception as e:
        print(f"Note: Could not generate GA fitness plot ({e}).")

    # Recalculate final scores for display (optional, but good practice)
    final_score, final_cov, final_match, final_qual = calculate_subset_score(
        best_indices, pose_distance_matrix, weighted_matchability_matrix, image_quality_scores, w_cov, w_match, w_qual
    )
    print(f"Final Best Score (recalculated): {final_score:.4f} (Cov: {final_cov:.3f}, Match: {final_match:.3f}, Qual: {final_qual:.3f})")

    # Clean up global data if desired (optional)
    fitness_data = {}

    return best_indices, final_score

# =============================================================================
#  Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select M views using Beam Search on View Graph Score (including Image Quality).")
    # --- Existing Args ---
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to the COLMAP reconstruction directory")
    parser.add_argument("--num_views", "-m", type=int, default=9, help="Number of views (M) to select")
    parser.add_argument("--w_coverage", type=float, default=1.0, help="Weight for coverage term")
    parser.add_argument("--w_matchability", type=float, default=0.1, help="Weight for matchability term")
    parser.add_argument("--w_quality", type=float, default=0.5, help="Weight for image quality term")
    parser.add_argument("--error_weight_factor", "-ewf", type=float, default=1.0, help="Factor 'k' for reprojection error weighting")
    parser.add_argument("--w_translation", type=float, default=0.1, help="Weight for translation in geodesic distance")
    parser.add_argument("--output_file", "-o", type=str, default="selected_views_ga.txt", help="File to save selected view names (default uses GA suffix)")
    parser.add_argument("--colmap_subdir", type=str, default="sparse/0", help="Subdirectory for COLMAP model files")
    parser.add_argument("--images_subdir", type=str, default="images", help="Subdirectory containing image files")

    # --- Selection Method Arg ---
    parser.add_argument("--method", type=str, choices=['ga', 'beam'], default='ga', help="Selection method: 'ga' (Genetic Algorithm) or 'beam' (Beam Search)")

    # --- GA Specific Args ---
    parser.add_argument("--generations", "-g", type=int, default=100, help="Number of generations for GA (used only if method='ga')")
    parser.add_argument("--population", "-p", type=int, default=50, help="Population size for GA (used only if method='ga')")
    parser.add_argument("--mutation", "-mut", type=float, default=10.0, help="Mutation percentage for GA (used only if method='ga')")

    args = parser.parse_args()

    source_path = Path(args.source_path)
    colmap_sparse_path = source_path / args.colmap_subdir
    images_path = source_path / args.images_subdir # <-- Define images path

    if not colmap_sparse_path.exists():
        print(f"Error: COLMAP sparse directory not found: {colmap_sparse_path}")
        sys.exit(1)
    if not images_path.exists():
        print(f"Error: Images directory not found: {images_path}. Needed for quality scoring.")
        sys.exit(1)
    if not (colmap_sparse_path / "images.bin").exists():
         print(f"Error: images.bin not found in {colmap_sparse_path}")
         sys.exit(1)
    if not (colmap_sparse_path / "points3D.bin").exists():
         print(f"Error: points3D.bin not found in {colmap_sparse_path}")
         sys.exit(1)


    print("Loading COLMAP data...")
    points3D_data = None
    try:
        # Use read_extrinsics_binary as it returns the necessary Image objects with 'name'
        images = read_extrinsics_binary(colmap_sparse_path / "images.bin")
        # images = read_images_binary(colmap_sparse_path / "images.bin") # Alternative if read_extrinsics causes issues
        points3D_data = read_points3D_binary_with_data(colmap_sparse_path / "points3D.bin")
        print(f"Loaded {len(points3D_data)} 3D points from binary.")
    except FileNotFoundError as e:
         print(f"Error loading COLMAP data: {e}. Check paths.")
         sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading COLMAP data: {e}")
         # Consider more specific error handling if needed
         sys.exit(1)


    print(f"Loaded {len(images)} image poses/data.")
    if not images: print("Error: No images loaded."); sys.exit(1)
    if not points3D_data: print("Warning: No 3D points loaded. Matchability score will be zero."); points3D_data = {}

    # --- Calculate Matrices ---
    print("Calculating Weighted Matchability Matrix...")
    weighted_matchability_matrix, image_ids_sorted, id_to_idx = calculate_weighted_matchability_matrix(
        images, points3D_data, args.error_weight_factor
    )
    num_total_views = len(image_ids_sorted) # Get N

    print("Calculating Pose Distance Matrix...")
    pose_distance_matrix = calculate_pose_distance_matrix(images, image_ids_sorted, args.w_translation)

    print("Calculating Image Quality Scores...")
    image_quality_scores = calculate_all_image_qualities(images, image_ids_sorted, images_path)

    # --- Perform Selection ---
    selected_indices = []
    best_score = float('-inf')
    best_indices = []
    n_runs = 5

    if args.method == 'ga':
        for run in range(n_runs):
            print(f"\n=== GA Run {run+1}/{n_runs} ===")
            indices, final_score = ga_view_selection(
                num_views_to_select=args.num_views,
                num_total_views=num_total_views,
                pose_distance_matrix=pose_distance_matrix,
                weighted_matchability_matrix=weighted_matchability_matrix,
                image_quality_scores=image_quality_scores,
                image_ids_sorted=image_ids_sorted,
                w_cov=args.w_coverage,
                w_match=args.w_matchability,
                w_qual=args.w_quality,
                num_generations=args.generations,
                population_size=args.population,
                mutation_percent=args.mutation,
            )
            print(f"GA run {run+1} final score: {final_score:.4f}")
            if final_score > best_score:
                best_score = final_score
                best_indices = indices

        selected_indices = best_indices
        print(f"\nBest GA final score after {n_runs} runs: {best_score:.4f}")

    # --- Output Results ---
    if selected_indices:
        # Ensure indices are unique and sorted (GA solution might not be sorted)
        selected_indices = sorted(list(set(selected_indices)))
        if len(selected_indices) != args.num_views:
             print(f"Warning: GA returned {len(selected_indices)} unique views, expected {args.num_views}. This might indicate issues with uniqueness constraints during evolution.")
             # Optionally: Trim or handle this case if necessary

        selected_view_ids = [image_ids_sorted[idx] for idx in selected_indices]
        selected_view_names = []
        print("\n--- Selected Views ---")
        print(f"{'Index':<6} {'COLMAP ID':<10} {'Name':<30} {'Quality Score':<15}")
        print("-" * 70)
        for i, idx in enumerate(selected_indices):
            img_id = image_ids_sorted[idx]
            if img_id in images:
                name = images[img_id].name
                # Handle potential index out of bounds if quality score calculation failed for some images
                quality = image_quality_scores[idx] if idx < len(image_quality_scores) else -1.0
                selected_view_names.append(name)
                print(f"{idx:<6} {img_id:<10} {name:<30} {quality:<15.4f}")
            else:
                 print(f"{idx:<6} {img_id:<10} {'(Name not found)'}")


        output_path = Path(args.output_file)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for name in selected_view_names:
                    f.write(f"{name}\n")
            print(f"\nSelected view names saved to {output_path}")
        except Exception as e:
             print(f"\nError saving selected views to {output_path}: {e}")

    else:
        print(f"\nView selection using method '{args.method}' failed or returned no results.")