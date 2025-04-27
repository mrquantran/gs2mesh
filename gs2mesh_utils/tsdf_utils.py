# =============================================================================
#  Imports
# =============================================================================

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import cv2
import copy

from gs2mesh_utils.io_utils import read_ply
import gs2mesh_utils.third_party.visualization.visualize as visualize

# =============================================================================
#  Class for TSDF fusion algorithm
# =============================================================================

class TSDF:
    def __init__(self, renderer, stereo, args, out_name):
        """
        Initialize the TSDF fusion algorithm.

        Parameters:
        renderer (Renderer): Renderer class object.
        stereo (Stereo): Stereo class object.
        args (ArgParser): Program arguments.
        out_name (str): Output name for saving the mesh.
        """
        self.model_name = stereo.model_name
        self.renderer = renderer
        self.out_name = out_name
        self.args = args

    def run(self, visualize=True):
        """
        Run the TSDF fusion algorithm.

        Parameters:
        visualize (bool): Flag to visualize the RGB and depth images for debugging.

        Returns:
        None
        """
        valid = self.args.TSDF_valid if  self.args.TSDF_valid is not None else list(range(len(self.renderer)))
        skip =  self.args.TSDF_skip if  self.args.TSDF_skip is not None else []
        voxel_length= self.args.TSDF_voxel/512

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_length),
            sdf_trunc= self.args.TSDF_sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        for camera_number, left_camera in enumerate(tqdm(self.renderer.left_cameras)):
            if camera_number % self.args.TSDF_dilate != 0:
                continue
            if valid is not None and camera_number not in valid:
                continue
            if skip is not None and camera_number in skip:
                continue
            output_dir = self.renderer.render_folder_name(camera_number)
            image = np.array(Image.open(os.path.join(output_dir, 'left.png'))).astype(np.uint8)
            depth = np.load(os.path.join(output_dir, f'out_{self.model_name}', 'depth.npy'))
            if  self.args.TSDF_use_mask:
                object_mask = (np.load(os.path.join(output_dir, 'left_mask.npy')).astype(bool))
                if  self.args.TSDF_invert_mask:
                    object_mask = ~object_mask
                if  self.args.TSDF_erode_mask:
                    closing_kernel = np.ones(( self.args.TSDF_closing_kernel_size,  self.args.TSDF_closing_kernel_size), np.uint8)
                    erosion_kernel = np.ones(( self.args.TSDF_erosion_kernel_size,  self.args.TSDF_erosion_kernel_size), np.uint8)
                    closing = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)
                    erosion = cv2.erode(closing, erosion_kernel, iterations=1)
                    object_mask = erosion > 0.5
                depth = depth * object_mask
            if  self.args.TSDF_use_occlusion_mask:
                occlusion_mask = np.load(os.path.join(output_dir, f'out_{self.model_name}', 'occlusion_mask.npy')).astype(bool)
                depth = depth * occlusion_mask

            depth = np.where(depth <  self.args.TSDF_min_depth_baselines * self.renderer.baseline, 0, depth)

            extrinsic_matrix = left_camera['extrinsic'].copy()
            extrinsic_matrix[:3, 3] /=  self.args.TSDF_scale

            depth_o3d = o3d.geometry.Image((depth).astype(np.float32))

            rgb_o3d = o3d.geometry.Image(image)

            depth_trunc = self.renderer.baseline *  self.args.TSDF_max_depth_baselines /  self.args.TSDF_scale
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale= self.args.TSDF_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

            if visualize:
                # Save color image
                color_img = np.asarray(rgbd_image.color)
                plt.imsave(os.path.join(output_dir, f'out_{self.model_name}', 'color_debug.png'), color_img)

                # Save depth image (normalize for visualization)
                depth_img = np.asarray(rgbd_image.depth)
                norm_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                norm_depth = norm_depth.astype(np.uint8)
                plt.imsave(os.path.join(output_dir, f'out_{self.model_name}', 'depth_debug.png'), norm_depth, cmap='gray')
                print(f"minimal depth: {np.asarray(rgbd_image.depth)[np.asarray(rgbd_image.depth) != 0].min()}, maximal depth: {np.asarray(rgbd_image.depth).max()}")

            intrinsics = o3d.camera.PinholeCameraIntrinsic(left_camera['width'], left_camera['height'], left_camera['fx'], left_camera['fy'], left_camera['cx'], left_camera['cy'])
            volume.integrate(rgbd_image, intrinsics, np.linalg.inv(extrinsic_matrix))
        self.mesh = volume.extract_triangle_mesh()
        self.mesh.scale( self.args.TSDF_scale, (0, 0, 0))
        self.mesh.compute_vertex_normals()

    def save_mesh(self):
        """
        Save the original mesh to a file.

        Returns:
        None
        """
        o3d.io.write_triangle_mesh(os.path.join(self.renderer.output_dir_root, f'{self.out_name}_mesh.ply'), self.mesh)
        print("SAVED MESH")

    def clean_mesh(self):
        """
        Clean the mesh by removing small clusters of triangles and save it.

        Returns:
        None
        """

        thres =  self.args.TSDF_cleaning_threshold/ self.args.TSDF_scale

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (self.mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < thres
        self.clean_mesh = copy.deepcopy(self.mesh)
        self.clean_mesh.remove_triangles_by_mask(triangles_to_remove)
        self.clean_mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(os.path.join(self.renderer.output_dir_root, f'{self.out_name}_cleaned_mesh.ply'), self.clean_mesh)
        print("SAVED CLEANED MESH")

    def visualize_mesh(self, subsample=100, GT_path=None, show_clean=True):
        """
        Plot the original/cleaned mesh using plotly.

        Parameters:
        subsample (int): Subsampling factor to reduce the number of visualized points. Use a larger value if your mesh is large.
        GT_path (str): Path to the ground truth mesh file, in case one exists and is aligned for comparison.
        show_clean (bool): Flag to show the cleaned mesh.

        Returns:
        None
        """
        mesh_to_show = None
        if show_clean:
            mesh_to_show = self.clean_mesh
        else:
            mesh_to_show = self.mesh
        points = np.asarray(mesh_to_show.vertices)[::subsample]
        x, y, z = *points.T,
        trace_points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=1, opacity=1), hoverinfo="skip", name="OURS")

        trace_points_GT = None
        if GT_path is not None:
            xyz_GT, _ = read_ply(GT_path)
            xyz_GT = xyz_GT[::subsample]
            x_GT, y_GT, z_GT = *xyz_GT.T,
            trace_points_GT = go.Scatter3d(x=x_GT, y=y_GT, z=z_GT, mode="markers", marker=dict(size=1, opacity=1), hoverinfo="skip", name="GT")

        vis_depth = 0.02 * 5

        traces_all = [trace_points, trace_points_GT] if trace_points_GT is not None else [trace_points]
        layout = go.Layout(scene=dict(xaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      yaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      zaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                      xaxis_title="X", yaxis_title="Y", zaxis_title="Z", dragmode="orbit",
                                      aspectratio=dict(x=1, y=1, z=1), aspectmode="data"), height=800)
        fig = go.Figure(data=traces_all, layout=layout)

        fig.show()
