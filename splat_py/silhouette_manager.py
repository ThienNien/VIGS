import torch
import numpy as np
import cv2
from splat_py.utils import transform_points_torch, inverse_sigmoid
from splat_py.cuda_autograd_functions import CameraPointProjection
from splat_py.structs import Gaussians
from splat_py.silhouette import render_silhouette

class SilhouetteManager:
    def __init__(self, config):
        """
        Initialize SilhouetteManager with configuration
        
        Args:
            config: Configuration object containing rendering parameters
        """
        self.config = config
        self.silhouette_threshold = 0.99  # From paper Eq.8
        
    def select_points_from_silhouette(self, gaussians, camera_T_world, camera, points_3d):
        """
        Select points based on silhouette rendering.
        
        Args:
            gaussians: Current Gaussian map
            camera_T_world: Camera pose matrix
            camera: Camera intrinsics
            points_3d: Nx3 tensor of points in world frame
            
        Returns:
            mask: Boolean mask for points to keep
        """
        if points_3d is None or len(points_3d) == 0:
            return None
            
        # Render silhouette image
        silhouette_image= render_silhouette(
            gaussians,
            camera_T_world,
            camera,
            near_thresh=self.config.near_thresh,
            far_thresh=self.config.far_thresh,
            cull_mask_padding=self.config.cull_mask_padding,
            mh_dist=self.config.mh_dist
        )
        
        # Create mask for underrepresented areas (M = V < 0.99)
        mask = silhouette_image < self.silhouette_threshold
        
        # Project points to image plane
        points_camera_frame = transform_points_torch(points_3d, camera_T_world)
        uv = CameraPointProjection.apply(points_camera_frame, camera.K)
        
        # Convert UV coordinates to pixel indices
        pixel_coords = torch.round(uv).long()
        
        # Create validity mask for points that project within image bounds
        valid_points = (
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < camera.width) &
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < camera.height)
        )
        
        # Get mask values at projected points
        point_mask = torch.zeros(len(points_3d), dtype=torch.bool, device=points_3d.device)
        valid_pixel_coords = pixel_coords[valid_points]
        point_mask[valid_points] = mask[
            valid_pixel_coords[:, 1],  # y coordinates
            valid_pixel_coords[:, 0],  # x coordinates
            0  # Remove singleton dimension
        ]
        
        return point_mask
        
    def filter_gaussians_by_silhouette(self, new_gaussians, camera_T_world, camera):
        """
        Filter new Gaussians based on silhouette rendering.
        
        Args:
            new_gaussians: New Gaussians to potentially add
            camera_T_world: Camera pose matrix
            camera: Camera intrinsics
            
        Returns:
            filtered_gaussians: Filtered Gaussians in underrepresented areas
        """
        # Get selection mask
        selection_mask = self.select_points_from_silhouette(
            new_gaussians, 
            camera_T_world, 
            camera, 
            new_gaussians.xyz
        )
        
        if selection_mask is None or not selection_mask.any():
            return None
            
        # Create filtered Gaussians
        filtered_gaussians = Gaussians(
            xyz=new_gaussians.xyz[selection_mask],
            quaternion=new_gaussians.quaternion[selection_mask],
            scale=new_gaussians.scale[selection_mask],
            opacity=new_gaussians.opacity[selection_mask],
            rgb=new_gaussians.rgb[selection_mask],
            sh=new_gaussians.sh[selection_mask] if new_gaussians.sh is not None else None
        )
        
        return filtered_gaussians
        
    def visualize_silhouette_mask(self, silhouette_image, save_path=None):
        """
        Visualize the silhouette and mask.
        
        Args:
            silhouette_image: Rendered silhouette tensor
            save_path: Optional path to save visualization
        """
        # Convert to numpy and normalize
        vis = (silhouette_image.detach().cpu().numpy() * 255).astype(np.uint8)
        
        # Create mask visualization
        mask = (silhouette_image < self.silhouette_threshold).cpu().numpy()
        mask_vis = (mask * 255).astype(np.uint8)
        
        # Optional: Apply color to better visualize
        mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        
        # Stack horizontally
        combined = np.hstack([vis, mask_vis])
        
        if save_path:
            cv2.imwrite(save_path, combined)
            
        return combined