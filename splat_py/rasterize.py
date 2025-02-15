import torch

import cv2
import numpy as np
import threading

from splat_py.utils import transform_points_torch, compute_rays_in_world_frame
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeConic,
    RenderImage,
    PrecomputeRGBFromSH,
)
from splat_py.structs import Gaussians, Tiles
# from splat_py.config import EuRoC
from splat_py.tile_culling import (
    get_splats,
)
# EuRoCCfg = EuRoC()

def visualize_uv_on_mat(uv, width, height):
    # Create a blank black image
    mat = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert UV coordinates to integer pixel coordinates
    pixels = uv.round().astype(int)
    
    # Filter out points outside the image boundaries
    mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels = pixels[mask]
    
    # Draw points on the image
    for x, y in pixels:
        cv2.circle(mat, (x, y), 1, (0, 0, 255), -1)  # Draw red dots
    
    # Display the image
    cv2.imshow('UV Coordinate Visualization', mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_visualization_thread(uv, width, height):
    # Convert uv to NumPy array if it's not already
    if isinstance(uv, torch.Tensor):
        uv = uv.detach().cpu().numpy()
    
    # Create a new thread for the visualization
    viz_thread = threading.Thread(target=visualize_uv_on_mat, args=(uv, width, height))
    
    # Start the thread
    viz_thread.start()
    
    # Note: We're not joining the thread here, allowing the main program to continue

# Example usage in your rasterize function

def imu_to_camera_transform(points_imu):
    R_calib = np.array([
        [0.9999754,  0.0038491,  0.0058547],
        [-0.0038287,  0.9999865, -0.0035019],
        [-0.0058681,  0.0034794,  0.9999768]], dtype=np.float32)
    T_calib = np.array([0.0203127935529, -0.00510325236246, -0.0112013882026], dtype=np.float32)
    # Inverse transformation
    R_inv = R_calib.t()  # Transpose of rotation matrix is its inverse
    T_inv =  -torch.matmul(R_inv, T_calib)

    points_camera = torch.matmul(R_inv,points_imu.t()).t() + T_inv.t()
    # print(points_camera[:4],"Gaussian_splatting")
    # print(points_camera)
    return points_camera
    
def calculate_dynamic_padding(gaussians, camera, xyz_camera_frame):
    """
    Calculate dynamic padding with memory efficient operations
    """
    with torch.no_grad():
        # Use existing depth tensor for calculations
        depths = xyz_camera_frame[:, 2:3]  # This is a view, not a new tensor
        
        # Calculate max scale (single value)
        max_scale = torch.exp(gaussians.scale).max().item()
        
        # Get minimum depth (single value)
        min_depth = max(0.1, depths.abs().min().item())
        
        # Calculate padding using scalar operations
        image_dimension = float(max(camera.height, camera.width))
        projected_size = max_scale / min_depth
        
        # Calculate final padding with scalar math
        scale_factor = 0.05
        safety_margin = 1.2
        base_padding = projected_size * image_dimension * scale_factor
        raw_padding = base_padding * safety_margin
        
        # Calculate limits
        min_padding = max(10.0, image_dimension * 0.01)
        max_padding = min(150.0, image_dimension * 0.15)
        
        # Clamp the final value
        padding = max(min_padding, min(raw_padding, max_padding))
        
        return int(padding)

def rasterize(
    gaussians,
    camera_T_world,
    camera,
    near_thresh,
    far_thresh,
    mh_dist,
    use_sh_precompute,
    background_rgb,
):
    max_tensor_size = torch.cuda.get_device_properties(gaussians.xyz.device).total_memory // gaussians.xyz.element_size()
    if gaussians.xyz.numel() > max_tensor_size:
        print(f"Warning: Tensor size ({gaussians.xyz.numel()}) exceeds maximum GPU tensor size ({max_tensor_size})")
        # Handle this case (e.g., process in batches or on CPU)
    
    if gaussians.xyz.shape[0] == 0:
        print("Warning: gaussians.xyz is empty!")
        # Return appropriate empty results or handle this case as needed
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    try:
        rays = torch.zeros(1, 1, 1, dtype=gaussians.xyz.dtype, device=gaussians.xyz.device)
    except RuntimeError as e:
        print(f"Error creating rays tensor: {e}")
        print("Attempting to create rays tensor on CPU...")
        rays = torch.zeros(1, 1, 1, dtype=gaussians.xyz.dtype, device='cpu')

    xyz_camera_frame = transform_points_torch(gaussians.xyz, camera_T_world)

    uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)
    # # print(uv)
    # run_visualization_thread(uv, camera.width, camera.height)
    
    cull_mask_padding = calculate_dynamic_padding(gaussians, camera, xyz_camera_frame)
    print(f"DEBUG: Cull_mask_padding,{cull_mask_padding}")
    culling_mask = torch.zeros(
        xyz_camera_frame.shape[0],
        dtype=torch.bool,
        device=gaussians.xyz.device,
    )
    culling_mask = (
        culling_mask
        | (xyz_camera_frame[:, 2] < near_thresh)
        | (xyz_camera_frame[:, 2] > far_thresh)
    )
    culling_mask = (
        culling_mask
        | (uv[:, 0] < -1 * cull_mask_padding)
        | (uv[:, 0] > camera.width + cull_mask_padding)
        | (uv[:, 1] < -1 * cull_mask_padding)
        | (uv[:, 1] > camera.height + cull_mask_padding)
    )
    # print(f"after len culling_mask {(culling_mask==True).sum()} and {len(xyz_camera_frame)}")

    # cull gaussians outside of camera frustrum
    uv = uv[~culling_mask, :]
    xyz_camera_frame = xyz_camera_frame[~culling_mask, :]
    
    # This is the reason

    if gaussians.sh is not None:
        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternion=gaussians.quaternion[~culling_mask, :],
            scale=gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(
                gaussians.opacity[~culling_mask]
            ),  # apply sigmoid activation to opacity
            rgb=gaussians.rgb[~culling_mask, :],
            sh=gaussians.sh[~culling_mask, :],
        )
    else:
        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternion=gaussians.quaternion[~culling_mask, :],
            scale=gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(
                gaussians.opacity[~culling_mask]
            ),  # apply sigmoid activation to opacity
            rgb=gaussians.rgb[~culling_mask, :],
        )

    sigma_world = ComputeSigmaWorld.apply(culled_gaussians.quaternion, culled_gaussians.scale)
    J = ComputeProjectionJacobian.apply(xyz_camera_frame, camera.K)
    conic = ComputeConic.apply(sigma_world, J, camera_T_world)

    # perform tile culling
    tiles = Tiles(camera.height, camera.width, uv.device)

    sorted_gaussian_idx_by_splat_idx, splat_start_end_idx_by_tile_idx = get_splats(
        uv, tiles, conic, xyz_camera_frame, mh_dist
    )
    rays = torch.zeros(1, 1, 1, dtype=gaussians.xyz.dtype, device=gaussians.xyz.device)
    if culled_gaussians.sh is not None:
        sh_coeffs = torch.cat((culled_gaussians.rgb.unsqueeze(dim=2), culled_gaussians.sh), dim=2)
        if use_sh_precompute:
            render_rgb = PrecomputeRGBFromSH.apply(
                sh_coeffs, culled_gaussians.xyz, torch.inverse(camera_T_world).contiguous()
            )
        else:
            render_rgb = sh_coeffs
            # actually need to compute rays here
            rays = compute_rays_in_world_frame(camera, camera_T_world)
    else:
        render_rgb = culled_gaussians.rgb

    image = RenderImage.apply(
        render_rgb,
        culled_gaussians.opacity,
        uv,
        conic,
        rays,
        splat_start_end_idx_by_tile_idx,
        sorted_gaussian_idx_by_splat_idx,
        torch.tensor([camera.height, camera.width], device=uv.device),
        background_rgb,
    )
    return image, culling_mask, uv

