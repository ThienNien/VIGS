import torch
from splat_cuda import render_silhouette_cuda
from splat_py.utils import transform_points_torch
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeConic,
)
from splat_py.structs import Gaussians, Tiles
from splat_py.tile_culling import get_splats


def render_silhouette(
    gaussians,
    camera_T_world,
    camera,
    near_thresh,
    far_thresh,
    cull_mask_padding,
    mh_dist,
):
    """
    Render a silhouette image where each pixel represents the accumulated opacity
    from all Gaussians that project to that pixel.
    """
    with torch.no_grad():
        # Transform points to camera frame
        xyz_camera_frame = transform_points_torch(gaussians.xyz, camera_T_world)
        uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)
        
        # Perform frustum culling
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

        # Cull gaussians outside of camera frustum
        uv = uv[~culling_mask, :]
        xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

        # Create culled gaussians struct
        culled_gaussians = Gaussians(
            xyz=gaussians.xyz[~culling_mask, :],
            quaternion=gaussians.quaternion[~culling_mask, :],
            scale=gaussians.scale[~culling_mask, :],
            opacity=torch.sigmoid(gaussians.opacity[~culling_mask]),
            rgb=gaussians.rgb[~culling_mask, :],
            sh=gaussians.sh[~culling_mask, :] if gaussians.sh is not None else None,
        )

        # Compute necessary parameters for rendering
        sigma_world = ComputeSigmaWorld.apply(culled_gaussians.quaternion, culled_gaussians.scale)
        J = ComputeProjectionJacobian.apply(xyz_camera_frame, camera.K)
        conic = ComputeConic.apply(sigma_world, J, camera_T_world)

        # Perform tile culling
        tiles = Tiles(camera.height, camera.width, uv.device)
        sorted_gaussian_idx_by_splat_idx, splat_start_end_idx_by_tile_idx = get_splats(
            uv, tiles, conic, xyz_camera_frame, mh_dist
        )

        # Create silhouette image tensor
        silhouette_image = torch.zeros(
            camera.height, camera.width, 1,
            dtype=torch.float32,
            device=uv.device
        )

        # Render silhouette
        render_silhouette_cuda(
            uv,
            culled_gaussians.opacity,
            conic,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            silhouette_image,
        )

        return silhouette_image