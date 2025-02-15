import torch

from splat_cuda import (
    get_sorted_gaussian_list,
)


def get_splats(
    uvs,
    tiles,
    conic,
    xyz_camera_frame,
    mh_dist,
):  

    try:
        non_finite = torch.any(~torch.isfinite(xyz_camera_frame))
    except RuntimeError as e:
        print(f"CUDA error in get_splats: {e}")
        print("Attempting to process on CPU...")
        xyz_camera_frame_cpu = xyz_camera_frame.cpu()
        non_finite = torch.any(~torch.isfinite(xyz_camera_frame_cpu))

    if non_finite:
        print("Warning: Non-finite values detected in xyz_camera_frame")
    

    # nan in xyz will cause an unrecoverable sorting failure
    if torch.any(~torch.isfinite(xyz_camera_frame)):
        print("xyz_camera_frame has NaN")
        exit()

    # if uvs.shape[0] == 0:
    #     exit()

    return get_sorted_gaussian_list(
        1024,
        uvs,
        xyz_camera_frame,
        conic,
        tiles.x_tiles_count,
        tiles.y_tiles_count,
        mh_dist,
    )
