from dataclasses import dataclass
import tyro
from typing_extensions import Literal
import yaml
import numpy as np


class yamlEnabled(object):
    """
    Decorator to enable yaml serialization for a class.
    from: https://stackoverflow.com/questions/74723634/how-do-you-use-a-frozen-dataclass-in-a-dictionary-and-export-it-to-yaml
    """

    def __init__(self, tag):
        self.tag = tag

    def __call__(self, cls):
        def to_yaml(dumper, data):
            return dumper.represent_mapping(self.tag, vars(data))

        yaml.SafeDumper.add_representer(cls, to_yaml)

        def from_yaml(loader, node):
            data = loader.construct_mapping(node)
            return cls(**data)

        yaml.SafeLoader.add_constructor(self.tag, from_yaml)
        return cls


@yamlEnabled("!SplatConfig")
@dataclass
class SplatConfig:
    def __post_init__(self):
        # Initialize dataset parameters
        fx, fy, cx, cy, R_calib, T_calib = self.dataset("Openloris")
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.R_calib = R_calib
        self.T_calib = T_calib
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)
    def dataset(self,dataset):
        if dataset == "Openloris":
            fx = 611.4509887695312
            fy = 611.4857177734375
            cx = 433.2039794921875
            cy = 249.4730224609375
            R_calib = np.array([
                [0.9999754,  0.0038491,  0.0058547],
                [-0.0038287,  0.9999865, -0.0035019],
                [-0.0058681,  0.0034794,  0.9999768]], dtype=np.float32)
            T_calib = np.array([0.0203127935529, -0.00510325236246, -0.0112013882026], dtype=np.float32)
        elif dataset == "EuRoC":
            R_calib = np.array([
                        [0.0148655429818, -0.999880929698, 0.00414029679422],
                        [0.999557249008, 0.0149672133247, 0.025715529948],
                        [-0.0257744366974, 0.00375618835797, 0.999660727178]], dtype=np.float32)
            T_calib = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949], dtype=np.float32)
            fx = 4.616e+02
            fy = 4.603e+02
            cx = 3.630e+02
            cy = 2.481e+02
        elif dataset == "Hidrive":
            R_calib = np.array([
                        [0.9999,    0.0065,   -0.0109],
                        [-0.0108,   -0.0124,   -0.9999],
                        [-0.0066,    0.9999,   -0.0123]], dtype=np.float32)
            T_calib = np.array([-0.1372, -0.0551, -0.0691], dtype=np.float32)
            fx = 579.4882694716105
            fy = 579.4882694716105
            cx = 342.1329879760742
            cy = 255.80628776550293
        
        elif dataset == "zed2i":
            R_calib = np.array([
                        [0.999986, -0.004323, 0.003173],
                        [0.004324, 0.999990, -0.000566],
                        [-0.003170, 0.000580, 0.999995]], dtype=np.float32)
            T_calib = np.array([23.061001, -0.217000, -2.000000], dtype=np.float32)
            fx = 266.6724853515625
            fy = 266.7025146484375
            cx = 314.6650085449219
            cy = 187.39224243164062
        
        return fx,fy,cx,cy,R_calib,T_calib
    


    using_camdepth: bool = False
    dataset_path: str = "data/garden"
    """downsample factor for the images - if applicable"""
    downsample_factor: int = 4
    """output directory for saving the results"""
    output_dir: str = "VIGS_output"

    """interval for saving checkpoints"""
    checkpoint_interval: int = 2500

    evaluate_interval:int = 1500
    """initialize gaussians from checkpoint"""
    load_checkpoint: bool = False
    """path to saved gaussian checkpoint"""
    checkpoint_path: str = ""

    """interval for saving debug training images"""
    save_debug_image_interval: int = 150
    """interval to print debug information"""
    print_interval: int = 2

    """initial opacity for gaussians initialized from a point cloud"""
    initial_opacity: float = 0.01#0.3False
    """number of neighbors used to compute the initial scale"""
    initial_scale_num_neighbors: int = 3#3=>5
    """factor to scale the distance to the nearest neighbors"""
    initial_scale_factor: float = 0.15
    """maximum initial scale"""
    max_initial_scale: float = 0.3#0.1 #good 0.3

    """gaussians closer than this are culled alongside points outside of fov"""
    near_thresh: float = 0.01 #0.3
    """gaussians farther than this are culled alongside points outside of fov"""
    far_thresh: float = 100.0 #500.0
    """mahalanobis distance for tile culling 3.0 = 99.7%"""
    mh_dist: float = 3.0
    """keep gaussians that project within this padding of image during frustrum culling"""
    cull_mask_padding: int = 100 #100
    """max rgb value for splatted image"""
    saturated_pixel_value: float = 255.0

    """number of iterations for training"""
    num_iters: int = 61
    """fraction of ssim loss to l1 loss"""
    ssim_frac: float = 0.2
    "base learning rate"
    base_lr: float = 0.01#0.002
    """learning rate multiplier for xyz"""
    xyz_lr_multiplier: float = 0.5#0.01/ 1.5 --- good
    """learning rate multiplier for quaternion"""
    quat_lr_multiplier: float = 0.25  #2#0.1 good#0.3
    """learning rate multiplier for scale"""
    scale_lr_multiplier: float =  0.3#5 0.2 #0.8 good 0.3
    """learning rate multiplier for opacity"""
    opacity_lr_multiplier: float = 2#10 #1--good
    """learning rate multiplier for rgb"""
    rgb_lr_multiplier: float = 0.5#2
    """learning rate multiplier for spherical harmonics"""
    sh_lr_multiplier: float = 0.2

    """interval to evaluate test images"""
    # test_eval_interval: int = 500
    # """select every nth image for the test split - 8 is same as GS and Mip-Nerf 360 papers"""
    # test_split_ratio: int = 8

    """use background color"""
    use_background: bool = True
    """background color end interval"""
    use_background_end: int = 2500 #6600

    """interval to reset all opacities to a fixed value"""
    reset_opacity_interval: int = 300
    """opacity value to reset to"""
    reset_opacity_value: float = 0.20 #0.20
    """start iteration for reset opacity"""
    reset_opacity_start: int = 20
    """end iteration for reset opacity"""
    reset_opacity_end: int = 3000000

    """precompute SH to RGB for each gaussian - speeds up computation ~1.4-2x"""
    use_sh_precompute: bool = True
    """max SH band to use - 0 is no view dependent color"""
    max_sh_band: Literal[0, 1, 2, 3] = 0
    """add SH band every interval until all are added"""
    add_sh_band_interval: int = 100 #1000

    """use split gaussians"""
    use_split: bool = True
    """use clone gaussians"""
    use_clone: bool = True
    """use delete gaussians"""
    use_delete: bool = True

    """start iteration for adaptive control"""
    adaptive_control_start: int = 100
    """end iteration for adaptive control"""
    adaptive_control_end: int = 3000000
    """interval for adaptive control"""
    adaptive_control_interval: int = 100

 


    """max number of gaussians"""
    max_gaussians: int = 4250000

    """delete gaussians with opacity below this threshold"""
    delete_opacity_threshold: float = 0.3 #0.3
    """clone gaussians with scale below this threshold"""
    clone_scale_threshold: float = 0.05#0.01
    """delete gaussians with scale norm above this threshold"""
    max_scale_norm: float = 0.8 #0.8
    """densify a fixed fraction of gaussians every iteration"""
    use_fractional_densification: bool = False 
    """front load densification - slower but slightly higher psnr"""
    use_adaptive_fractional_densification: bool = True

    """densify gaussians over this percentile - only used if use_fractional_densification is True"""
    uv_grad_percentile: float = 0.96
    """densify gaussians over this percentile - only used if use_fractional_densification is True"""
    scale_norm_percentile: float = 0.99 #0.99

    """densify gaussians over this threshold - only used if use_fractional_densification is False"""
    uv_grad_threshold: float = 0.0002#0.0002

    """decrease scale of split gaussians by this factor"""
    split_scale_factor: float = 1.5
    """number of samples to split gaussians into"""
    num_split_samples: int = 2

    save_3D_map_interval: int = 10000
    
SplatConfigs = tyro.extras.subcommand_type_from_defaults(
    {
        "7k": SplatConfig(num_iters=200,
                          save_debug_image_interval = 15,
                          print_interval = 15,
                          adaptive_control_start=100,
                          adaptive_control_interval=100),  # default config is 7k
        "30k": SplatConfig(
            num_iters=1000,
            adaptive_control_start=1500,
            adaptive_control_end=27500,
            adaptive_control_interval=300,
            reset_opacity_end=27500,
            use_background_end=28000,
        ),
    }
)
