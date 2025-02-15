import cv2
import numpy as np
import torch
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure
import time
from splat_py.optimizer_manager import OptimizerManager
from splat_py.structs import GSMetrics
from splat_py.rasterize import rasterize
from splat_py.utils import (
    inverse_sigmoid,
    quaternion_to_rotation_torch,
    save_ply
)
import time 
from splat_py.plyfile_generate import save_ply
import csv
import os
import json

class SplatTrainer:
    def __init__(self, mp_kf, config):
        
        # kf include [[Cam,new_gaussians]]
        #return a list with  Cam.roscamera[1] posed image, Cam.roscamera[2] image param/matrix, cam_idx=Cam.roscamera[0]
        Cam,new_gaussians=mp_kf
        self.gaussians = new_gaussians
        # self.cameras = Cam.roscamera[2]
        self.config = config
        self.cam_idx=Cam.roscamera[0]
        self.image = Cam.roscamera[1]
        self.optimizer_manager = OptimizerManager(new_gaussians, self.config)
        self.metrics = GSMetrics()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.gaussians.xyz.device)
        self.cameras_data = []
        self.cameras_data_train = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.save_cfg_args()
        self.reset_grad_accum()

    def reset_grad_accum(self):
        # reset grad accumulators
        self.uv_grad_accum = torch.zeros(
            (self.gaussians.xyz.shape[0], 2),
            dtype=self.gaussians.xyz.dtype,
            device=self.gaussians.xyz.device,
        )
        self.xyz_grad_accum = torch.zeros(
            self.gaussians.xyz.shape,
            dtype=self.gaussians.xyz.dtype,
            device=self.gaussians.xyz.device,
        )
        self.grad_accum_count = torch.zeros(
            self.gaussians.xyz.shape[0],
            dtype=torch.int,
            device=self.gaussians.xyz.device,
        )
    def setup_image(self,image):
        return image.to(torch.float32) / self.config.saturated_pixel_value
    
        
    def reset_opacity(self):
        print("\t\tResetting opacity")
        self.gaussians.opacity = torch.nn.Parameter(
            torch.ones_like(self.gaussians.opacity)
            * inverse_sigmoid(self.config.reset_opacity_value)
        ) # reset_opacity_value = 0.2
        self.optimizer_manager.reset_opacity_exp_avg(self.gaussians)
        self.reset_grad_accum()

    def add_sh_band(self):
        num_gaussians = self.gaussians.xyz.shape[0]
        if self.config.max_sh_band == 0:
            return
        elif self.gaussians.sh is None:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                3,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_to_optimizer(self.gaussians)
        elif self.gaussians.sh.shape[2] == 3 and self.config.max_sh_band > 1:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                8,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            new_sh[:, :, :3] = self.gaussians.sh
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_band_to_optimizer(self.gaussians)
        elif self.gaussians.sh.shape[2] == 8 and self.config.max_sh_band > 2:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                15,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            new_sh[:, :, :8] = self.gaussians.sh
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_band_to_optimizer(self.gaussians)

    def delete_gaussians(self, keep_mask):
        self.gaussians.filter_in_place(keep_mask)
        self.uv_grad_accum = self.uv_grad_accum[keep_mask, :]
        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask, :]
        self.grad_accum_count = self.grad_accum_count[keep_mask]

        # remove deleted gaussians from optimizer
        self.optimizer_manager.delete_gaussians_from_optimizer(self.gaussians, keep_mask)

    def clone_gaussians(self, clone_mask, xyz_grad_avg):
        # create cloned gaussians
        
        cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()

        cloned_xyz -= xyz_grad_avg[clone_mask, :] * 0.01
        cloned_quaternion = self.gaussians.quaternion[clone_mask, :].clone().detach()
        cloned_scale = self.gaussians.scale[clone_mask, :].clone().detach()
        cloned_opacity = self.gaussians.opacity[clone_mask].clone().detach()
        cloned_rgb = self.gaussians.rgb[clone_mask, :].clone().detach()
        if self.gaussians.sh is not None:
            cloned_sh = self.gaussians.sh[clone_mask, :].clone().detach()

        # keep grads up to date
        self.uv_grad_accum = torch.cat(
            [self.uv_grad_accum, self.uv_grad_accum[clone_mask, :]], dim=0
        )
        self.xyz_grad_accum = torch.cat(
            [self.xyz_grad_accum, self.xyz_grad_accum[clone_mask, :]], dim=0
        )
        self.grad_accum_count = torch.cat(
            [self.grad_accum_count, self.grad_accum_count[clone_mask]], dim=0
        )

        # clone gaussians
        if self.gaussians.sh is not None:
            self.gaussians.append(
                cloned_xyz,
                cloned_rgb,
                cloned_opacity,
                cloned_scale,
                cloned_quaternion,
                cloned_sh,
            )
        else:
            self.gaussians.append(
                cloned_xyz, cloned_rgb, cloned_opacity, cloned_scale, cloned_quaternion
            )
        print("clone mask", torch.sum(clone_mask).detach().cpu().numpy())
        self.optimizer_manager.add_gaussians_to_optimizer(
            self.gaussians, torch.sum(clone_mask).detach().cpu().numpy()
        )

    def densify_postfix(self, new_gaussians):
        clone_mask = torch.ones(new_gaussians.xyz.size(0), dtype=torch.bool)
        cloned_xyz =  new_gaussians.xyz

        cloned_quaternion = new_gaussians.quaternion
        cloned_scale = new_gaussians.scale
        cloned_opacity = new_gaussians.opacity
        cloned_rgb = new_gaussians.rgb
        if self.gaussians.sh is not None:
            # if self.gaussians.sh.shape[0] != clone_mask.shape[0]:
                # raise ValueError(
                #     f"Shape mismatch: mask has shape {clone_mask.shape} but gaussians.sh has shape {self.gaussians.sh.shape}"
                # )
            cloned_sh = torch.zeros(
                new_gaussians.xyz.size(0),
                3,
                3,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )

        new_uv_grad_accum = torch.zeros(
            (new_gaussians.xyz.shape[0], 2),
            dtype=new_gaussians.xyz.dtype,
            device=new_gaussians.xyz.device,
        )
        new_xyz_grad_accum = torch.zeros(
            new_gaussians.xyz.shape,
            dtype=new_gaussians.xyz.dtype,
            device=new_gaussians.xyz.device,
        )
        new_grad_accum_count = torch.zeros(
            new_gaussians.xyz.shape[0],
            dtype=torch.int,
            device=new_gaussians.xyz.device,
        )

        # print("shape_here",self.uv_grad_accum.shape, self.uv_grad_accum[clone_mask, :].shape)

        self.uv_grad_accum = torch.cat(
            [self.uv_grad_accum, new_uv_grad_accum], dim=0
        )
        self.xyz_grad_accum = torch.cat(
            [self.xyz_grad_accum, new_xyz_grad_accum], dim=0
        )
        self.grad_accum_count = torch.cat(
            [self.grad_accum_count, new_grad_accum_count], dim=0
        )

        # clone gaussians
        if self.gaussians.sh is not None:
            self.gaussians.append(
                cloned_xyz,
                cloned_rgb,
                cloned_opacity,
                cloned_scale,
                cloned_quaternion,
                cloned_sh,
            )
        else:
            self.gaussians.append(
                cloned_xyz, cloned_rgb, cloned_opacity, cloned_scale, cloned_quaternion
            )
        
        # print("add_newgs", clone_mask
        self.optimizer_manager.add_gaussians_to_optimizer(
            self.gaussians, torch.sum(clone_mask).detach().cpu().numpy()
        )

    def split_gaussians(self, split_mask):
        samples = self.config.num_split_samples
        # create split gaussians
        split_quaternion = (
            self.gaussians.quaternion[split_mask, :].clone().detach().repeat(samples, 1)
        )
        split_scale = self.gaussians.scale[split_mask, :].clone().detach().repeat(samples, 1)
        split_opacity = self.gaussians.opacity[split_mask].clone().detach().repeat(samples, 1)
        split_rgb = self.gaussians.rgb[split_mask, :].clone().detach().repeat(samples, 1)
        if self.gaussians.sh is not None:
            split_sh = self.gaussians.sh[split_mask, :].clone().detach().repeat(samples, 1, 1)
        split_xyz = self.gaussians.xyz[split_mask, :].clone().detach().repeat(samples, 1)

        # centered random samples
        random_samples = torch.rand(split_mask.sum() * samples, 3, device=self.gaussians.xyz.device)
        # scale by scale factors
        scale_factors = torch.exp(split_scale)
        random_samples = random_samples * scale_factors
        # rotate by quaternion
        split_quaternion = split_quaternion / torch.norm(split_quaternion, dim=1, keepdim=True)
        split_rotations = quaternion_to_rotation_torch(split_quaternion)

        random_samples = torch.bmm(split_rotations, random_samples.unsqueeze(-1)).squeeze(-1)
        # translate by original mean locations
        split_xyz += random_samples

        # update scale
        split_scale = torch.log(torch.exp(split_scale) / self.config.split_scale_factor)

        # delete original split gaussians
        self.delete_gaussians(~split_mask)

        # add split gaussians
        if self.gaussians.sh is not None:
            self.gaussians.append(
                split_xyz, split_rgb, split_opacity, split_scale, split_quaternion, split_sh
            )
        else:
            self.gaussians.append(
                split_xyz, split_rgb, split_opacity, split_scale, split_quaternion
            )
        # print(f"clone mask shape {split_mask.shape}")
        # print("split mask................", torch.sum(split_mask).detach().cpu().numpy())

        self.optimizer_manager.add_gaussians_to_optimizer(
            self.gaussians, torch.sum(split_mask).detach().cpu().numpy() * samples
        )

    def adaptive_density_control(self, iter):
        if not (self.config.use_delete or self.config.use_clone or self.config.use_split):
            return
        print("Adaptive_density control update")

        # Step 1. Delete gaussians
        # low opacity
        keep_mask = self.gaussians.opacity > inverse_sigmoid(self.config.delete_opacity_threshold)
        keep_mask = keep_mask.squeeze(1)
        print("\tlow opacity mask: ", torch.sum(~keep_mask).detach().cpu().numpy())
        # no views or grad
        # zero_view_mask = self.grad_accum_count == 0
        # zero_grad_mask = torch.norm(self.uv_grad_accum, dim=1) == 0.0
        # print("\tzero view mask: ", torch.sum(zero_view_mask).detach().cpu().numpy())
        # print("\tzero grad mask: ", torch.sum(zero_grad_mask).detach().cpu().numpy())
        # keep_mask &= ~zero_view_mask
        # keep_mask &= ~zero_grad_mask

        if len(self.gaussians) > self.config.max_gaussians:
            print("Max gaussians exceeded, skipping densification")
            self.reset_grad_accum()
            return

        # Step 2. Densify gaussians
        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()

        uv_grad_avg_norm = torch.norm(uv_grad_avg, dim=1)

        if self.config.use_fractional_densification:
            if self.config.use_adaptive_fractional_densification:
                scale_factor = (
                    float(self.config.adaptive_control_end - iter)
                    / float(self.config.adaptive_control_end - self.config.adaptive_control_start)
                    * 2.0
                )
            else:
                scale_factor = 1.0
            uv_percentile = 1.0 - (1.0 - self.config.uv_grad_percentile) * scale_factor
            uv_split_val = torch.quantile(uv_grad_avg_norm, uv_percentile).item()
        else:
            uv_split_val = self.config.uv_grad_threshold
        densify_mask = uv_grad_avg_norm > uv_split_val
        print(
            "\tDensify mask: ",
            torch.sum(densify_mask).detach().cpu().numpy(),
            "split_val",
            uv_split_val,
        )

        scale_max = self.gaussians.scale.exp().max(dim=-1).values
        clone_mask = densify_mask & (scale_max <= self.config.clone_scale_threshold)
        print("\tClone Mask: ", torch.sum(clone_mask).detach().cpu().numpy())

        # Step 2.1 clone gaussians
        if clone_mask.any() and self.config.use_clone:
           
            self.clone_gaussians(clone_mask, xyz_grad_avg)
            # keep masks up to date
            densify_mask = torch.cat([densify_mask, densify_mask[clone_mask]], dim=0)
            scale_max = torch.cat([scale_max, scale_max[clone_mask]], dim=0)
        split_mask = densify_mask & (scale_max > self.config.clone_scale_threshold)

        if self.config.use_adaptive_fractional_densification:
            scale_factor = (
                float(self.config.adaptive_control_end - iter)
                / float(self.config.adaptive_control_end - self.config.adaptive_control_start)
                * 2.0
            )
        else:
            scale_factor = 1.0
        scale_percentile = 1.0 - (1.0 - self.config.scale_norm_percentile) * scale_factor

        scale_split = torch.quantile(scale_max, scale_percentile).item()
        too_big_mask = scale_max > scale_split
        split_mask = split_mask | too_big_mask

        print("\tSplit Mask: ", torch.sum(split_mask).detach().cpu().numpy())
        # Step 2.2 split gaussians
        if split_mask.any() and self.config.use_split:
            self.split_gaussians(split_mask)
        # print("Adaptive density control running")
        self.reset_grad_accum()
    

    def splat_and_compute_loss(self, camera_T_world, camera, background_rgb,image_gt):
        # print("Debug:",self.gaussians.rgb[0])
        image, culling_mask, uv = rasterize(
            self.gaussians,
            camera_T_world,
            camera,
            near_thresh=self.config.near_thresh,
            far_thresh=self.config.far_thresh,
            mh_dist=self.config.mh_dist,
            use_sh_precompute=self.config.use_sh_precompute,
            background_rgb=background_rgb,
        )
        uv.retain_grad()

        gt_image = image_gt.to(torch.device("cuda"))
        gt_image=self.setup_image(gt_image)
        l1_loss = torch.nn.functional.l1_loss(image, gt_image)
        # l1_loss = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True).mean()
        # for debug only
        l2_loss = torch.nn.functional.mse_loss(image, gt_image)
        psnr = -10 * torch.log10(l2_loss)
        # mse = (((image - gt_image)) ** 2).view(image.shape[0], -1).mean(1, keepdim=True)
        # psnr =  20 * torch.log10(1.0 / torch.sqrt(mse)).mean()
        
        
        

        # channel first tensor for SSIM
        ssim_loss = 1.0 - self.ssim(
            image.unsqueeze(0).permute(0, 3, 1, 2),
            gt_image.unsqueeze(0).permute(0, 3, 1, 2),
        )
        loss = (1.0 - self.config.ssim_frac) * l1_loss + self.config.ssim_frac * ssim_loss
        loss.backward()
        self.optimizer_manager.optimizer.step()

        # scale uv grad back to world coordinates - this way, uv grad is consistent across multiple cameras
        uv_grad = uv.grad.detach()
        uv_grad[:, 0] = uv_grad[:, 0] * camera.K[0, 0]
        uv_grad[:, 1] = uv_grad[:, 1] * camera.K[1, 1]
        # print("culling mask",culling_mask.shape)
        self.uv_grad_accum[~culling_mask] += torch.abs(uv_grad)
        self.xyz_grad_accum += torch.abs(self.gaussians.xyz.grad.detach())
        self.grad_accum_count += (~culling_mask).int()

        return image,gt_image,psnr         
    
    def train_debug(self, it, Cam_KF, is_newKF):
            self.total_start_time = time.time()

            # Initialize cameras data list if not exists
            if not hasattr(self, 'cameras_data'):
                self.cameras_data = []
            if is_newKF:
                newCam = Cam_KF[-1]  # New keyframe
                # Collect camera data
                cam_data = self._collect_camera_data(newCam)
                cam_data_train = self._collect_camera_data_train(newCam)
                self.cameras_data.append(cam_data)
                self.cameras_data_train.append(cam_data_train)
                # self.save_transform_train_json()  # Save transforms
                
                # If this isn't the first frame and we have previous frame's first iteration data
                if len(Cam_KF) > 1 and hasattr(self, 'first_iter_rendered_image') and hasattr(self, 'first_iter_gt_image'):
                    # Save the results from the first iteration of previous frame
                    debug_dir = os.path.join(self.config.output_dir, 'first_iterations')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Calculate PSNR from first iteration
                    first_l2_loss = torch.nn.functional.mse_loss(self.first_iter_rendered_image, self.first_iter_gt_image)
                    first_psnr = -10 * torch.log10(first_l2_loss)
                    
                    # Save to CSV
                    with open(os.path.join(self.config.output_dir, "first_iterations.csv"), "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([it-1, first_psnr.item(), self.gaussians.xyz.shape[0]])
                    
                    # Save rendered image
                    debug_image = self.first_iter_rendered_image.clip(0, 1).detach().cpu().numpy()
                    rendered_np = (debug_image * self.config.saturated_pixel_value).astype(np.uint8)[..., ::-1]
                    # rendered_np = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(debug_dir, f'iter_{it-1}_rendered.png'), rendered_np)
                    
                    # Save ground truth image
                    gt_np = (self.first_iter_gt_image.detach().cpu().numpy() * 255).astype(np.uint8)
                    # gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(debug_dir, f'iter_{it-1}_gt.png'), gt_np)
                    
                    # Save updated cameras.json and training image
                    self._save_cameras_json()
                    # self.save_train_image(newCam, len(self.cameras_data) - 1)
                    
                # Reset first iteration storage for new frame
                self.first_iter_rendered_image = None
                self.first_iter_gt_image = None
                    
            else:
                newCam = Cam_KF[-2] if len(Cam_KF) > 1 else Cam_KF[0]

            self.optimizer_manager.optimizer.zero_grad()

            camera_T_world = newCam.roscamera[1].camera_T_world  # poses
            camera = newCam.roscamera[2]  # intrinsic
            background_rgb = torch.zeros(
                3, device=self.gaussians.xyz.device, dtype=self.gaussians.xyz.dtype
            )

            # if self.config.use_background and it < self.config.use_background_end:
            #     background_rgb = (
            #         torch.ones(3, device=self.gaussians.xyz.device, dtype=self.gaussians.xyz.dtype)
            #         * float(it % 255)
            #         / 255.0
            #     )

            # Compute loss and get rendered images
            image, gt_image, psnr = self.splat_and_compute_loss(
                camera_T_world, camera, background_rgb=background_rgb, image_gt=newCam.roscamera[1].image
            )

            # Store first iteration results if we haven't stored them yet for this frame
            if self.first_iter_rendered_image is None:
                self.first_iter_rendered_image = image.clone()
                self.first_iter_gt_image = gt_image.clone()

            # Update metrics
            self.metrics.train_psnr.append(psnr.item())
            self.metrics.num_gaussians.append(self.gaussians.xyz.shape[0])
            
            print(
                "Iter: {}, PSNR: {}, N: {}".format(
                    it, psnr.detach().cpu().numpy(), self.gaussians.xyz.shape[0]
                )
            )
            if (
                it > self.config.adaptive_control_start
                and it % self.config.adaptive_control_interval == 0
                and it < self.config.adaptive_control_end
            ):
                self.adaptive_density_control(it)
        
    def collect_visualize_data(self, Cam_KF, newCam, it):
        cam_data = self._collect_camera_data(newCam)
        self.cameras_data.append(cam_data)
        
        if len(Cam_KF) > 1:
            # Save the results from the last iteration of the previous frame
            debug_dir = os.path.join(self.config.output_dir, 'final_iterations')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Calculate PSNR from last iteration
            final_l2_loss = torch.nn.functional.mse_loss(self.prev_rendered_image, self.prev_gt_image)
            
            final_psnr = -10 * torch.log10(final_l2_loss)
            
            # Save to CSV
            with open(os.path.join(self.config.output_dir, "final_iterations.csv"), "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([it - 1, final_psnr.item(), self.gaussians.xyz.shape[0]])
            
            # Save rendered image
            debug_image = self.prev_rendered_image.clip(0, 1).detach().cpu().numpy()
            rendered_np = (debug_image * self.config.saturated_pixel_value).astype(np.uint8)[..., ::-1]
            cv2.imwrite(os.path.join(debug_dir, f'iter_{it-1}_rendered.png'), rendered_np)
            
            # Save ground truth image
            gt_np = (self.prev_gt_image.detach().cpu().numpy() * 255).astype(np.uint8)
            gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(debug_dir, f'iter_{it-1}_gt.png'), gt_np)
            self._save_cameras_json()
        # self.save_train_image(newCam, len(self.cameras_data) - 1)

    def reset_buffer_collector(self):
        with torch.no_grad():
            self.prev_rendered_image = None
            self.prev_gt_image = None
            self.prev_psnr = None
            
    def clone_buffer_collector(self,image,gt_image,psnr):
        # Store current results for next iteration
        with torch.no_grad():
            self.prev_rendered_image = image.clone()
            self.prev_gt_image = gt_image.clone()
            self.prev_psnr = psnr.clone()
        
    

    def train(self, it, Cam_KF, is_newKF):
        self.total_start_time = time.time()

        # Initialize cameras data list if not exists
        self.cameras_data = []
        if is_newKF:
            newCam = Cam_KF[-1]  # New keyframe
            with torch.no_grad():
                self.collect_visualize_data(Cam_KF, newCam,it)                    
        else:
            newCam = Cam_KF[-2] if len(Cam_KF) > 1 else Cam_KF[0]
        
        self.reset_buffer_collector()
        # Add gradient clipping here
        self.optimizer_manager.optimizer.zero_grad()

        camera_T_world = newCam.roscamera[1].camera_T_world  # poses
        camera = newCam.roscamera[2]  # intrinsic
        background_rgb = torch.ones(
            3, device=self.gaussians.xyz.device, dtype=self.gaussians.xyz.dtype
        )

        # if self.config.use_background and it < self.config.use_background_end:
        #     background_rgb = (
        #         torch.ones(3, device=self.gaussians.xyz.device, dtype=self.gaussians.xyz.dtype)
        #         * float(it % 255)
        #         / 255.0
        #     )

        # Compute loss and get rendered images
        image, gt_image, psnr = self.splat_and_compute_loss(
            camera_T_world, camera, background_rgb=background_rgb, image_gt=newCam.roscamera[1].image
        )
        # with torch.no_grad():s
        self.clone_buffer_collector(image,gt_image,psnr)    

        # Update metrics
        self.metrics.train_psnr.append(psnr.item())
        self.metrics.num_gaussians.append(self.gaussians.xyz.shape[0])
        
        print(
            "Iter: {}, PSNR: {}, N: {}".format(
                it, psnr.detach().cpu().numpy(), self.gaussians.xyz.shape[0]
            )
        )

        if (
            it > self.config.adaptive_control_start
            and it % self.config.adaptive_control_interval == 0
            and it < self.config.adaptive_control_end
        ):
            self.adaptive_density_control(it)
        # if (
        #     it > self.config.reset_opacity_start
        #     and it < self.config.reset_opacity_end
        #     and it % self.config.reset_opacity_interval == 0
        # ):
        #     self.reset_opacity()

        if it % self.config.save_3D_map_interval == 0 and it > 0:
            with torch.no_grad():
                self.save_ply_file(it)
        
        if it == 0:
            self.save_cfg_args()    
        

    def save_ply_file(self, it):
        """Save PLY file with current Gaussian state."""
        ply_dir = os.path.join(self.config.output_dir, 'point_cloud', f'iteration_{it}')
        os.makedirs(ply_dir, exist_ok=True)
        final_ply_path = os.path.join(ply_dir, "point_cloud.ply")
        
        save_ply(final_ply_path, self.gaussians)
        print(f"Saved PLY file to {final_ply_path} in iteration {it}")

    # Modified collect camera data method
    def _collect_camera_data_train(self, newCam):
        """Collect camera data in the required format from ROS camera."""
        # Camera pose from camera_T_world (invert to get world_T_camera)
        camera_T_world = newCam.roscamera[1].camera_T_world  # 4x4 transform matrix
        # Convert to numpy and invert to get world_T_camera
        c2w = np.linalg.inv(camera_T_world.cpu().numpy())
        
        # Extract position (translation) and rotation
        position = c2w[:3, 3].tolist()
        rotation = c2w[:3, :3].tolist()
        
        return {
            "file_path": f"./ResultData/train/r_{len(self.cameras_data)}",
            "rotation": rotation,
            "position": position,
            "transform_matrix": [
                [float(c2w[0,0]), float(c2w[0,1]), float(c2w[0,2]), float(c2w[0,3])],
                [float(c2w[1,0]), float(c2w[1,1]), float(c2w[1,2]), float(c2w[1,3])],
                [float(c2w[2,0]), float(c2w[2,1]), float(c2w[2,2]), float(c2w[2,3])],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }

    def _collect_camera_data(self, newCam):
        """Collect camera data in the required format from ROS camera with pose alignment."""
        # Get camera_T_world transform 
        camera_T_world = newCam.roscamera[1].camera_T_world  # 4x4 transform matrix
        
        # Convert to numpy for manipulation
        T = camera_T_world.cpu().numpy()
        
        # Create calibration transform
        R2 = self.config.R_calib.T
        T2 = -self.config.R_calib @ self.config.T_calib
        
        Tc = np.eye(4)
        Tc[:3, :3] = R2
        Tc[:3, 3] = T2
        
        # Apply calibration alignment
        aligned_transform = Tc @ np.linalg.inv(T)
        
        # Convert to world_T_camera (c2w)
        c2w = np.linalg.inv(aligned_transform)
        
        # Extract position (translation) and rotation
        position = c2w[:3, 3].tolist()
        rotation = c2w[:3, :3].tolist()
        
        return {
            "id": len(self.cameras_data),
            "img_name": f"r_{len(self.cameras_data)}",
            "width": newCam.roscamera[2].width,
            "height": newCam.roscamera[2].height,
            "position": position,
            "rotation": rotation,
            "fx": newCam.roscamera[2].K[0, 0].item(),
            "fy": newCam.roscamera[2].K[1, 1].item()
        }

    def _save_cameras_json(self):
        """Save camera data in a single line compact JSON format."""
        json_path = os.path.join(self.config.output_dir, "cameras.json")
        # Use ensure_ascii=False to keep unicode characters as is
        with open(json_path, 'w') as f:
            f.write('[' + ','.join(json.dumps(camera, separators=(',',':'), ensure_ascii=False) 
                                for camera in self.cameras_data) + ']')
    
            
    def save_cfg_args(self):
        """Save configuration arguments to cfg_args file."""
        config_data = {
            "data_device": "cuda",
            "eval": False,
            "images": "images",
            "model_path": self.config.output_dir,  # Using your output directory
            "resolution": -1,
            "sh_degree": 0,
            "source_path": self.config.source_path if hasattr(self.config, 'source_path') else "images",
            "white_background": False
        }
        
        # Convert to Namespace style string
        namespace_str = f"Namespace({', '.join(f'{k}={repr(v)}' for k, v in config_data.items())})"
        
        # Save to file
        cfg_path = os.path.join(self.config.output_dir, "cfg_args")
        with open(cfg_path, 'w') as f:
            f.write(namespace_str)
