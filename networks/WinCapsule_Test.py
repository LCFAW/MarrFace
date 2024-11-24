import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from renderer.renderer_ddp import Renderer,Renderer_black
from .Backbone_WinT import WinTransformer
import cv2

class WinCapsule():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name')
        self.device = cfgs.get('device', 'cpu')
        self.save_dir = cfgs.get('checkpoint_dir', False)

        self.image_size = cfgs.get('image_size', 224)
        self.map_size = cfgs.get('map_size', 64)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        
        self.patch_size = cfgs.get('patch_size', 4)
        self.embed_dim = cfgs.get('embed_dim', 864)
        self.depths = cfgs.get('depths', [2,2,6,2])
        self.num_heads = cfgs.get('num_heads', [3,6,12,24])
        self.num_cap_part = cfgs.get('num_cap_part', 6)
        self.num_cap_obj = cfgs.get('num_cap_obj', 1)

        self.num_parts = 4
        self.num_subparts = 16
        self.num_superparts = 6

        self.use_conf_map = True
        
        self.renderer = Renderer(cfgs)
        self.renderer_black = Renderer_black(cfgs)

        #* depth args
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        
        #* light args
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)

        #* pose args
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)

        self.depth_dim = 384
        self.albedo_dim = 384
        self.light_dim = 48
        self.view_dim = 48       

        #* Encoder
        if self.model_name == 'wint':
            self.net_enc = WinTransformer(img_size = self.image_size,
                                    patch_size = self.patch_size,
                                    in_chans = 3,
                                    num_classes = self.num_cap_part + self.num_cap_obj, 
                                    embed_dim = 96,
                                    depths = self.depths,
                                    num_heads= self.num_heads,
                                    window_size= 7,
                                    drop_path_rate=0.2,
                                    embed_dim_output = self.embed_dim,
                                    caps_num=[1,1,1,1])

        self.net_supercaps = Translator_NN_caps(cin=768, cout =self.embed_dim, num_caps=self.num_superparts, num_each_caps=1)      
        self.net_caps = Translator_NN_caps(cin=384, cout =self.embed_dim, num_caps=self.num_parts, num_each_caps=1)
        self.net_subcaps = Translator_NN_caps(cin=192, cout =self.embed_dim, num_caps=self.num_subparts, num_each_caps=1)
        
        #* super Decoder
        self.netD_superdec = Translator_NN_depth_view(cin=self.depth_dim, cout=1,nf=64, activation=None, cfgs=cfgs)
        self.netA_superdec = Translator_NN_albedo(cin=self.albedo_dim, cout=3, nf=64, cfgs=cfgs)
        self.netL_super = Translator_NN_light(cin=48, cout=4, nf=32, cfgs=cfgs)
        self.netV_super = Translator_NN_View(cin=48, cout=6, nf=32, cfgs=cfgs)

        #* Decoder
        self.netD_dec = Translator_NN_depth_view(cin=self.depth_dim, cout=1,nf=64, activation=None, cfgs=cfgs)
        self.netA_dec = Translator_NN_albedo(cin=self.albedo_dim, cout=3, nf=64, cfgs=cfgs)
        self.netL = Translator_NN_light(cin=48, cout=4, nf=32, cfgs=cfgs)
        self.netV = Translator_NN_View(cin=48, cout=6, nf=32, cfgs=cfgs)
        
        #* sub Decoder
        self.netD_subdec = Translator_NN_depth_view(cin=self.depth_dim, cout=1,nf=64, activation=None, cfgs=cfgs)
        self.netA_subdec = Translator_NN_albedo(cin=self.albedo_dim, cout=3, nf=64, cfgs=cfgs)
        self.netL_sub = Translator_NN_light(cin=48, cout=4, nf=32, cfgs=cfgs)
        self.netV_sub = Translator_NN_View(cin=48, cout=6, nf=32, cfgs=cfgs)

        if self.use_conf_map:
            self.netC = ConfNet(cin=3, cout=2, nf=64, zdim=128)
        
        self.tar_resize_224 = torchvision.transforms.Resize(size=(224, 224))
        self.tar_resize_128 = torchvision.transforms.Resize(size=(128, 128))
        self.tar_resize_64 = torchvision.transforms.Resize(size=(64, 64))
        self.tar_resize = torchvision.transforms.Resize(size=(self.map_size, self.map_size))
        
        #* optimizer
        self.network_names = [k for k in vars(self) if 'net' in k]
        self.to_device()

        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.count = 0
        self.check = None
        
        self.super_depth_var = []
        self.depth_var = []
        self.sub_depth_var = []
        
        self.super_normal_var = []
        self.normal_var = []
        self.sub_normal_var = []

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                network = getattr(self, k)
                if hasattr(network, 'module'):
                    network.module.load_state_dict(cp[k])
                else:
                    network.load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k], strict=False)

    def to_device(self):
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to('cuda'))

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def forward(self, input):
        """Feedforward once."""
        input,name = input
        input = input.to('cuda') *2.-1.
        input_im = self.tar_resize(input)  # 64x64
        b, c, h, w = input_im.shape

        #* 1. Image Encoding
        obj_caps, part_caps_list = self.net_enc(input)  # 64x64
        super_part_caps,super_parts_map = self.net_supercaps(part_caps_list, layer=-1)
        part_caps, parts_map = self.net_caps(part_caps_list, layer=-2)
        sub_part_caps, sub_parts_map = self.net_subcaps(part_caps_list, layer=-3)

        #*###################################################### High-Level #####################################################

        #* 2. Part-level Encoding
        super_part_caps_depth = super_part_caps[:,:,:384]
        super_part_caps_albedo = super_part_caps[:,:,384:768]
        super_part_caps_view = super_part_caps[:,:,768:816]
        super_part_caps_light = super_part_caps[:,:,816:]  

        #* 3. Predict Depth, Albedo, Light and Confidence
        #* 3.1 Depth Decoding      
        super_canon_depth, super_canon_depth_list, super_depth_one_hot  = self.netD_superdec(super_part_caps_depth)

        #* 3.2 View Decoding
        super_views = self.netV_super(super_part_caps_view)

        #* 3.3 Albedo Decoding
        super_canon_albedo, _, _, _ = self.netA_superdec(super_part_caps_albedo, super_depth_one_hot)  # Bx3xHxW

        #* 3.4 Lighting Decoding
        super_canon_light_a, super_canon_light_b, super_canon_light_d = self.netL_super(super_part_caps_light.mean(1))  # Bx4

        #* 4. Lambertian Model Rendering
        #* 4.1 shading
        super_canon_normal = self.renderer.get_normal_from_depth(super_canon_depth) 
        super_canon_diffuse_shading = (super_canon_normal * super_canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        super_canon_shading = super_canon_light_a.view(-1,1,1,1) + super_canon_light_b.view(-1,1,1,1)*super_canon_diffuse_shading

        #* used to visualize
        super_canon_depth_show = ((super_canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1).repeat((1,3,1,1))        
        super_canon_normal_show = super_canon_normal.detach().cpu()
        super_canon_shading_show = super_canon_shading.detach().cpu() 

        #* 4.2 cannon_pixel
        super_canon_im = (super_canon_albedo/2+0.5) * super_canon_shading *2-1

        #* 4.3 Rendering
        self.renderer.set_transform_matrices(super_views)
        super_recon_depth = self.renderer.warp_canon_depth(super_canon_depth) 
        super_grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(super_recon_depth)
        super_recon_im = nn.functional.grid_sample(super_canon_im, super_grid_2d_from_canon, mode='bilinear', align_corners=True)
        
        #* 5. calculate variations of depth and z-axis normal: depth variations*25 because of the scaling factor
        self.super_depth_var.append(super_canon_depth.reshape(b*2,-1).var(-1)*25)
        self.super_normal_var.append(super_canon_normal.reshape(b*2,-1,3).var(-2))

        #*###################################################### Mid-Level #####################################################

        #* Part-level Encoding: similar to Super part
        part_caps_depth = part_caps[:,:,:384]
        part_caps_albedo = part_caps[:,:,384:768]
        part_caps_view = part_caps[:,:,768:816]
        part_caps_light = part_caps[:,:,816:]        

        #* Predict Depth, Albedo, Light and Confidence
        canon_depth, canon_depth_mask, depth_one_hot  = self.netD_dec(part_caps_depth)       
        views = self.netV(part_caps_view)
        canon_albedo, _, _, _ = self.netA_dec(part_caps_albedo, depth_one_hot)  # Bx3xHxW
        canon_light_a, canon_light_b, canon_light_d = self.netL(part_caps_light.mean(1))  # Bx4

        #* shading
        canon_normal = self.renderer.get_normal_from_depth(canon_depth)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1,1,1,1) + canon_light_b.view(-1,1,1,1)*canon_diffuse_shading

        canon_depth_show = ((canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1).repeat((1,3,1,1))
        canon_normal_show = canon_normal.detach().cpu()
        canon_shading_show = canon_shading.detach().cpu() 

        tmp_d = torch.zeros_like(canon_light_d.view(-1,1,1,3), device=input.device)
        tmp_d[:,:,:,2] = 1.    

        #* cannon_pixel
        canon_im = (canon_albedo/2+0.5) * canon_shading *2-1

        #* Rendering
        self.renderer.set_transform_matrices(views)
        recon_depth = self.renderer.warp_canon_depth(canon_depth)      
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        recon_im = nn.functional.grid_sample(canon_im, grid_2d_from_canon, mode='bilinear', align_corners=True)

        #* calculate variations
        self.depth_var.append(canon_depth.reshape(b*2,-1).var(-1)*25)
        self.normal_var.append(canon_normal.reshape(b*2,-1,3).var(-2))

        #*############################################################ Low-Level #################################################### 
        
        #* subPart-level Encoding
        sub_part_caps_depth = sub_part_caps[:,:,:384]
        sub_part_caps_albedo = sub_part_caps[:,:,384:768]
        sub_part_caps_view = sub_part_caps[:,:,768:816]
        sub_part_caps_light = sub_part_caps[:,:,816:]

        #* Predict     
        sub_canon_depth, sub_canon_depth_mask, sub_depth_one_hot  = self.netD_subdec(sub_part_caps_depth)       
        sub_views = self.netV_sub(sub_part_caps_view)
        sub_canon_albedo, _, _,  _ = self.netA_subdec(sub_part_caps_albedo, sub_depth_one_hot)
        sub_canon_light_a, sub_canon_light_b, sub_canon_light_d = self.netL_sub(sub_part_caps_light.mean(1))

        #* shading
        sub_canon_normal = self.renderer.get_normal_from_depth(sub_canon_depth)
        sub_canon_diffuse_shading = (sub_canon_normal * sub_canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        sub_canon_shading = sub_canon_light_a.view(-1,1,1,1) + sub_canon_light_b.view(-1,1,1,1)*sub_canon_diffuse_shading

        sub_canon_depth_show = ((sub_canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1).repeat((1,3,1,1))
        sub_canon_normal_show = sub_canon_normal.detach().cpu()
        sub_canon_shading_show = sub_canon_shading.detach().cpu() 

        #* cannon_pixel
        sub_canon_im = (sub_canon_albedo/2+0.5) * sub_canon_shading *2-1

        #* Rendering
        self.renderer.set_transform_matrices(sub_views)
        sub_recon_depth = self.renderer.warp_canon_depth(sub_canon_depth)    
        sub_grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(sub_recon_depth)
        sub_recon_im = nn.functional.grid_sample(sub_canon_im, sub_grid_2d_from_canon, mode='bilinear', align_corners=True)

        self.sub_depth_var.append(sub_canon_depth.reshape(b*2,-1).var(-1)*25)
        self.sub_normal_var.append(sub_canon_normal.reshape(b*2,-1,3).var(-2))

        #* For visualization
        save_dir =self.save_dir + f'/{name[0]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        super_dir = os.path.join(save_dir,'super_part')
        if not os.path.exists(super_dir):
            os.mkdir(super_dir)

        mid_dir = os.path.join(save_dir,'mid_part')
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)

        sub_dir = os.path.join(save_dir,'sub_part')
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        depth_dir = os.path.join(save_dir,'depth')
        if not os.path.exists(depth_dir):
            os.mkdir(depth_dir)

        normal_dir = os.path.join(save_dir,'normal')
        if not os.path.exists(normal_dir):
            os.mkdir(normal_dir)

        albedo_dir = os.path.join(save_dir,'albedo')
        if not os.path.exists(albedo_dir):
            os.mkdir(albedo_dir)

        #* reconstructed image
        super_recon_im_show = ((super_recon_im[0]+1)/2).clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
        super_recon_im_show = np.uint8(super_recon_im_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,'super_reconstruct.jpg'), super_recon_im_show)

        recon_im_show = ((recon_im[0]+1)/2).clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
        recon_im_show = np.uint8(recon_im_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,'mid_reconstruct.jpg'), recon_im_show)

        sub_recon_im_show = ((sub_recon_im[0]+1)/2).clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
        sub_recon_im_show = np.uint8(sub_recon_im_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,'low_reconstruct.jpg'), sub_recon_im_show)

        #* whole albedo
        super_canon_albedo_show =  super_canon_albedo[0].detach().cpu()/2 + 0.5
        super_canon_albedo_show = super_canon_albedo_show.clamp(0,1).numpy().transpose(1,2,0)
        super_canon_albedo_show = np.uint8(super_canon_albedo_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,f'super_albedo.jpg'), super_canon_albedo_show)

        canon_albedo_show =  canon_albedo[0].detach().cpu()/2 + 0.5
        canon_albedo_show = canon_albedo_show.clamp(0,1).numpy().transpose(1,2,0)
        canon_albedo_show = np.uint8(canon_albedo_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,f'mid_albedo.jpg'), canon_albedo_show)

        sub_canon_albedo_show =  sub_canon_albedo[0].detach().cpu()/2 + 0.5
        sub_canon_albedo_show = sub_canon_albedo_show.clamp(0,1).numpy().transpose(1,2,0)
        sub_canon_albedo_show = np.uint8(sub_canon_albedo_show[...,::-1]*255.)
        cv2.imwrite(os.path.join(albedo_dir,f'low_albedo.jpg'), sub_canon_albedo_show)

        #* whole depth
        super_depth1 = ((super_canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().numpy()[1]
        np.save(os.path.join(depth_dir, f'super_canon_depth.npy'), super_depth1)

        mid_depth1 = ((canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().numpy()[1]
        np.save(os.path.join(depth_dir, f'mid_canon_depth.npy'), mid_depth1)

        low_depth1 = ((sub_canon_depth - self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().numpy()[1]
        np.save(os.path.join(depth_dir, f'low_canon_depth.npy'), low_depth1)

        #* whole normal 
        super_normal1 = super_canon_normal_show.numpy()[0]
        np.save(os.path.join(normal_dir, f'super_canon_normal.npy'), super_normal1)

        mid_normal1 = canon_normal_show.numpy()[0]
        np.save(os.path.join(normal_dir, f'mid_canon_normal.npy'), mid_normal1)

        low_normal1 = sub_canon_normal_show.numpy()[0]
        np.save(os.path.join(normal_dir, f'low_canon_normal.npy'), low_normal1)
        
        def save_img_all(dir, depth, normal, light):
            depth = depth.clamp(0,1).numpy().transpose(0,2,3,1)[0]
            depth = np.uint8(depth[...,::-1]*255.)
            cv2.imwrite(os.path.join(dir,'depth'+'.jpg'), depth)         

            normal = normal/2.+0.5
            normal = normal.clamp(0,1).numpy()[0]
            normal = np.uint8(normal[...,::-1]*255.)
            cv2.imwrite(os.path.join(dir, 'normal'+'.jpg'), normal)
            
            light = light.detach().cpu().clamp(0,1).numpy().transpose(0,2,3,1)[0]
            light = np.uint8(light[...,::-1]*255.)
            cv2.imwrite(os.path.join(dir, 'light'+'.jpg'), light) 

        save_img_all(super_dir, super_canon_depth_show, super_canon_normal_show, super_canon_shading_show)
        save_img_all(mid_dir, canon_depth_show, canon_normal_show, canon_shading_show)
        save_img_all(sub_dir, sub_canon_depth_show, sub_canon_normal_show, sub_canon_shading_show)          

        self.count += 1
        print(f'{self.count}, finish rendering {name[0]} \n')
        
        return True
        
 
    def forward_and_draw(self):

        super_depth_var = torch.cat(self.super_depth_var, dim=0)
        super_normal_var = torch.cat(self.super_normal_var, dim=0)
        depth_var = torch.cat(self.depth_var, dim=0)
        normal_var = torch.cat(self.normal_var, dim=0)
        sub_depth_var = torch.cat(self.sub_depth_var, dim=0)
        sub_normal_var = torch.cat(self.sub_normal_var, dim=0)

        super_depth_var = super_depth_var.cpu().numpy()
        super_normal_var = super_normal_var.cpu().numpy() 
        depth_var = depth_var.cpu().numpy()
        normal_var = normal_var.cpu().numpy() 
        sub_depth_var = sub_depth_var.cpu().numpy()
        sub_normal_var = sub_normal_var.cpu().numpy() 

        print('---------------------------------------------------------------')
        print('super_depth_var:',super_depth_var.mean())
        print('depth_var:',depth_var.mean())
        print('sub_depth_var:',sub_depth_var.mean())
        print('---------------------------------------------------------------')
        print('super_normal_var:',np.mean(super_normal_var[:,2]))
        print('normal_var:',np.mean(normal_var[:,2]))
        print('sub_normal_var:',np.mean(sub_normal_var[:,2]))
        print('---------------------------------------------------------------')
        return True

#* ########################################################### Capsule ##########################################################

class Translator_NN_depth_view(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh, cfgs=None):
        super(Translator_NN_depth_view, self).__init__()

        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)

        #* depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.border_depth = cfgs.get('border_depth', self.max_depth)

        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)

        #* upsampling
        network_dec = [
            nn.ConvTranspose2d(cin, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False), 
            
                      
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
                       
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network_dec += [activation()]
            
        self.network_dec = nn.Sequential(*network_dec)

        self.depth_scale = nn.Linear(cin, cin)
        nn.init.normal_(self.depth_scale.weight.data, 0.0, 0.01)
        nn.init.constant_(self.depth_scale.bias.data, 0.) 


    def forward(self, embed_list, global_flag = False, flip_flag=True):

        #* Estimate raw depth by CNN
        if global_flag:
            embed_list = self.depth_scale(embed_list)
            canon_depth_raw =  self.network_dec(embed_list.unsqueeze(-1).unsqueeze(-1))
            
            #* rescale
            canon_depth_raw = canon_depth_raw.squeeze(1)
            b, h, w = canon_depth_raw.shape
            canon_depth = canon_depth_raw - canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
            #* depth rescaler: -1~1 -> min_deph~max_deph
            canon_depth = canon_depth.tanh()
            canon_depth = self.depth_rescaler(canon_depth)
            
            depth_border = torch.zeros(1,h,w-8).to(canon_depth.device)
            depth_border = nn.functional.pad(depth_border, (4,4), mode='constant', value=1)            
            canon_depth = canon_depth*(1-depth_border) + depth_border *self.border_depth
            
            return canon_depth


        else:
            canon_depth_raw_list = []
            view_list= []
            embed_list = self.depth_scale(embed_list)
            embed_list = embed_list.split(dim=1, split_size=1)  #* split every capsule
            
            for i in range(len(embed_list)):
                depth_feat_part = embed_list[i].squeeze(1)
                depth_curr =  self.network_dec(depth_feat_part.unsqueeze(-1).unsqueeze(-1))
                canon_depth_raw_list.append(depth_curr) 

            
            #* refine raw depth to real depth
            b, _, h, w = canon_depth_raw_list[0].shape
            canon_depth_list = []
            for i in range(len(embed_list)):
                #* recale depth
                canon_depth_raw = canon_depth_raw_list[i].squeeze(1).squeeze(1)
                canon_depth = canon_depth_raw - canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
                canon_depth = canon_depth.tanh()
                #* depth rescaler: -1~1 -> min_deph~max_deph
                canon_depth = self.depth_rescaler(canon_depth)
                canon_depth_list.append(canon_depth.unsqueeze(1))
            
            canon_depth_concat = torch.cat(canon_depth_list, dim=1)

            #*  combine part depth into one
            min_ind = torch.argmin(canon_depth_concat, dim=1)            
            depth_one_hot = torch.zeros_like(canon_depth_concat, device=canon_depth_concat.device)
            depth_one_hot.scatter_(1, min_ind.unsqueeze(1), 1)
    
            canon_depth_mask = torch.softmax(-canon_depth_concat*1e3, dim=1)
            canon_depth_combine = (canon_depth_concat*canon_depth_mask).sum(1)

            if flip_flag:
                canon_depth_combine = torch.cat([canon_depth_combine, canon_depth_combine.flip(2)], 0)
                canon_depth_concat = torch.cat([canon_depth_concat, canon_depth_concat.flip(3)], 0)
                canon_depth_mask = torch.cat([canon_depth_mask, canon_depth_mask.flip(3)], 0)

            return canon_depth_combine, canon_depth_concat, depth_one_hot
    

class Translator_NN_albedo(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh, cfgs=None):
        super(Translator_NN_albedo, self).__init__()

        #* upsampling
        network_dec = [                     
            nn.ConvTranspose2d(cin, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),           
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),  # 4x4 -> 8x8
            nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
                        
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
 
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network_dec += [activation()]
            
        self.network_dec = nn.Sequential(*network_dec)

        self.albedo_scale = nn.Linear(cin, cin, bias=False)
        nn.init.normal_(self.albedo_scale.weight.data, 0.0, 0.01)        

    def forward(self, caps_feat_list, depth_one_hot, global_flag = False):

        if global_flag:
            caps_feat = self.albedo_scale(caps_feat_list)
            canon_albedo =  self.network_dec(caps_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1))
            return canon_albedo
        
        else:
            caps_feat_list = self.albedo_scale(caps_feat_list)

            canon_albedo_list = []
            caps_feat_list = caps_feat_list.split(dim=1, split_size=1)

            for i in range(len(caps_feat_list)):
                albedo_curr =  self.network_dec(caps_feat_list[i].squeeze(1).unsqueeze(-1).unsqueeze(-1))
                canon_albedo_list.append(albedo_curr.unsqueeze(1)) 

            canon_albedo_concat = torch.cat(canon_albedo_list, dim=1)        
            albedo_one_hot = torch.repeat_interleave(depth_one_hot.unsqueeze(2), repeats=3, dim=2)         
            canon_albedo_list = canon_albedo_concat*albedo_one_hot       
            canon_albedo_combine = canon_albedo_list.sum(1)

            canon_albedo = torch.cat([canon_albedo_combine, canon_albedo_combine.flip(3)], 0) 
            albedo_one_hot = torch.cat([albedo_one_hot, albedo_one_hot.flip(4)], 0) 

            return canon_albedo, canon_albedo_list, albedo_one_hot, canon_albedo_concat

class Translator_NN_light(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh, cfgs=None):
        super(Translator_NN_light, self).__init__()
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

        network = [
            nn.Linear(cin, nf),
            nn.Linear(nf, cout)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

        for m in self.network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
                nn.init.constant_(m.bias.data, 0.)        

    def forward(self, input):
        b = input.shape[0]
        canon_light = self.network(input).reshape(input.size(0),-1)
        canon_light = canon_light.repeat(2,1)  
        canon_light_a = self.amb_light_rescaler(canon_light[:,:1])  #* ambience term
        canon_light_b = self.diff_light_rescaler(canon_light[:,1:2])  #* diffuse term
        canon_light_dxy = canon_light[:,2:]
        canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(input.device)], 1)
        canon_light_d = canon_light_d / ((canon_light_d**2).sum(1, keepdim=True))**0.5  #* diffuse light direction
        
        return canon_light_a, canon_light_b, canon_light_d 


class Translator_NN_caps(nn.Module):
    def __init__(self, cin, cout, num_caps, num_each_caps):
        super(Translator_NN_caps, self).__init__()

        self.caps_basis = nn.Parameter(torch.ones((1, num_caps, num_each_caps, cout)), requires_grad=True)
        nn.init.kaiming_normal_(self.caps_basis)
        network =  [
            nn.Linear(cin, cout),
            ]
        self.network = nn.Sequential(*network)

        for m in self.network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
                nn.init.constant_(m.bias.data, 0.) 
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_caps = num_caps
        self.num_each_caps = num_each_caps
    
    def forward(self,feat_list, layer=-2,depth_dim=384,light_dim=48):
        x = feat_list[layer]
        B, N,_,_,_ = x.shape  
       
        x = x.reshape(B,N*N,self.num_each_caps,-1)
        caps_feat = self.network(x)      
        
        caps_attn_map = caps_feat*self.caps_basis
        caps_attn_final = torch.softmax(caps_attn_map,dim=1)  #* softmax among all capsules
        ind_onehot = torch.argmax(caps_attn_final, dim=1)  #* only the biggest capsule is activated 
        attn_onehot = torch.zeros_like(caps_attn_final, device=caps_attn_final.device) 
        attn_onehot.scatter_(1, ind_onehot.unsqueeze(1), 1)  #* one-hot embedding
        
        #* vgg
        attn_tmp1 = attn_onehot[:,:,:,:depth_dim]
        attn_tmp2 = attn_onehot[:,:,:,depth_dim*2:depth_dim*2+light_dim]*0. +1./self.num_caps

        attn_tmp = torch.cat((attn_tmp1, attn_tmp1, attn_tmp2,attn_tmp2),dim=-1)
        output = (caps_attn_final*caps_feat*attn_tmp).sum(dim=2)        

        return output, caps_attn_map.squeeze(dim=2)



class Translator_NN_View(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh, cfgs=None):
        super(Translator_NN_View, self).__init__()

        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 90)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)

        network = [
            nn.Linear(cin, nf),
            nn.Linear(nf, cout)
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

        for m in self.network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
                nn.init.constant_(m.bias.data, 0.)  

    def forward(self, input, global_flag = False):
        if global_flag:
           views = self.network(input).repeat(2,1)
        else:
           views = self.network(input).mean(1).repeat(2,1)    
        views = torch.cat([
            views[:,:3] *math.pi/180 *self.xyz_rotation_range,
            views[:,3:5] *self.xy_translation_range,
            views[:,5:] *self.z_translation_range], 1)        

        return views


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()

        #* downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        
        #* upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)