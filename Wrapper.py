import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import cv2
import torchvision
from torch.optim import AdamW
import torch.nn.functional as F
import csv
from torch.optim.lr_scheduler import ExponentialLR

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used : {device}")
np.random.seed(0)

def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """

    transf_path = ""

    if mode == "train":
        transf_path = "transforms_train.json"
    else:
        transf_path = "transforms_test.json"


    folder_path = os.path.join(data_path,mode)
    

    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    with open(os.path.join(data_path,transf_path), "r") as file:
        data = json.load(file)
    
    H, W, _ = images[0].shape

    fov = data["camera_angle_x"]

    f = W / (2*np.tan(fov/2))

    K = np.array([
        [f, 0, W / 2],  # fx, 0, cx
        [0, f, H / 2],  # 0, fy, cy
        [0, 0, 1]  # 0, 0, 1
    ])

    poses = []

    for value in data["frames"]:

        tranf_matrix = np.empty((4,4))

        for row in value["transform_matrix"]:

            tranf_matrix = np.vstack((tranf_matrix,row))

        poses.append(tranf_matrix[4:,:])

    return images, poses, [W, H, K]




def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """

    H, W, K = camera_info

    # Extract rotation matrix (3x3) and translation vector (3x1)
    R = pose[:3, :3]  # Rotation
    t = pose[:3, 3]   # Translation

    # Compute camera origin in world coordinates
    ray_origin = -np.dot(R.T, t)

    
    # Compute ray direction
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Convert pixel to normalized camera coordinates
    u,v  = pixelPosition
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_dir_camera = np.array([x, y, 1.0])  # Direction in camera space

    # Compute world ray direction
    ray_dir_world = R @ ray_dir_camera
    ray_dir_world /= np.linalg.norm(ray_dir_world)  # Normalize

    
    near = 1.0
    
    # Compute normalized ray origin
    t = -(near + ray_origin[2]) / ray_dir_world[2]  
    rays_o = ray_origin + t * ray_dir_world             # Move origin forward
    
    # Apply perspective transformation
    o0 = (rays_o[0] / rays_o[2]) * -fx / (W / 2)
    o1 = (rays_o[1] / rays_o[2]) * -fy / (H / 2)
    o2 = 1 + 2 * near / rays_o[2]  

    d0 = (ray_dir_world[0] / ray_dir_world[2] - rays_o[0] / rays_o[2])* -fx / (W / 2)
    d1 = (ray_dir_world[1] / ray_dir_world[2] - rays_o[1] / rays_o[2])* -fy / (H / 2)
    d2 = -2 * near / rays_o[2]  

    
    ndc_ray_o = [o0, o1, o2]
    ndc_ray_d = [d0, d1, d2]

    return np.array(ndc_ray_o), np.array(ndc_ray_d)


    
    

def findColorandWeights(rgb_values, sigma_values, samples):

    """
    Compute the color given RGB values, color density values and ray sample locations

    Input:
        rgb_values: RGB value for every ray sample 
        sigma_values: color density value for every sample
        samples: sample locations along a ray
    Outputs:
        color value for pixel and weights(for hierarchical sampling)    


    """

    color_value = 0
    color_weights = []


    for i in range(len(samples)):

        t_i_1 = 1 if i==len(samples)-1 else samples[i+1]

        delta_i = t_i_1 - samples[i]
        sigma_i = sigma_values[i]

        sum = 0
        for j in range(1,i-1):
            sum+= (samples[j+1] - samples[j])*(sigma_values[j])

        T_i = torch.exp(torch.tensor(-sum))
        # sum_t = torch.tensor(-sum).clone().detach()
        # T_i = torch.exp(-sum)

        w_i = T_i*(1 - torch.exp(-delta_i*sigma_i))
        color_weights.append(w_i)

        color_value+= w_i*rgb_values[i]

    return color_value, color_weights



def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

    H, W, K = camera_info

    ray_origins = []
    ray_directions = []
    pixel_colors = []

    for j in range(args.n_rays_batch):

        img_index = np.random.randint(1,len(images))

        h_index = np.random.randint(1,H)
        w_index = np.random.randint(1,W)

        resized_image = cv2.resize(images[img_index], (400, 400))
        pixel_colors.append(resized_image[w_index,h_index,:])

        random_ray_o, random_ray_d = PixelToRay(camera_info,poses[img_index],[w_index,h_index],args)

        ray_origins.append(random_ray_o)
        ray_directions.append(random_ray_d)

    return np.array(ray_origins), np.array(ray_directions), np.array(pixel_colors)



def generateTestBatch(testIndex, pose, camera_info, args):
    """
    Input:
        testIndex: test image index
        poses: corresponding camera poses in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

    H, W, K = camera_info

    ray_origins = []
    ray_directions = []
    pixel_colors = []


    for u in range(W):

        for v in range(H):

            random_ray_o, random_ray_d = PixelToRay(camera_info,pose[testIndex],[u,v],args)

            ray_origins.append(random_ray_o)
            ray_directions.append(random_ray_d)

    return np.array(ray_origins), np.array(ray_directions), np.array(pixel_colors)




def stratified_sampling(t_near,t_far,Nc):

    """
    Perform stratified sampling along a ray

    Input:
        t_near: nearest sample
        t_far: furthest sample
        Nc: Number of samples
    Outputs:
        depth(t) of the samples
    
    """

    values = []
    bins = [t_near]

    for k in range(1,Nc+1):

        lower_limit = t_near + (((k-1)/Nc)*(t_far - t_near))
        uppper_limit = t_near + ((k/Nc)*(t_far - t_near))

        bins.append(uppper_limit)

        val = np.random.uniform(lower_limit,uppper_limit)

        values.append(val)

    return values, bins



def hierarchical_sample(bins, weights, N_fine):
    """
    Perform hierarchical volume sampling based on computed weights.

    Input:
        bins: bins constructed along the ray which aids in sampling
        weights: weights computed during stratified sampling 
        N_fine: Number of samples
    Outputs:
        depth(t) of the samples    
    
    """

    mod_weights = [wht.item() for wht in weights]
    # print(len(mod_weights))

    # Normalize weights to get a valid PDF
    pdf = np.array(mod_weights) + 1e-5  # Prevent div by zero
    pdf = pdf / np.sum(pdf)  # Normalize to sum to 1

    # Convert PDF to CDF
    cdf = np.cumsum(pdf, axis=0)  # Compute CDF
    cdf = np.concatenate([np.zeros(1), cdf])  # Add zero at the beginning

    # Sample new points using inverse transform sampling
    u = np.random.rand(N_fine)  # Generate uniform random samples in [0,1]
    fine_samples = np.interp(u, cdf, bins)  # Map uniform samples to t values

    return fine_samples




def getDepthsFromRaySamples(ray_origin,ray_dir,samples):

    """
    Ouputs the 3D coordinate of the sample given sample, ray origin and direction

    Input:
        ray_origin: ray origin coordinates 
        ray_dir: ray direction vector
        samples: array of depth corresponding to each sample
    Outputs:
        3D coordinate of the sample
    """

    depths = []

    for sample in samples:

        x = ray_origin + (sample*ray_dir)

        depths.append(torch.tensor(x))

    return depths




def loss(groundtruth, prediction):
    """
    Computes loss using the coarse and fine network outputs

    Input:
        groundtruth: pixel color values
        prediction: coarse and fine network predictions
    Outputs:
        total MSE loss of both networks

    """

    coarse_pred, fine_pred = prediction

    loss_coarse = F.mse_loss(coarse_pred, groundtruth)  # Coarse loss
    loss_fine = F.mse_loss(fine_pred, groundtruth)      # Fine loss

    total_loss = loss_coarse + loss_fine  # Sum both losses
    return total_loss


def train(images, poses, camera_info, args):
    """
    Main training function
    """

    # Initialise coarse and fine MLP networks
    coarse_network = NeRFmodel(60,24).to(device)
    fine_network = NeRFmodel(60,24).to(device)

    optimizer = torch.optim.AdamW(
    list(coarse_network.parameters()) + list(fine_network.parameters()),
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-7
    )   

    # Learning rate scheduler for decaying LR
    scheduler = ExponentialLR(optimizer, gamma=0.1**(1/args.max_iters))

    # Initialize additional parameters
    Nc = 64     # coarse network samples
    Nf = 128    # fine network samples
    L_p = 10    # positional encoding length
    L_d = 4     # directional encoding length

    H, W, K = camera_info
    camera_info = [400, 400, K]
    H = 400
    W = 400

    # Initialize CSV file for logging
    csv_file = "training_logs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Iteration', 'Training Loss'])


    for i in range(args.max_iters):

        # get batch of rays from all train images
        ray_origins, ray_dirs, pixel_colors = generateBatch(images,poses,camera_info,args)
        
        loss_per_epoch = []

        print(f"Processing batch {i+1} ...")

        pos_enc_coarse = np.zeros((1,60))
        dir_enc_coarse = np.zeros((1,24))
        dir_enc_vals = np.zeros((1,24))

        pos_enc_hier = np.zeros((1,60))
        dir_enc_hier = np.zeros((1,24))

        stratified_samples = np.zeros(Nc)
        hierarchical_samples = np.zeros(Nf)
        ray_bins = np.zeros(Nc+1)

        all_colors_coarse = []
        all_colors_fine = []

        optimizer.zero_grad(set_to_none=True)


        for idx in range(len(ray_origins)):

            stratified_sampling_values, bins = stratified_sampling(-1,1,Nc)

            stratified_samples = np.vstack((stratified_samples, np.array(stratified_sampling_values)))
            ray_bins = np.vstack((ray_bins, np.array(bins)))

            x_stratified = getDepthsFromRaySamples(ray_origins[idx],ray_dirs[idx],stratified_sampling_values)

            x_stratified_torch = torch.stack(x_stratified).to(device, non_blocking=True)
            # print(f"stratified depths: {x_stratified_torch}")

            
            pos_enc = np.zeros((len(x_stratified),1))        # position encoding
            dir_enc = np.zeros((1,1))        # direction encoding

            # perform positional encoding
            for dummy in range(3):
                encoding1 = coarse_network.position_encoding(x_stratified_torch[:,dummy],10)
                pos_enc = np.hstack((pos_enc,encoding1))

                encoding2 = coarse_network.position_encoding([torch.tensor(ray_dirs[idx,dummy],dtype=torch.float32)],4)
                dir_enc = np.hstack((dir_enc,encoding2))

            pos_enc = pos_enc[:,1:]
            dir_enc = dir_enc[:,1:]


            dir_enc_vals = np.vstack((dir_enc_vals, dir_enc))

            dir_enc = np.broadcast_to(dir_enc,(len(x_stratified),dir_enc.shape[1]))


            pos_enc_coarse  = np.vstack((pos_enc_coarse,pos_enc))
            dir_enc_coarse = np.vstack((dir_enc_coarse,dir_enc))
           


        pos_enc_coarse = pos_enc_coarse[1:,:]
        dir_enc_coarse = dir_enc_coarse[1:,:]

        pos_enc_coarse = np.reshape(pos_enc_coarse,(len(ray_origins),Nc,L_p*2*3))
        dir_enc_coarse = np.reshape(dir_enc_coarse,(len(ray_origins),Nc,L_d*2*3))

        pos_enc_t = torch.from_numpy(pos_enc_coarse).to(dtype=torch.float32, device=device, non_blocking=True)
        dir_enc_coarse = np.array(dir_enc_coarse, copy=True)
        dir_enc_t = torch.from_numpy(dir_enc_coarse).to(dtype=torch.float32, device=device, non_blocking=True)


        print("Passing through coarse network ...")

        rgb_coarse, sigma_coarse = coarse_network(pos_enc_t,dir_enc_t)
        
        stratified_samples = stratified_samples[1:,:]
        ray_bins = ray_bins[1:,:]

        for idx in range(len(ray_origins)):

            colors_coarse, weights_coarse = findColorandWeights(rgb_coarse[idx],sigma_coarse[idx],stratified_samples[idx,:])

            all_colors_coarse.append(colors_coarse)

            x_fine_samples = hierarchical_sample(ray_bins[idx,:], weights_coarse, Nf)

            hierarchical_samples = np.vstack((hierarchical_samples,x_fine_samples))

            x_fine = getDepthsFromRaySamples(ray_origins[idx],ray_dirs[idx],x_fine_samples)

            # x_new = x_stratified + x_fine

            x_fine_torch = torch.stack(x_fine).to(device, non_blocking=True)

            pos_enc_fine = np.zeros((len(x_fine),1))

            for dum in range(3):
                encoding3 = fine_network.position_encoding(x_fine_torch[:,dum],10)
                pos_enc_fine = np.hstack((pos_enc_fine,encoding3))

            pos_enc_fine = pos_enc_fine[:,1:]

            pos_enc_hier = np.vstack((pos_enc_hier, pos_enc_fine))

            dir_enc2 = dir_enc_vals[idx,:]
            dir_enc_fine = np.broadcast_to(dir_enc2,(Nf,dir_enc.shape[1]))

            dir_enc_hier = np.vstack((dir_enc_hier, dir_enc_fine))

            
        pos_enc_hier = pos_enc_hier[1:,:]
        dir_enc_hier = dir_enc_hier[1:,:]

        pos_enc_hier = np.reshape(pos_enc_hier,(len(ray_origins),Nf,L_p*2*3))
        dir_enc_hier = np.reshape(dir_enc_hier,(len(ray_origins),Nf,L_d*2*3))

        pos_enc_final = np.hstack((pos_enc_coarse, pos_enc_hier))
        dir_enc_final = np.hstack((dir_enc_coarse, dir_enc_hier))

        
        pos_enc_f = torch.from_numpy(pos_enc_final).to(dtype=torch.float32, device=device, non_blocking=True)
        dir_enc_fine = np.array(dir_enc_fine, copy=True)
        dir_enc_f = torch.from_numpy(dir_enc_final).to(dtype=torch.float32, device=device, non_blocking=True)

        print("Passing through fine network ...")
        rgb_final, sigma_final = fine_network(pos_enc_f, dir_enc_f)

        hierarchical_samples = hierarchical_samples[1:,:]

        combined_samples = np.hstack((stratified_samples, hierarchical_samples))


        for idx in range(len(ray_origins)):
            colors_final, _ = findColorandWeights(rgb_final[idx],sigma_final[idx],combined_samples[idx,:])
            all_colors_fine.append(colors_final)


        pix_colors = torch.from_numpy(pixel_colors).to(dtype=torch.float32, device=device, non_blocking=True)
        colors_coarse = torch.stack(all_colors_coarse).to(device=device, non_blocking=True)
        colors_final = torch.stack(all_colors_fine).to(device=device, non_blocking=True)

        loss_fin = loss(pix_colors,[colors_coarse,colors_final])
        

        
        loss_fin.backward()
        optimizer.step()
        scheduler.step()
        


        print(f"Iteration:[{i}] loss:[{loss_fin.item()}]")

        with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, loss_fin.item()])

        
        if i%10 == 0:
            # Save the Model learnt
            SaveName1 = (
                args.checkpoint_path
                + str(i)
                + "coarse_model.ckpt"
            )
            SaveName2 = (
                args.checkpoint_path
                + str(i)
                + "fine_model.ckpt"
            )

            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": coarse_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_fin,
                },
                SaveName1,
            )
            print("\n" + SaveName1 + " Model Saved...")

            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": fine_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_fin,
                },
                SaveName2,
            )
            print("\n" + SaveName2 + " Model Saved...")

            

def loadTestData(data_path):
    """
    Load relevant data for testing and process it
    """

    folder_path = os.path.join(data_path,"test")

    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    with open(os.path.join(data_path,"transforms_test.json"), "r") as file:
        data = json.load(file)
    
    H, W, _ = images[0].shape

    fov = data["camera_angle_x"]

    f = W / (2*np.tan(fov/2))

    K = np.array([
        [f, 0, W / 2],  # fx, 0, cx
        [0, f, H / 2],  # 0, fy, cy
        [0, 0, 1]  # 0, 0, 1
    ])

    poses = []

    for value in data["frames"]:

        tranf_matrix = np.empty((4,4))

        for row in value["transform_matrix"]:

            tranf_matrix = np.vstack((tranf_matrix,row))

        poses.append(tranf_matrix[4:,:])

    return poses, [W, H, K]





def test(images, poses, camera_info, args):
    """
    Test the trained model
    """

    coarse_network = NeRFmodel(60,24).to(device)
    fine_network = NeRFmodel(60,24).to(device)

    optimizer = torch.optim.AdamW(
    list(coarse_network.parameters()) + list(fine_network.parameters()),
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-7
    )   

    scheduler = ExponentialLR(optimizer, gamma=0.1**(1/args.max_iters))

    # Initialize additional parameters
    Nc = 64     # coarse network samples
    Nf = 128    # fine network samples
    L_p = 10    # positional encoding length
    L_d = 4     # directional encoding length

    H, W, K = camera_info
    camera_info = [400, 400, K]
    H = 400
    W = 400
    
    coarse_model_path = os.path.join(args.checkpoint_path,"coarse_model.ckpt")
    fine_model_path = os.path.join(args.checkpoint_path,"fine_model.ckpt")

    # Load saved model
    CheckPoint1 = torch.load(coarse_model_path)
    coarse_network.load_state_dict(CheckPoint1['model_state_dict'])

    CheckPoint2 = torch.load(fine_model_path)
    fine_network.load_state_dict(CheckPoint2['model_state_dict'])

    for im in range(len(images)):

        ray_origins, ray_dirs, pixel_colors = generateTestBatch(im,poses,camera_info,args)
        
        loss_per_epoch = []

        print(f"Processing image {im+1} ...")

        pos_enc_coarse = np.zeros((1,60))
        dir_enc_coarse = np.zeros((1,24))
        dir_enc_vals = np.zeros((1,24))

        pos_enc_hier = np.zeros((1,60))
        dir_enc_hier = np.zeros((1,24))

        stratified_samples = np.zeros(Nc)
        hierarchical_samples = np.zeros(Nf)
        ray_bins = np.zeros(Nc+1)

        all_colors_coarse = []
        all_colors_fine = []

        optimizer.zero_grad(set_to_none=True)


        for idx in range(len(ray_origins)):

            stratified_sampling_values, bins = stratified_sampling(-1,1,Nc)

            stratified_samples = np.vstack((stratified_samples, np.array(stratified_sampling_values)))
            ray_bins = np.vstack((ray_bins, np.array(bins)))

            x_stratified = getDepthsFromRaySamples(ray_origins[idx],ray_dirs[idx],stratified_sampling_values)

            x_stratified_torch = torch.stack(x_stratified).to(device, non_blocking=True)
            
            pos_enc = np.zeros((len(x_stratified),1))        # position encoding
            dir_enc = np.zeros((1,1))        # direction encoding

            for dummy in range(3):
                encoding1 = coarse_network.position_encoding(x_stratified_torch[:,dummy],10)
                pos_enc = np.hstack((pos_enc,encoding1))

                encoding2 = coarse_network.position_encoding([torch.tensor(ray_dirs[idx,dummy],dtype=torch.float32)],4)
                dir_enc = np.hstack((dir_enc,encoding2))

            pos_enc = pos_enc[:,1:]
            dir_enc = dir_enc[:,1:]


            dir_enc_vals = np.vstack((dir_enc_vals, dir_enc))

            dir_enc = np.broadcast_to(dir_enc,(len(x_stratified),dir_enc.shape[1]))


            pos_enc_coarse  = np.vstack((pos_enc_coarse,pos_enc))
            dir_enc_coarse = np.vstack((dir_enc_coarse,dir_enc))
            


        pos_enc_coarse = pos_enc_coarse[1:,:]
        dir_enc_coarse = dir_enc_coarse[1:,:]

        pos_enc_coarse = np.reshape(pos_enc_coarse,(len(ray_origins),Nc,L_p*2*3))
        dir_enc_coarse = np.reshape(dir_enc_coarse,(len(ray_origins),Nc,L_d*2*3))

        pos_enc_t = torch.from_numpy(pos_enc_coarse).to(dtype=torch.float32, device=device, non_blocking=True)
        dir_enc_coarse = np.array(dir_enc_coarse, copy=True)
        dir_enc_t = torch.from_numpy(dir_enc_coarse).to(dtype=torch.float32, device=device, non_blocking=True)

        print("Passing through coarse network ...")

        rgb_coarse, sigma_coarse = coarse_network(pos_enc_t,dir_enc_t)
        stratified_samples = stratified_samples[1:,:]
        ray_bins = ray_bins[1:,:]

        for idx in range(len(ray_origins)):

            colors_coarse, weights_coarse = findColorandWeights(rgb_coarse[idx],sigma_coarse[idx],stratified_samples[idx,:])

            all_colors_coarse.append(colors_coarse)

            x_fine_samples = hierarchical_sample(ray_bins[idx,:], weights_coarse, Nf)

            hierarchical_samples = np.vstack((hierarchical_samples,x_fine_samples))

            x_fine = getDepthsFromRaySamples(ray_origins[idx],ray_dirs[idx],x_fine_samples)

            x_fine_torch = torch.stack(x_fine).to(device, non_blocking=True)

            pos_enc_fine = np.zeros((len(x_fine),1))

            for dum in range(3):
                encoding3 = fine_network.position_encoding(x_fine_torch[:,dum],10)
                pos_enc_fine = np.hstack((pos_enc_fine,encoding3))

            pos_enc_fine = pos_enc_fine[:,1:]

            pos_enc_hier = np.vstack((pos_enc_hier, pos_enc_fine))

            dir_enc2 = dir_enc_vals[idx,:]
            dir_enc_fine = np.broadcast_to(dir_enc2,(Nf,dir_enc.shape[1]))

            dir_enc_hier = np.vstack((dir_enc_hier, dir_enc_fine))

        
        pos_enc_hier = pos_enc_hier[1:,:]
        dir_enc_hier = dir_enc_hier[1:,:]

        pos_enc_hier = np.reshape(pos_enc_hier,(len(ray_origins),Nf,L_p*2*3))
        dir_enc_hier = np.reshape(dir_enc_hier,(len(ray_origins),Nf,L_d*2*3))

        pos_enc_final = np.hstack((pos_enc_coarse, pos_enc_hier))
        dir_enc_final = np.hstack((dir_enc_coarse, dir_enc_hier))


        pos_enc_f = torch.from_numpy(pos_enc_final).to(dtype=torch.float32, device=device, non_blocking=True)
        dir_enc_fine = np.array(dir_enc_fine, copy=True)
        dir_enc_f = torch.from_numpy(dir_enc_final).to(dtype=torch.float32, device=device, non_blocking=True)

        print("Passing through fine network ...")
        rgb_final, sigma_final = fine_network(pos_enc_f, dir_enc_f)

        hierarchical_samples = hierarchical_samples[1:,:]

        combined_samples = np.hstack((stratified_samples, hierarchical_samples))


        for idx in range(len(ray_origins)):
            colors_final, _ = findColorandWeights(rgb_final[idx],sigma_final[idx],combined_samples[idx,:])
            all_colors_fine.append(colors_final)


        pix_colors = torch.from_numpy(pixel_colors).to(dtype=torch.float32, device=device, non_blocking=True)
        colors_coarse = torch.stack(all_colors_coarse).to(device=device, non_blocking=True)
        colors_final = torch.stack(all_colors_fine).to(device=device, non_blocking=True)

        pixel_values_final = torch.cat(all_colors_fine, dim=0).cpu().detach().numpy().reshape(H, W, 3)
        rendered_image = (pixel_values_final * 255).astype(np.uint8)  # Convert to 0-255 range
        cv2.imwrite(f"rendered_final_image_{im}.png",rendered_image)





def main(args):
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)


    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./nerf_synthetic/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=1024,help="number of rays per batch")
    parser.add_argument('--n_sample',default=400,help="number of sample per ray")
    parser.add_argument('--max_iters',default=100000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)