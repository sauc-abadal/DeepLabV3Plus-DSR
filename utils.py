import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torchvision
import os

def label2class():
    classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled'     
    ]
    return classes

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
    
            img_color[row, col] = np.array(label_to_color[label])
    
    img_color = img_color.astype("uint8") ### in order to use matplotlib in the range [0, 255] and avoid clipping
    return img_color

def imshow(img, ax):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    npimg = np.clip(npimg, 0, 255).astype(np.uint8)
    ax.imshow(npimg)
    plt.pause(0.001)
    
def unnormalize(img, mean, std):
    img_un = (img * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)) * 255.0 
    return img_un

def visualize_example(data):
    img = data["img"]
    gt_sisr = data["gt_sisr"]
    gt_sssr = data["gt_sssr"]
    
    mean = data["mean"]
    std = data["std"]
    
    img_un = unnormalize(img, mean, std)
    gt_sisr_un = unnormalize(gt_sisr, mean, std)

    imgs = torchvision.utils.make_grid(img_un) # grid from the 2 images of the minibatch
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    imshow(imgs, ax1) 
    
    gt_SISR = torchvision.utils.make_grid(gt_sisr_un)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    imshow(gt_SISR, ax2) 
    
    # SSSR Ground Truth in gray scale + colormap
    #gt_SSSR = torch.unsqueeze(gt_sssr, dim=1) ### add the channel dimension (gray-scale image)
    #gt_SSSR = torchvision.utils.make_grid(gt_SSSR) ### when making the grid, torchvision sets the number of channels to 3 instead of 1 
    ### if the gray-scale image (labels) is set to have 3 channels, the image is seen darker since no color-map is applied
    #gt_SSSR = gt_SSSR[0,:,:].unsqueeze(dim=0) ### get only the first channel and unsqueeze the first dimension
    #fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    #imshow(gt_SSSR, ax3, normalized=False) 
    
    # SSSR Ground Truth in colors
    b, nrows, ncols = gt_sssr.shape
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    
    gt_sssr_color = np.zeros((b, nrows, ncols, 3))
    for b_ in range(b):
        gt_sssr_color[b_] = label_img_to_color(gt_sssr[b_].numpy()) 
    gt_SSSR_color = torch.from_numpy(gt_sssr_color)
    gt_SSSR_color = gt_SSSR_color.permute((0, 3, 1, 2)) 
    gt_SSSR_color = torchvision.utils.make_grid(gt_SSSR_color) 
    imshow(gt_SSSR_color, ax3)
    
def visualize_results(data, sssr, sisr, sssr_FA):
    img = data["img"]
    gt_sisr = data["gt_sisr"]
    gt_sssr = data["gt_sssr"]
    
    mean = data["mean"]
    std = data["std"]
    
    img_un = unnormalize(img, mean, std)
    gt_sisr_un = unnormalize(gt_sisr, mean, std)
    sisr_un = unnormalize(sisr, mean, std)
    sssr_FA_un = unnormalize(sssr_FA, mean, std)
    
    ############################################################################
    ###################### SISR BRANCH RESULTS #################################
    ############################################################################
    # SISR Ground Truth
    gt_SISR = torchvision.utils.make_grid(gt_sisr_un)
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    imshow(gt_SISR, ax1)
    # SISR Reconstruction
    SISR = torchvision.utils.make_grid(sisr_un)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
    imshow(SISR.cpu().detach(), ax2)
    
    SSSR_FA = torchvision.utils.make_grid(sssr_FA_un)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
    imshow(SSSR_FA.cpu().detach(), ax2)

    ############################################################################
    ###################### SSSR BRANCH RESULTS #################################
    ############################################################################
    # SSSR Ground Truth
    b, nrows, ncols = gt_sssr.shape
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    sssr_preds = torch.argmax(sssr, dim=1)
    
    gt_sssr_color = np.zeros((b, nrows, ncols, 3))
    sssr_color = np.zeros((b, nrows, ncols, 3))
    for b_ in range(b):
        gt_sssr_color[b_] = label_img_to_color(gt_sssr[b_].numpy())   
        # SSSR Predicted mask
        sssr_color[b_] = label_img_to_color(sssr_preds[b_].cpu().numpy())
        
    gt_SSSR_color = torch.from_numpy(gt_sssr_color)
    gt_SSSR_color = gt_SSSR_color.permute((0, 3, 1, 2)) 
    gt_SSSR_color = torchvision.utils.make_grid(gt_SSSR_color)
    imshow(gt_SSSR_color, ax3)
    
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    SSSR_color = torch.from_numpy(sssr_color)
    SSSR_color = SSSR_color.permute((0, 3, 1, 2)) 
    SSSR_color = torchvision.utils.make_grid(SSSR_color)
    imshow(SSSR_color, ax3)
    
def visualize_results_SISR(data, sisr):  
    img = data["img"]
    gt_sisr = data["gt_sisr"]
    gt_sssr = data["gt_sssr"]
    
    mean = data["mean"]
    std = data["std"]
    
    img_un = unnormalize(img, mean, std)
    gt_sisr_un = unnormalize(gt_sisr, mean, std)
    sisr_un = unnormalize(sisr, mean, std)
    
    ############################################################################
    ###################### SISR BRANCH RESULTS #################################
    ############################################################################
    # SISR Ground Truth
    gt_SISR = torchvision.utils.make_grid(gt_sisr_un)
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    imshow(gt_SISR, ax1)
    # SISR Reconstruction
    SISR = torchvision.utils.make_grid(sisr_un)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
    imshow(SISR.cpu().detach(), ax2)

def visualize_results_SSSR(gt_sssr, sssr):    
    ############################################################################
    ###################### SSSR BRANCH RESULTS #################################
    ############################################################################
    # SSSR Ground Truth
    #gt_sssr = data["gt_sssr"]
    b, nrows, ncols = gt_sssr.shape
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    sssr_preds = torch.argmax(sssr, dim=1)
    
    gt_sssr_color = np.zeros((b, nrows, ncols, 3))
    sssr_color = np.zeros((b, nrows, ncols, 3))
    for b_ in range(b):
        gt_sssr_color[b_] = label_img_to_color(gt_sssr[b_].numpy())   
        # SSSR Predicted mask
        sssr_color[b_] = label_img_to_color(sssr_preds[b_].cpu().numpy())
        
    gt_SSSR_color = torch.from_numpy(gt_sssr_color)
    gt_SSSR_color = gt_SSSR_color.permute((0, 3, 1, 2)) 
    gt_SSSR_color = torchvision.utils.make_grid(gt_SSSR_color)
    imshow(gt_SSSR_color, ax1)
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5)) 
    SSSR_color = torch.from_numpy(sssr_color)
    SSSR_color = SSSR_color.permute((0, 3, 1, 2)) 
    SSSR_color = torchvision.utils.make_grid(SSSR_color)
    imshow(SSSR_color, ax1)
    
    
    
def similarity_matrix(f, train=True):
    # f expected shape (Bs, C', H', W')
    if train:
        f = F.interpolate(f, size=(f.shape[2] // 8, f.shape[3] // 8), mode='nearest')   # before computing the relationship of every pair of pixels, 
                                                                                        # subsample the feature map to its 1/8
    else:
         f = F.interpolate(f, size=(f.shape[2] // 16, f.shape[3] // 16), mode='nearest')   # before computing the relationship of every pair of pixels, 
                                                                                        # subsample the feature map to its 1/16
    f = f.permute((0,2,3,1))
    f = torch.reshape(f, (f.shape[0], -1, f.shape[3])) # shape (Bs, H'xW', C')
    f_n = torch.linalg.norm(f, ord=None, dim=2).unsqueeze(-1) # ord=None indicates 2-Norm, 
                                                #unsqueeze last dimension to broadcast later
    eps = 1e-8
    f_norm = f / torch.max(f_n, eps * torch.ones_like(f_n))
    sim_mt = f_norm @ f_norm.transpose(2, 1)
    return sim_mt

def FA_Loss(sssr_sim_mt, sisr_sim_mt):
    nelem = sssr_sim_mt.shape[1] * sssr_sim_mt.shape[2]
    dist = torch.abs(sssr_sim_mt - sisr_sim_mt)
    l_fa = 1/nelem * torch.sum(dist, dim=[1, 2])
    return l_fa.mean()

def load_checkpoint(model, optimizer, scheduler, PATH):
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        train_iou = checkpoint['train_iou']
        valid_iou = checkpoint['valid_iou']
        train_rmse = checkpoint['train_rmse']
        valid_rmse = checkpoint['valid_rmse']
        print("=> loaded checkpoint '{}' (epoch {})".format(PATH, checkpoint['epoch']))     
    else:
        print("=> no checkpoint found at '{}'".format(PATH))
        start_epoch = 0
        train_loss = []
        valid_loss = []
        train_iou = []
        valid_iou = []
        train_rmse = []
        valid_rmse = []
    
    return model, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_iou, valid_iou, train_rmse, valid_rmse