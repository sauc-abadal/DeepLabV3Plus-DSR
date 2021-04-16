import dataset
import metrics
import utils

from deeplabv3 import DeepLabV3Plus
import encoders
from typing import Optional

import torch
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.autograd import Variable
import time
import copy
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

############################################################################################################################################
############# Load the Dataset #############################################################################################################
############################################################################################################################################
cityscapes_data_path = "/mnt/gpid07/datasets/cityscapes" 
cityscapes_meta_path = "/mnt/gpid07/imatge/sauc.abadal/datasets/cityscapes/meta"

train_dataset = dataset.DatasetTrain(cityscapes_data_path, cityscapes_meta_path)
val_dataset = dataset.DatasetVal(cityscapes_data_path, cityscapes_meta_path)

train_batch_size = 8
valid_batch_size = 2

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

num_train_batches = math.ceil(len(train_dataset)/train_batch_size)
num_val_batches = math.ceil(len(val_dataset)/valid_batch_size)

with open("/mnt/gpid07/imatge/sauc.abadal/datasets/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

############################################################################################################################################
############# METRICS ######################################################################################################################
############################################################################################################################################
mIoU = metrics.Jaccard()
psnr = metrics.PSNR()
rmse = metrics.RMSE()
ssim = metrics.SSIM()

############################################################################################################################################
########### LOSS AND OPTIMIZER #############################################################################################################
############################################################################################################################################
l_ce = nn.CrossEntropyLoss(weight=class_weights)
l_mse = nn.MSELoss()

# w1 and w2 set to make the loss value ranges comparable
w1 = 0.1
w2 = 1.0

def loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=True):
    sssr_sim_mt = utils.similarity_matrix(sssr_FA, train)
    sisr_sim_mt = utils.similarity_matrix(sisr.detach(), train)
    loss = l_ce(sssr, gt_sssr) + w1*l_mse(sisr, gt_sisr) + w2*utils.FA_Loss(sssr_sim_mt, sisr_sim_mt)
    return loss

model = DeepLabV3Plus(
            encoder_name = "resnet101",
            encoder_depth = 5,
            encoder_weights = "imagenet",
            encoder_output_stride = 16,
            decoder_channels = 128,
            decoder_atrous_rates = (12, 24, 36),
            in_channels = 3,
            classes = 20,
            upsampling = 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#lr = 0.01 # to apply later poly learning rate strategy with power 0.9
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
lr = 2e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

PATH = "./deeplab2s/checkpoints_3/DSR_state_0.pth"
model, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse = utils.load_checkpoint(model, optimizer, scheduler, PATH)

############################################################################################################################################
############### TRAINING LOOP ##############################################################################################################
############################################################################################################################################
def train_model(model, loss_func, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    if valid_m_iou:
        best_iou = valid_m_iou[-1]
    else:
        best_iou = 0.0
    print(f'Best valid_iou until now: {best_iou:.4f}')
    end_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, end_epoch):
        print()
        print(f'[Epoch {epoch+1}/{end_epoch}]')
        print('-' * 10)
        model.train() # set model to training mode

        train_loss_b = []
        train_w_iou_b = []
        train_m_iou_b = []
        train_rmse_b = []
        for batch_idx, batch in enumerate(train_loader, 0):     

            optimizer.zero_grad()

            img = batch["img"].to(device)
            gt_sisr = batch["gt_sisr"].to(device)
            gt_sssr = batch["gt_sssr"].to(device)

            sssr, sisr, sssr_FA = model(img)

            # Segmentation metrics
            #sssr_preds = torch.argmax(sssr, dim=1)
            w_iou, m_iou, _ = mIoU(sssr, gt_sssr)
            train_w_iou_b.append(w_iou.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
            train_m_iou_b.append(m_iou.item() * img.shape[0])

            # Super-resolution metrics
            root_mse = rmse(sisr, gt_sisr)
            train_rmse_b.append(root_mse.item() * img.shape[0])

            loss = loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=True)
            train_loss_b.append(loss.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                 print(f'Batch {batch_idx+1} / {num_train_batches}, Training mIoU = {iou.item():.4f}, Loss = {loss.item():.4f}, RMSE = {root_mse.item():.4f}')

        train_loss.append(np.sum(train_loss_b)/train_loader.dataset.num_examples)
        train_w_iou.append(np.sum(train_w_iou_b)/train_loader.dataset.num_examples)
        train_m_iou.append(np.sum(train_m_iou_b)/train_loader.dataset.num_examples)
        train_rmse.append(np.sum(train_rmse_b)/train_loader.dataset.num_examples)
        print(f'-END of Epoch- Training wIoU = {train_w_iou[-1]:.4f}, mIoU = {train_m_iou[-1]:.4f}, Loss = {train_loss[-1]:.4f}, RMSE = {train_rmse[-1]:.4f}')
        scheduler.step()

        model.eval() # set model to validation mode
        valid_loss_b = []
        valid_w_iou_b = []
        valid_m_iou_b = []
        valid_rmse_b = []
        for batch in val_loader:
            with torch.no_grad():    
                
                img = batch["img"].to(device)
                gt_sisr = batch["gt_sisr"].to(device)
                gt_sssr = batch["gt_sssr"].to(device)

                sssr, sisr, sssr_FA = model(img)

                # Segmentation metrics
                #sssr_preds = torch.argmax(sssr, dim=1)
                w_iou, m_iou, _ = mIoU(sssr, gt_sssr)
                valid_w_iou_b.append(w_iou.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
                valid_m_iou_b.append(m_iou.item() * img.shape[0])

                # Super-resolution metrics
                root_mse = rmse(sisr, gt_sisr)
                valid_rmse_b.append(root_mse.item() * img.shape[0])

                loss = loss_func(sssr, sisr, gt_sssr, gt_sisr, sssr_FA, train=False)
                valid_loss_b.append(loss.item() * img.shape[0]) # append batchwise IoU weighted by the bacth size
        print()
        valid_loss.append(np.sum(valid_loss_b)/val_loader.dataset.num_examples)
        valid_w_iou.append(np.sum(valid_w_iou_b)/val_loader.dataset.num_examples)
        valid_m_iou.append(np.sum(valid_m_iou_b)/val_loader.dataset.num_examples)
        valid_rmse.append(np.sum(valid_rmse_b)/val_loader.dataset.num_examples)
        print(f'Validation wIoU = {valid_w_iou[-1]:.4f}, mIoU = {valid_m_iou[-1]:.4f}, Loss = {valid_loss[-1]:.4f}. RMSE = {valid_rmse[-1]:.4f}')

        # deep copy the model 
        if valid_m_iou[-1] > best_iou:
            best_iou = valid_m_iou[-1] 
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':  train_loss,
                'train_w_iou': train_w_iou,
                'train_m_iou': train_m_iou,
                'train_rmse': train_rmse,
                'valid_loss': valid_loss,
                'valid_w_iou': valid_w_iou,
                'valid_m_iou': valid_m_iou,
                'valid_rmse': valid_rmse
            }
            torch.save(state, "./deeplab2s/checkpoints_3/DSR_state_0.pth")
        print() # end of epoch

    print('Finished Training!')
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val mIoU: {best_iou:.4f}')
    model.load_state_dict(best_model_wts)
    
    fig1, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax1.plot(range(0, end_epoch), train_loss, label="Training Loss")
    ax1.plot(range(0, end_epoch), valid_loss, label="Validation Loss")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    plt.close(fig1)
    fig1.savefig('./deeplab2s/Plots_3/loss_v_state_0.png')

    fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 2 axis
    ax2.plot(range(0, end_epoch), train_w_iou, label="Training wIoU")
    ax2.plot(range(0, end_epoch), valid_w_iou, label="Validation wIoU")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('wIoU')
    ax2.set_title('Training vs Validation wIoU')
    ax2.legend()
    plt.close(fig2)
    fig2.savefig('./deeplab2s/Plots_3/w_iou_v_state_0.png')

    fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 3 axis
    ax3.plot(range(0, end_epoch), train_m_iou, label="Training mIoU")
    ax3.plot(range(0, end_epoch), valid_m_iou, label="Validation mIoU")
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('mIoU')
    ax3.set_title('Training vs Validation mIoU')
    ax3.legend()
    plt.close(fig3)
    fig3.savefig('./deeplab2s/Plots_3/m_iou_v_state_0.png')

    fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 4 axis
    ax4.plot(range(0, end_epoch), train_rmse, label="Training RMSE")
    ax4.plot(range(0, end_epoch), valid_rmse, label="Validation RMSE")
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Training vs Validation RMSE')
    ax4.legend()
    plt.close(fig4)
    fig4.savefig('./deeplab2s/Plots_3/rmse_v_state_0.png')
    
    return model

model = train_model(model, loss_func, optimizer, scheduler, start_epoch, train_loss, valid_loss, train_w_iou, train_m_iou, valid_w_iou, valid_m_iou, train_rmse, valid_rmse, num_epochs=150)
torch.save(model.state_dict(), "./deeplab2s/trained_models_3/DSR_v_0.pth")