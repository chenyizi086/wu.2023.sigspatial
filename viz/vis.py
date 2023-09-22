import os
import numpy as np
import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
from einops import rearrange


def visualize(temporal_ID_imgs, anno_ID_img, out, attn_temp, attn_spatial, spatial_ID_imgs, sample_save_path, epoch, index):
    #############################################################################################################################################
    train_sum = np.concatenate([(temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                                (anno_ID_img[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                                (out[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)

    attn_t = nn.Upsample(size=out.shape[-2:], mode="bilinear", align_corners=False)(torch.max(attn_temp, axis = 0)[0])[0]
    attn_sum_t = (attn_t[0].detach().cpu().numpy()*255).astype(np.uint8)
    input_sum_t = (temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)

    for j in range(1, attn_t.shape[0]):
        attn_sum_t = np.concatenate([attn_sum_t, (attn_t[j].detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)
        input_sum_t = np.concatenate([input_sum_t, (temporal_ID_imgs[0][j].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis=1)

    attn_sum_t = np.repeat(np.expand_dims(attn_sum_t, -1), input_sum_t.shape[-1], axis=-1)
    attn_sum_t = cv2.applyColorMap(attn_sum_t, cv2.COLORMAP_JET)
    attn_sum_t = cv2.cvtColor(attn_sum_t, cv2.COLOR_BGR2RGB) # Remember to switch from BGR-RGB

    vis_temporal = 0.7 * input_sum_t + 0.3 * attn_sum_t
    vis_temporal = np.concatenate([vis_temporal, input_sum_t], axis = 0)
    vis_temporal = vis_temporal.astype(np.uint8)

    #############################################################################################################################################

    attn_s = nn.Upsample(size=out.shape[-2:], mode="bilinear", align_corners=False)(torch.max(attn_spatial, axis=0)[0])[0]
    attn_s = torch.stack([attn_s[0], attn_s[1], \
                          attn_s[2], attn_s[3], \
                          attn_s[4], attn_s[5], \
                          attn_s[6], attn_s[7], \
                          attn_s[8]], axis=-1)
    attn_s = rearrange(attn_s, 'h w (p1 p2)-> (p1 h) (p2 w)', p1=3, p2=3)
    attn_s = (attn_s.detach().cpu().numpy()*255).astype(np.uint8)
    attn_s = cv2.applyColorMap(attn_s, cv2.COLORMAP_JET)
    attn_s = cv2.cvtColor(attn_s, cv2.COLOR_BGR2RGB) # Remember to switch from BGR-RGB

    input_sum_s = spatial_ID_imgs[0].permute(1, 2, 3, 0)
    input_sum_s = rearrange(input_sum_s, 'c h w (p1 p2) -> (p1 h) (p2 w) c', p1=3, p2=3)
    input_sum_s = (input_sum_s.detach().cpu().numpy()*255).astype(np.uint8)

    # Average and add the grid to the image
    vis_spatial = 0.6 * input_sum_s + 0.4 * attn_s

    vis_spatial[::256, :] = (255, 255, 255)
    vis_spatial[:, ::256] = (255, 255, 255)

    vis_spatial = vis_spatial.astype(np.uint8)

    #############################################################################################################################################

    plt.imsave(os.path.join(sample_save_path, '{}_temporal_atten_{}.jpg'.format(epoch, index)), vis_temporal)
    plt.imsave(os.path.join(sample_save_path, '{}_train_{}.jpg'.format(epoch, index)), train_sum)
    plt.imsave(os.path.join(sample_save_path, '{}_spatial_atten_{}.jpg'.format(epoch, index)), vis_spatial)


def visualize_temp(temporal_ID_imgs, anno_ID_img, out, attn_temp, sample_save_path, epoch, index):
    train_sum = np.concatenate([(temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                                (anno_ID_img[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                                (out[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)

    attn_t = nn.Upsample(size=out.shape[-2:], mode="bilinear", align_corners=False)(torch.max(attn_temp, axis = 0)[0])[0]
    attn_sum_t = (attn_t[0].detach().cpu().numpy()*255).astype(np.uint8)
    input_sum_t = (temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)

    for j in range(1, attn_t.shape[0]):
        attn_sum_t = np.concatenate([attn_sum_t, (attn_t[j].detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)
        input_sum_t = np.concatenate([input_sum_t, (temporal_ID_imgs[0][j].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis=1)

    attn_sum_t = np.repeat(np.expand_dims(attn_sum_t, -1), input_sum_t.shape[-1], axis=-1)
    attn_sum_t = cv2.applyColorMap(attn_sum_t, cv2.COLORMAP_JET)

    vis_temporal = 0.7 * input_sum_t + 0.3 * attn_sum_t
    vis_temporal = np.concatenate([vis_temporal, input_sum_t], axis = 0)

    #############################################################################################################################################

    cv2.imwrite(os.path.join(sample_save_path, '{}_temporal_atten_{}.jpg'.format(epoch, index)), vis_temporal)
    cv2.imwrite(os.path.join(sample_save_path, '{}_train_{}.jpg'.format(epoch, index)), train_sum)
