import sys
sys.path.insert(1, '../')

import os
import numpy as np
import torch
import argparse
import datetime
from tqdm.auto import tqdm
from data import Data
from model.spa.UNETSpa import UNETSpa
from loss.dice_loss import DiceLoss
import matplotlib.pyplot as plt
import random

import log


def train(args):
    model = UNETSpa(n_channels=3, n_classes=4, n_head=args.n_head)

    model_name = args.model_type

    data_location_train = './dataset/sigspatial_npy_big/train'
    data_location_val = './dataset/sigspatial_npy_big/validation'

    train_img = Data(data_location_train)
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
    n_train = len(trainloader)

    val_img = Data(data_location_val)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    n_val = len(valloader)

    # Change it to adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

    current_time = str(datetime.datetime.now().strftime("%b-%d_%H_%M_%S"))

    if args.cuda:
        model.cuda()

    best_val = np.inf

    # Create res directory
    res_dir = os.path.join(args.res_dir, model_name, current_time + '_lr_' + str(args.base_lr))  + '_bs_'+ str(args.batch_size)
    print('Model save in {}'.format(res_dir))

    # Initialize the res directory
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    sample_save_path = os.path.join(res_dir, 'sample')
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    parm_save_path = os.path.join(res_dir, 'params')
    if not os.path.exists(parm_save_path):
        os.makedirs(parm_save_path)

    dice_loss = DiceLoss()
    logger = log.get_logger(os.path.join(res_dir, '{}.txt'.format(args.model_type)))
    epochs = args.epochs
    for epoch in range(0, epochs):
        model.train()
        mean_loss = []
        with tqdm(total=int(n_train*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (spatial_ID_imgs, temporal_ID_imgs, anno_ID_img) in enumerate(trainloader):
                # Set the gradient in the model into 0
                optimizer.zero_grad()

                # If batchsize not equal to batch index , calculate the current loss
                if args.cuda:
                    spatial_ID_imgs, temporal_ID_imgs, anno_ID_img = spatial_ID_imgs.cuda(), temporal_ID_imgs.cuda(), anno_ID_img.cuda()
                
                out  = model(temporal_ID_imgs, spatial_ID_imgs, return_att=False)
                c_loss = dice_loss(out, anno_ID_img)
                c_loss.backward()
                optimizer.step()

                mean_loss.append(c_loss.item())
                
                # Update the pbar
                pbar.update(anno_ID_img.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'total_loss': c_loss.item()})

                if i % 20 == 0:
                    train_sum = np.concatenate([(temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                        (anno_ID_img[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                        (out[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)
                    plt.imsave(os.path.join(sample_save_path, '{}_train_{}.jpg'.format(1, i)), train_sum)

            train_mean_loss = np.mean(mean_loss)

        model.eval()
        val_mean_loss = []
        with tqdm(total=int(n_val*args.batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (spatial_ID_imgs, temporal_ID_imgs, anno_ID_img) in enumerate(valloader):
                if args.cuda:
                    spatial_ID_imgs, temporal_ID_imgs, anno_ID_img = spatial_ID_imgs.cuda(), temporal_ID_imgs.cuda(), anno_ID_img.cuda()

                with torch.no_grad():
                    val_out = model(temporal_ID_imgs, spatial_ID_imgs)

                val_c_loss = dice_loss(val_out, anno_ID_img)

                val_mean_loss.append(val_c_loss.item())
                
                # Update the pbar
                pbar.update(val_out.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'val_total_loss': val_c_loss.item()})

                if i % 10 == 0:
                    val_sum = np.concatenate([(temporal_ID_imgs[0][0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), \
                                              (anno_ID_img[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),    \
                                              (val_out[0, 0:3].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)], axis = 1)
                    plt.imsave(os.path.join(sample_save_path, '{}_val_{}.jpg'.format(1, i)), val_sum)

            val_mean_loss = np.mean(val_mean_loss)

        logger.info('lr: %e, train_total_loss: %f, val_total_loss: %f' % 
                    (optimizer.param_groups[0]['lr'],
                        torch.from_numpy(np.array(train_mean_loss)).cuda(),
                        torch.from_numpy(np.array(val_mean_loss)).cuda(),
                        )
                    )

        if np.array(val_mean_loss) < best_val:
            best_val = np.array(val_mean_loss)
            torch.save(model.state_dict(), '{}/best_val.pth'.format(parm_save_path))  # Save best weight
            print("save best model at epochs: ", epoch)

        # Learning rate schedular to change learning
        scheduler.step(val_mean_loss)
        print('Current learning rate             {}'.format(optimizer.param_groups[0]['lr']))

    print('Best val loss: {}'.format(best_val))

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train map temporal.')

    parser.add_argument('-l', '--log', type=str, default='log.txt',
                        help='the file to store log, default is log.txt')
    parser.add_argument('--model_type', type=str, default='unet-spa',
                        help='The type of the model')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--param_dir', type=str, default='params',
                        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-4,
                        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch to train network, default is 100')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size of one iteration, default 2')

    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')
    parser.add_argument('--n_head', type=int, default=16,
                        help='Number of heads in the self-attention layers')

    parser.add_argument('--res_dir', type=str, default='./training_info/',
                        help='the dir to store result')

    return parser.parse_args()


if __name__ == '__main__':
    main()
