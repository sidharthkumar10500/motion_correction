import numpy as np
import matplotlib.pyplot as plt
import torch, os
plt.rcParams.update({'font.size': 18})
plt.ioff(); plt.close('all')

"""
The objective of this script is to occasionally plot some of the training and val set images so that the user can see
if they are converging or not. 
"""
def plotter_GAN(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader):
    saved_results = torch.load(tosave_weights)
    G_loss_l1   =  saved_results['G_loss_l1']
    G_loss_adv  =  saved_results['G_loss_adv']
    D_loss_real =  saved_results['D_loss_real']
    D_loss_fake =  saved_results['D_loss_fake']
    G_loss      =  saved_results['G_loss_list']
    D_loss      = saved_results['D_loss_list']
    D_out_fake  = saved_results['D_out_fake']
    D_out_real  = saved_results['D_out_real']
    Lambda      = hparams.Lambda
    fig, ax1 = plt.subplots(figsize=(8,20), nrows=4, ncols=1)
    ax2 = ax1[0].twinx()
    ax1[0].plot(np.mean(G_loss_l1,axis=(1,2)), 'g-')
    ax2.plot(np.mean(G_loss_adv,axis=(1,2)), 'b-')
#     ax1.set_ylim([0, Lambda*.50])
    ax1[0].set_xlabel('Epoch index')
    ax1[0].set_ylabel('{} loss'.format(hparams.loss_type), color='g')
    ax1[0].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Adv loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Generator ({} and adv), $\lambda$ = {}'.format(hparams.loss_type ,Lambda))

    ax2 = ax1[1].twinx()
    ax1[1].plot(np.mean(D_loss_real,axis=(1,2)), 'g-')
    ax2.plot(np.mean(D_loss_fake,axis=(1,2)), 'b-')

    ax1[1].set_xlabel('Epoch index')
    ax1[1].set_ylabel('Disc real loss', color='g')
    ax1[1].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Disc fake loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Disc Loss (real and fake), $\lambda$ = {}'.format(Lambda))


    ax2 = ax1[2].twinx()
    ax1[2].plot(np.mean(D_loss,axis=(1,2)), 'g-')
    ax2.plot(np.mean(G_loss,axis=(1,2)), 'b-')
    ax1[2].set_xlabel('Epoch index')
    ax1[2].set_ylabel('Disc loss', color='g')
    ax1[2].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Generator loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('GAN Loss, $\lambda$ = {}'.format(Lambda))


    ax2 = ax1[3].twinx()
    ax1[3].plot(np.mean(D_out_real,axis=(1,2)), 'g-')
    ax2.plot(np.mean(D_out_fake,axis=(1,2)), 'b-')
    ax1[3].set_xlabel('Epoch index')
    ax1[3].set_ylabel('Disc out real', color='g')
    ax1[3].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Disc out fake', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Disc out, $\lambda$ = {}'.format(Lambda))

    # Save
    plt.tight_layout()
    plt.savefig(local_dir + '/GAN&DISC_loss_curves.png', dpi=100)
    plt.close()

    if not os.path.exists(local_dir + '/test_images'):
        os.makedirs(local_dir + '/test_images')
    if not os.path.exists(local_dir + '/train_images'):
        os.makedirs(local_dir + '/train_images')
    if not os.path.exists(local_dir + '/val_images'):
        os.makedirs(local_dir + '/val_images')
    img_plotter(hparams, UNet1,val_loader,train_loader,local_dir)


def plotter_UNET(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader):
    saved_results =  torch.load(tosave_weights)
    train_loss    =  saved_results['train_loss']
    val_loss      =  saved_results['val_loss']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(hparams.epochs),np.mean(train_loss,axis=1), 'g-')
    ax2.plot(np.arange(hparams.epochs),np.mean(val_loss,axis=1), 'b-')
#     ax1.set_ylim([0, Lambda*.50])
    ax1.set_xlabel('Epoch index')
    ax1.set_ylabel('Train loss', color='g')
    ax1.tick_params(axis='y', colors='g')
    ax2.set_ylabel('Val loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Train ({}) and val loss'.format(hparams.loss_type))
    # Save
    plt.tight_layout()
    plt.savefig(local_dir + '/UNET_loss_curves.png', dpi=100)
    plt.close()


    if not os.path.exists(local_dir + '/test_images'):
        os.makedirs(local_dir + '/test_images')
    if not os.path.exists(local_dir + '/train_images'):
        os.makedirs(local_dir + '/train_images')
    if not os.path.exists(local_dir + '/val_images'):
        os.makedirs(local_dir + '/val_images')

    img_plotter(hparams, UNet1,val_loader,train_loader,local_dir)



# function for plotting the train and validation images
def img_plotter(hparams, UNet1,val_loader,train_loader,local_dir):
    for index, sample in (enumerate(val_loader)):
        input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
        model_out            = UNet1(input_img.to(hparams.device)).permute(0,2,3,1)
        out                  = torch.view_as_complex(model_out.contiguous())
        generated_image      = torch.abs(out)
        target_img           = torch.abs(sample['img_gt'])

        #only plotting the first figure in the batch
        NN_output = generated_image[0,:,:].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,:,:].cpu().detach().numpy().squeeze()
        actual_in = torch.abs(sample['img_motion_corrupt'])[0,:,:].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        # plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(*params[0],params[1]), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Input')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir + '/val_images'+ '/val_image_index = {}.png'.format(index), dpi=100)
        plt.close()

    for index, sample in (enumerate(train_loader)):
        input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
        model_out            = UNet1(input_img.to(hparams.device)).permute(0,2,3,1)
        out                  = torch.view_as_complex(model_out.contiguous())
        generated_image      = torch.abs(out)
        target_img           = torch.abs(sample['img_gt'])

        NN_output = generated_image[0,:,:].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,:,:].cpu().detach().numpy().squeeze()
        actual_in = torch.abs(sample['img_motion_corrupt'])[0,:,:].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        # plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(*params[0],params[1]), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Input')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir +'/train_images'+ '/train_image_index = {}.png'.format(index), dpi=100)
        plt.close()

    for index, sample in (enumerate(hparams.test_loader)):
        input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
        model_out            = UNet1(input_img.to(hparams.device)).permute(0,2,3,1)
        out                  = torch.view_as_complex(model_out.contiguous())
        generated_image      = torch.abs(out)
        target_img           = torch.abs(sample['img_gt'])

        #only plotting the first figure in the batch
        NN_output = generated_image[0,:,:].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,:,:].cpu().detach().numpy().squeeze()
        actual_in = torch.abs(sample['img_motion_corrupt'])[0,:,:].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        # plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(*params[0],params[1]), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Input')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir + '/test_images'+ '/val_image_index = {}.png'.format(index), dpi=100)
        plt.close()