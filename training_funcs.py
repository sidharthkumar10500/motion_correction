import torch, os
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss, VGGPerceptualLoss
from plotter import plotter_GAN, plotter_UNET


def binary_acc(disc_out, actual_out):#function for calculating accuracy of discriminator
    m = nn.Sigmoid()#sigmoid is removed from the discriminator def to automatically handle the edge cases
    output = m(disc_out)
    disc_prediction = output>0.5
    actual_out = actual_out*torch.ones(disc_prediction.shape)
    compare = actual_out == disc_prediction
    out = torch.sum(compare)/torch.prod(torch.tensor(list(actual_out.size())))
    return out


def GAN_training(hparams):#separate function for doing generative training
    #load the parameters of interest
    device = hparams.device  
    epochs = hparams.epochs
    lr           = hparams.learn_rate
    Lambda = hparams.Lambda
    UNet1 = hparams.generator
    Discriminator1 = hparams.discriminator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   

    # choosing betas after talking with Ali, this are required for the case of GANs
    G_optimizer = optim.Adam(UNet1.parameters(), lr=lr, betas=(0.5, 0.999))
    G_scheduler = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    D_optimizer = optim.Adam(Discriminator1.parameters(), lr=0.00001, betas=(0.5, 0.999))
    D_scheduler = StepLR(D_optimizer, 5, 0.5)
    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss       = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss
    elif (hparams.loss_type=='Perc_L'):#perceptual loss based on vgg
        main_loss  = nn.L1Loss() #I will add the VGG loss later during the loss calculation time
    VGG_loss  = VGGPerceptualLoss().to(device)
    # figuring out the issue with weak discriminator in training GAN

    disc_epoch = hparams.disc_epoch #discriminator will be trained 10 times as much as generator and it will be trained first
    gen_epoch  = hparams.gen_epoch #generator will be trained for these many iterations 

    #lists to store the losses of discriminator and generator
    G_loss_l1, G_loss_adv    = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,gen_epoch,train_data_len)) 
    D_loss_real, D_loss_fake = np.zeros((epochs,disc_epoch,train_data_len)), np.zeros((epochs,disc_epoch,train_data_len))
    D_out_real, D_out_fake   = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,gen_epoch,train_data_len))
    G_loss_list, D_loss_list = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,disc_epoch,train_data_len))
    D_out_acc                = np.zeros((epochs,disc_epoch,train_data_len))
    accuracy_results         = np.zeros((epochs,disc_epoch))
    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        # at each epoch I re-initiate the discriminator optimizer
        for disc_epoch_idx in range(disc_epoch):
            for index, sample in (enumerate(train_loader)):
                # Move to CUDA
                for key in sample.keys():
                    try:
                        sample[key] = sample[key].to(device)
                    except:
                        pass

                input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
                model_out            = UNet1(input_img).permute(0,2,3,1)
                out                  = torch.view_as_complex(model_out.contiguous())
                generated_image      = torch.abs(out)
                target_img           = torch.abs(sample['img_gt'])

                G = Discriminator1(generated_image[:,None,:,:])

                # ground truth labels real and fake
                #using soft targets
                real_target = torch.ones(list(G.size())).to(device)
                fake_target = torch.zeros(list(G.size())).to(device)

                disc_inp_fake = generated_image.detach()
                D_fake = Discriminator1(disc_inp_fake[:,None,:,:])
                D_fake_loss = discriminator_loss(D_fake, fake_target)
                #Disc real loss
                disc_inp_real = target_img                
                D_real = Discriminator1(disc_inp_real[:,None,:,:])
                D_real_loss = discriminator_loss(D_real,  real_target)

                # average discriminator loss
                D_total_loss = (D_real_loss + D_fake_loss) / 2
                # compute gradients and run optimizer step
                D_total_loss.backward()
                D_optimizer.step()
                D_out_acc[epoch,disc_epoch_idx,index] = (binary_acc(D_real.cpu(), True) + binary_acc(D_fake.cpu(), False))
                D_loss_list[epoch,disc_epoch_idx,index] =  D_total_loss.cpu().detach().numpy()
                D_loss_real[epoch,disc_epoch_idx,index] =  D_real_loss.cpu().detach().numpy()
                D_loss_fake[epoch,disc_epoch_idx,index] =  D_fake_loss.cpu().detach().numpy()
            accuracy_results[epoch,disc_epoch_idx] = np.sum(D_out_acc[epoch,disc_epoch_idx,:])/(2*train_data_len)
            D_scheduler.step()
        # D_loss_list[epoch,:] =  D_loss_list[epoch,:]/disc_epoch #avg loss over disc_epoch training of discriminator
        # D_loss_real[epoch,:] =  D_loss_real[epoch,:]/disc_epoch
        # D_loss_fake[epoch,:] =  D_loss_fake[epoch,:]/disc_epoch


        for gen_epoch_idx in range(gen_epoch):
            for index, sample in (enumerate(train_loader)):
                # Move to CUDA
                for key in sample.keys():
                    try:
                        sample[key] = sample[key].to(device)
                    except:
                        pass

                input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
                model_out            = UNet1(input_img).permute(0,2,3,1)
                out                  = torch.view_as_complex(model_out.contiguous())
                generated_image      = torch.abs(out)
                target_img           = torch.abs(sample['img_gt'])

                G = Discriminator1(generated_image[:,None,:,:])
                # ground truth labels real and fake
                real_target = (torch.ones(list(G.size())).to(device))
                fake_target = torch.zeros(list(G.size())).to(device)
        
                gen_loss = adversarial_loss(G, real_target)
                #the 1 tensor need to be changed based on the max value in the input images
                if (hparams.loss_type=='SSIM'):
                    loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:], torch.tensor([1]).to(device)) + VGG_loss(generated_image[:,None,:,:], target_img[:,None,:,:])
                else:
                    loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:]) + VGG_loss(generated_image[:,None,:,:], target_img[:,None,:,:])

                G_loss = gen_loss + (Lambda* loss_val)  
                # compute gradients and run optimizer step
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                # store loss values
                G_loss_list[epoch,gen_epoch_idx,index] = G_loss.cpu().detach().numpy()
                G_loss_l1[epoch,gen_epoch_idx,index], G_loss_adv[epoch,gen_epoch_idx,index] = loss_val.cpu().detach().numpy(), gen_loss.cpu().detach().numpy()   
                #storing discriminator outputs 
                D_out_fake[epoch,gen_epoch_idx,index] = np.mean(G.cpu().detach().numpy())             
                G_real = Discriminator1(target_img[:,None,:,:])
                D_out_real[epoch,gen_epoch_idx,index] = np.mean(G_real.cpu().detach().numpy())

            # G_loss_list[epoch,:] = G_loss_list[epoch,:]/gen_epoch 
            # G_loss_l1[epoch,:], G_loss_adv[epoch,:] = G_loss_l1[epoch,:]/gen_epoch, G_loss_adv[epoch,:]/gen_epoch
            # #storing discriminator outputs 
            # D_out_fake[epoch,:] = D_out_fake[epoch,:]/gen_epoch          
            # D_out_real[epoch,:] = D_out_fake[epoch,:]/gen_epoch 
            # Generator training ends
        # Scheduler
        G_scheduler.step()
    # Save models
    local_dir = hparams.local_dir + '/learning_rate_{:.4f}_epochs_{}_lambda_{}_gen_epoch_{}_disc_epoch_{}'.format(hparams.lr,hparams.epochs,hparams.Lambda,hparams.gen_epoch,hparams.disc_epoch)  
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    tosave_weights = local_dir +'/saved_weights.pt' 
    torch.save({
        'epoch': epoch,
        'model_state_dict': UNet1.state_dict(),
        'optimizer_state_dict': G_optimizer.state_dict(),
        'Discriminator_state_dict':Discriminator1.state_dict(),
        'G_loss_list': G_loss_list,
        'G_loss_l1': G_loss_l1,
        'G_loss_adv': G_loss_adv,
        'D_loss_list': D_loss_list,
        'D_loss_real': D_loss_real,
        'D_loss_fake': D_loss_fake,
        'D_out_real':D_out_real,
        'D_out_fake':D_out_fake,
        'D_out_acc':D_out_acc,
        'hparams': hparams}, tosave_weights)
    plotter_GAN(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)


def UNET_training(hparams):
    device       = hparams.device  
    epochs       = hparams.epochs
    lr           = hparams.learn_rate
    UNet1        = hparams.generator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   
    G_optimizer = optim.Adam(UNet1.parameters(), lr=lr)#right now choosing Adam, other option is SGD
    scheduler = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    val_data_len = val_loader.__len__()
    # Criterions
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss  = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss
    elif (hparams.loss_type=='Perc_L'):#perceptual loss based on vgg
        main_loss  = VGGPerceptualLoss().to(device)
    VGG_loss  = VGGPerceptualLoss().to(device)
    train_loss = np.zeros((epochs,train_data_len)) #lists to store the losses of discriminator and generator
    val_loss = np.zeros((epochs,val_data_len)) #lists to store the losses of discriminator and generator

    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        for sample_idx, sample in (enumerate(train_loader)):
            # Move to CUDA
            for key in sample.keys():
                try:
                    sample[key] = sample[key].to(device)
                except:
                    pass

            input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
            model_out            = UNet1(input_img).permute(0,2,3,1)
            out                  = torch.view_as_complex(model_out.contiguous())
            generated_image      = torch.abs(out)
            target_img           = torch.abs(sample['img_gt'])

            #the 1 tensor need to be changed based on the max value in the input images
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:], torch.tensor([1]).to(device)) + VGG_loss(generated_image[:,None,:,:], target_img[:,None,:,:])
            else:
                loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:]) + VGG_loss(generated_image[:,None,:,:], target_img[:,None,:,:])

            # compute gradients and run optimizer step
            G_optimizer.zero_grad()
            loss_val.backward()
            G_optimizer.step()
            train_loss[epoch,sample_idx] = loss_val.cpu().detach().numpy()
        # Scheduler
        scheduler.step()
        for sample_idx, sample in (enumerate(val_loader)):
            # Move to CUDA
            for key in sample.keys():
                try:
                    sample[key] = sample[key].to(device)
                except:
                    pass     
            input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
            model_out            = UNet1(input_img).permute(0,2,3,1)
            out                  = torch.view_as_complex(model_out.contiguous())
            generated_image      = torch.abs(out)
            target_img           = torch.abs(sample['img_gt'])
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:], torch.tensor([1]).to(device))
            else:
                loss_val = main_loss(generated_image[:,None,:,:], target_img[:,None,:,:])

            val_loss[epoch,sample_idx] = loss_val.cpu().detach().numpy()
    # Save models
    local_dir = hparams.local_dir + '/learning_rate_{:.4f}_epochs_{}_lambda_{}_loss_type'.format(hparams.lr,hparams.epochs,hparams.Lambda,hparams.loss_type) 
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    tosave_weights = local_dir +'/saved_weights.pt' 
    torch.save({
        'epoch': epoch,
        'model_state_dict': UNet1.state_dict(),
        'optimizer_state_dict': G_optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hparams': hparams}, tosave_weights)
    plotter_UNET(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)