import numpy as np
from PIL import Image

def trainModel(model, device, loader, loss_list, criterion, epoch, save = None):

    # Turn on training mode
    model.train(True)

    # loop through all batches
    for j, (img_batch, msk_batch, img_name) in enumerate(loader):
        # img needs shape [batch_size, channels, height, width]
        # mask needs [batch, H, W]

        img_batch, msk_batch = img_batch.to(device), msk_batch.to(device)

        # reset gradients
        optimizer.zero_grad()

        # process batch through network
        out = model(img_batch.float())

        # calculate loss
        loss_val = criterion(out.to(device), msk_batch.type(torch.long))  

        # Create an argmax map for a visualization of the results
        map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
        map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)

        #calculate the loss
        print("*" * 20)
        print('training loss', loss_val.tolist())
        print()
        
        # track batch loss
        loss_list.append(loss_val.item())
        # backpropagation
        loss_val.backward()
    
        # update the parameters
        optimizer.step()

        if save:
            map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
            # Create an argmax map
            #map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            #plt.imshow(map_out)
            #map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
            im = Image.fromarray(map_out)
            im.save(save)

    return loss_list

def validateModel(model, device, loader, loss_list, epoch, criterion, save = None):

    # Turn off training mode
    model.train(False)

    # loop through all batches
    with torch.no_grad():
        for img_batch, msk_batch, img_name in loader:
            # load image and mask
            img_batch, msk_batch = img_batch.to(device), msk_batch.to(device)

            # process batch through network
            out = model(img_batch.float())

            # calculate loss
            loss_val = criterion(out.to(device), msk_batch.type(torch.long))  

            #calculate the loss
            print("*" * 20)
            print('testing loss', loss_val.tolist())
            print()
            
            # track batch loss
            loss_list.append(loss_val.item())

            map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
            #print('mean backg', np.mean(map[0,:,:]))
            #print('mean hedges', np.mean(map[1,:,:]))
            #print('max map', np.max(map_out))
            #print()
            if save:
                # Create an argmax map
                #map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
                #plt.imshow(map_out)
                #map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
                im = Image.fromarray(map_out)
                im.save(save)

        return loss_list
