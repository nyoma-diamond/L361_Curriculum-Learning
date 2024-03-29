import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from PIL import Image
# from torchvision.utils import save_image

#TODO:
from cv2 import imwrite

def calculate_threshold(net, loss_func, trainloader, loss_threshold, DEVICE, loss_type, percentile_type):
    loss_indv = []

    if loss_type == 1:
        for images, labels in trainloader:

            images = images.to(DEVICE)      # @ N'yoma make sure to set images/labels to the device you're using
            labels = labels.to(DEVICE)

            losses = loss_func(net(images), labels).data.cpu().numpy()
            for loss in losses:
                loss_indv.append(loss) # get individual losses
        
        return np.percentile(np.array(loss_indv), loss_threshold, method = percentile_type)

    else:
        return loss_threshold



def curriculum_learning_loss(net, loss_func, images, labels, loss_threshold, DEVICE):
    """
    @ N'yoma i just realized I implemented this for batch per batch rather than epochs... damn

    Inputs: 
    - net: neural network
    - loss_func: torch.nn.CrossEntropyLoss(reduction='none')
        - Make sure to set reduction to none!
    - Images: batch of images
    - labels: batch of labels
    - loss_threshold: depending on what you enter as loss type, this can be actual loss value
        or the percentile value you want to test for your scenario
    - DEVICE: CPU/GPU
    - loss_type: This has the following options:
        - '0': use loss_threshold value as a singular number if a loss value is too high or low
        - '1': use loss_threshold value as a percentile in a normal distribution curve to see 
               if a loss value is too high or low 
    - percentile_type: check out this link for more info: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
        - "Linear": true percentile
        - "normal_unbiased": sampling from a normal distribution curve
    """


    images = images.to(DEVICE)      # @ N'yoma make sure to set images/labels to the device you're using
    labels = labels.to(DEVICE)


    loss_indv = loss_func(net(images), labels) # get individual losses

    # print(loss_indv)
    b = loss_indv >= loss_threshold # get indicies of which are larger than loss_threshold
    trash_indices = b.nonzero()
    d = loss_indv < loss_threshold # get indicies of which are larger than loss_threshold
    keep_indices = d.nonzero()

    return trash_indices, keep_indices, loss_threshold, loss_indv

#config['test_name']
def save_data(losses, images_failed, test_name, cid, DEVICE):
    folder_name = 'results/'+test_name+"/cid_"+str(cid)
    round = 0
    x = pd.DataFrame(losses, columns=["sample_loss",
                                      "loss_threshold_of_batch",
                                      "epoch",
                                      "batch_count"])

    if not os.path.exists(folder_name+"/round_0"):
        os.makedirs(folder_name+"/round_0")
        os.makedirs(folder_name+"/round_"+str(round)+"/imgs_failed")

    else:
        s = os.listdir(folder_name)
        s = [int(i.split("_")[-1]) for i in s]
        round = max(s) + 1
        print(round)
        if not os.path.exists(folder_name+"/round_"+str(round)):
            os.makedirs(folder_name+"/round_"+str(round))
            os.makedirs(folder_name+"/round_"+str(round)+"/imgs_failed")
    x.to_csv(folder_name+"/round_"+str(round)+"/losses.csv")

    # for [image_1,label,loss_ind, loss_threshold] in images_failed:
    #     file_name = folder_name+"/round_"+str(round)+"/imgs_failed/"+str(label.item())+"_"+str(loss_ind.item())+"_"+str(loss_threshold)+".jpg"   
    #     # take first image
    #     image = image_1[0].to(DEVICE)
    #     # Reshape the image
    #     image = image.reshape(3,32,32).to(DEVICE)
    #     # Transpose the image
    #     image = image.permute(1, 2, 0).to(DEVICE)
    #     # Display the image

    #     plt.imshow(image.cpu())
    #     plt.show()
        # imwrite(file_name, image.cpu())

    #     s=image_1.cpu()
    #     s = s.reshape(3,32,32)
    #     s = s.permute(1, 2, 0)
    #     s = s.numpy()
    #     print(s.shape)
    #     imwrite(file_name,s)
    #     # print(image_1)
    #     # tensor = image_1*255
    #     # tensor = np.array(tensor.cpu(), dtype=np.uint8)
    #     # if np.ndim(tensor)>3:
    #     #     assert tensor.shape[0] == 1
    #     #     tensor = tensor[0]
    #     # im = Image.fromarray(tensor)
    #     # im = im.convert('RGB')
    #     # im.save(file_name, "PNG")
    #     # img = Image.fromarray(image_1.cpu())
    #     # img.save(file_name)
    #     # batch_tensor=image_1.cpu().data.numpy()
    #     # batch_tensor=np.array(batch_tensor,dtype=np.uint8)
        
    #     # batch_tensor=np.transpose(batch_tensor,(0,2,3,1))
    #     # for index, image in enumerate(batch_tensor):
    #     #     ret_tensor=im.fromarray(image)
    #     #     ret_tensor.save(file_name)
    #     # print(file_name)
    #     # image = image_1.cpu().numpy()
    #     # imwrite(file_name, image)
    #     # take first image
    #     # image = image_1.to(DEVICE)
    #     # # Reshape the image
    #     # image = image.reshape(3,32,32).to(DEVICE)
    #     # # Transpose the image
    #     # image = image.permute(1, 2, 0).to(DEVICE)
    #     # image = image.numpy()
    #     # # Display the image
    #     # file_name = folder_name+"/round_"+str(round)+"/imgs_failed/"+str(label)+"_"+str(loss_ind)+"_"+str(loss_threshold)+'.jpg'    
    #     # plt.imshow(image.cpu())
    #     # plt.imsave(file_name, image.cpu())

    #     # count +=1 
    #     # plt.imshow(image.cpu())
    #     # plt.show()
    return


# TODO
def show_failed_imgs(failed_images, failed_labels, losses_failed, DEVICE, test_name, cid):
    folder_name = 'results/'+test_name+"/cid_"+str(cid)+"failed_img"
    count = 0
    s = os.listdir(folder_name)
    if not os.path.exists(folder_name+"/round_0"):
        os.makedirs(folder_name+"/round_0")
    # results folder to save all failed images in
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/'+str(batch_count)):
        os.makedirs('results/'+str(batch_count))

    count = 0
    # print(len(failed_images))
        
    for images in failed_images:
        # if count > 3:
        #     break

        # take first image
        image = images[0].to(DEVICE)
        # Reshape the image
        image = image.reshape(3,32,32).to(DEVICE)
        # Transpose the image
        image = image.permute(1, 2, 0).to(DEVICE)
        # Display the image
        imwrite('results/'+str(count)+'.jpg', image.cpu())

        count +=1 
        # plt.imshow(image.cpu())
        # plt.show()


"""
if not os.path.exists('out'):
    os.makedirs('out')
if not os.path.exists('results'):
    os.makedirs('results')
    
with open(f'out/{target_name}_{run_uuid}_out.fred', 'w') as file:
        file.write(filedata.format(**vars))
        

    imwrite('diff.jpg', diff_image)

    from cv2 import imread, resize, imwrite, erode
"""