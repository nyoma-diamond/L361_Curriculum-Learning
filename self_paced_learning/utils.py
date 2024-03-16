import numpy as np
# import matplotlib.pyplot as plt
import os
import pandas as pd

#TODO:
# from cv2 import imwrite

def curriculum_learning_loss(net, loss_func, images, labels, loss_threshold, DEVICE, loss_type, percentile_type):
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

    if loss_type == 1:
        loss_threshold = np.percentile(loss_indv.data.cpu().numpy(), loss_threshold, method = percentile_type)

    # print(loss_indv)
    b = loss_indv >= loss_threshold # get indicies of which are larger than loss_threshold
    trash_indices = b.nonzero()
    d = loss_indv < loss_threshold # get indicies of which are larger than loss_threshold
    keep_indices = d.nonzero()

    return trash_indices, keep_indices, loss_threshold, loss_indv

#config['test_name']
def save_data(losses, test_name, cid):
    folder_name = 'results/'+test_name+"/cid_"+str(cid)
    round = 0
    x = pd.DataFrame(losses, columns=["sample_loss",
                                      "loss_threshold_of_batch",
                                      "epoch",
                                      "batch_count"])

    if not os.path.exists(folder_name+"/round_0"):
        os.makedirs(folder_name+"/round_0")
    else:
        s = os.listdir(folder_name)
        round = int(s[0][-1]) + 1
        if not os.path.exists(folder_name+"/round_"+str(round)):
            os.makedirs(folder_name+"/round_"+str(round))
    x.to_csv(folder_name+"/round_"+str(round)+"/losses.csv")
    return


# TODO
def show_failed_imgs(failed_images, failed_labels, losses_failed, DEVICE, batch_count):
    
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