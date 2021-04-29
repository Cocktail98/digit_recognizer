from __future__ import print_function, division

import torch
import matplotlib.pyplot as plt


def visualize_model(model, dataloaders, device, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for index, bath_data in enumerate(dataloaders['val']):
            inputs = bath_data['image'].to(device)
            labels = bath_data['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('true:{} predict: {}'.format(labels.cpu().data[j], preds[j]))
                plt.imshow(inputs.cpu().data[j][0], cmap='gray')

                if images_so_far == num_images:
                    plt.savefig('./res/pre_examples.jpg')
                    # plt.show()
                    return
