from __future__ import print_function, division

import torch


def pred_test(model, dataloader, device,res_path):
    with open(res_path, 'w') as f:
        f.write('ImageId,Label\n')
    model.eval()
    img_id = 1
    for index, bath_data in enumerate(dataloader):
        inputs = bath_data['image'].to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in preds:
            with open(res_path, 'a') as f:
                f.write('{},{}\n'.format(img_id, i))
            img_id += 1

    print('Predict is finished!')
