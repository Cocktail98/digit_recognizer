from __future__ import print_function, division

import time
import torch
import matplotlib.pyplot as plt


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, save_model_path,
                num_epochs=25):
    start_time = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_list = {'train': [], 'val': []}
    acc_list = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if 'train' == phase:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for index, bath_data in enumerate(dataloaders[phase]):
                inputs = bath_data['image'].to(device)
                labels = bath_data['label'].to(device)

                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled('train' == phase):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if 'train' == phase:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} | Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.
                  format(phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)
            if 'val' == phase and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_model_path + 'best.pt')

        torch.save(model.state_dict(), save_model_path + 'last.pt')
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # draw image
    plt.figure(figsize=(12, 5))
    # loss
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Epoch Loss')
    l1 = plt.plot(loss_list['train'], 'r--', label='train_loss')
    l2 = plt.plot(loss_list['val'], 'g--', label='val_loss')
    ax.plot(loss_list['train'], 'ro-', loss_list['val'], 'g+-')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    # acc
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Accuracy')
    l1 = plt.plot(acc_list['train'], 'r--', label='train_loss')
    l2 = plt.plot(acc_list['val'], 'g--', label='val_loss')
    ax.plot(acc_list['train'], 'ro-', acc_list['val'], 'g+-')
    ax.set_xlabel('accuracy')
    ax.set_ylabel('loss')
    ax.legend()
    plt.savefig('./res/loss&acc.jpg')
    # plt.show()

    # load best model weights
    model.load_state_dict(torch.load(save_model_path + 'best.pt'))
    return model
