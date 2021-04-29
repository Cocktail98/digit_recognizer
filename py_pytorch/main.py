from __future__ import print_function, division

import argparse
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import *
from model import *
from train import *
from visualize import *
from pred_test import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter.')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--step_size', default=5, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--test_path', default='../data/test.csv', type=str)
    parser.add_argument('--train_path', default='../data/train.csv', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--save_model_path', default='./model_weight/', type=str)
    args = parser.parse_args()
    print(args)

    # divide to train_set and val_set
    df = pd.read_csv(args.train_path)
    df = df.sample(frac=1.0)  # upset all
    cut_idx = int(round(0.1 * df.shape[0]))
    df_val, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]

    image_datasets = {'train': DigitDataset(df_train), 'val': DigitDataset(df_val)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() and 'cuda' == args.device else "cpu")

    model = DigitModel().to(device)
    print(model)

    try:
        model.load_state_dict(torch.load(args.save_model_path + 'best.pt'))
    except BaseException:
        pass

    if 'train' == args.mode:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        train_model(model=model, dataloaders=dataloaders, dataset_sizes=dataset_sizes, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, device=device, save_model_path=args.save_model_path,
                    num_epochs=args.epochs)
        visualize_model(model=model, dataloaders=dataloaders, device=device)

    elif 'test' == args.mode:
        visualize_model(model=model, dataloaders=dataloaders, device=device)

        df_test = pd.read_csv(args.test_path)
        test_datasets = DigitTestDataset(df_test)
        test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size)
        test_dataset_sizes = len(test_datasets)

        pred_test(model=model, dataloader=test_dataloader, device=device, res_path='./res/submission.csv')

    else:
        print('wrong mode!')
