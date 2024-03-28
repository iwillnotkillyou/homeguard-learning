from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from imports import *


def train(model, trainloader, valloader, device, bad_val_inds, config):
    # declare loss and move to specified device
    teacher_momentum = 0.75
    optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"], eps=config["eps"])

    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["learning_rate_decay"])
    # set model to train, important if your network has e.g. dropout or batchnorm layers

    for x in model.teacher.parameters():
        x.requires_grad = False
    model.train()
    criterion = Dino_loss(config["embedding_dim"], 1 / config["sharpness"], 1,
                          center_momentum=config["center_momentum"])
    criterion.train()
    criterion.to(device)

    # keep track of best validation accuracy achieved so that we can save the weights
    best_loss_val = np.inf
    loss_curves = []
    # keep track of running average of train loss for printing
    train_loss = []
    train_loss_sep = []
    train_separate_losses_mean = 0
    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            optimizer.zero_grad()
            loss_total, losses = backward_loss_multitask_with_dino(batch, device, model, config, criterion, epoch, False)
            optimizer.step()

            # loss logging
            train_loss.append(loss_total)
            train_loss_sep.append(losses)
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == 0 and iteration > 0:
                losses_mean, train_separate_losses_mean, most_separate_losses_mean = get_seperate_losses_means(
                    train_loss_sep)
                print(
                    f'[{epoch:03d}/{i:05d}] train_loss: {np.mean(train_loss)}, {train_separate_losses_mean}, {most_separate_losses_mean}')
                train_loss = []
                train_loss_sep = []

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == 0 and iteration > 0:

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                criterion.eval()

                loss_val = []
                loss_val_sep = []
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    with torch.no_grad():
                        loss_total, losses = backward_loss_multitask_with_dino(batch, device, model, config,
                                                                               criterion, epoch, True)
                    loss_val_sep.append(losses)
                    loss_val.append(loss_total.item())

                mlv = np.mean(loss_val)

                losses_mean, separate_losses_mean, most_separate_losses_mean = get_seperate_losses_means(loss_val_sep)
                loss_curves.append((train_separate_losses_mean, separate_losses_mean))
                print(loss_curves)
                for i in range(len(loss_curves[0][0])):
                    plt.plot([x[0][i] for x in loss_curves], label=f"train {i}")
                    plt.plot([x[1][i] for x in loss_curves], label=f"val {i}")
                plt.legend()
                plt.show()
                plt.clf()
                if mlv < best_loss_val and False:
                    torch.save(model.state_dict(), f'data/models/model.ckpt')
                    best_loss_val = mlv
                avg_auc = None
                if iteration % (config['validate_every_n'] * 3) == 0 and iteration > 0:
                    embeds = []
                    for batch_val in valloader:
                        data = batch_val[0].to(config["device"])
                        with torch.no_grad():
                            embeds.append(model.embedding(data))
                    embeds = torch.concatenate(embeds, 0)
                    avg_auc = averageauc(embeds, bad_val_inds)
                # set model back to train
                print(
                    f'[{epoch:03d}/{i:05d}] auc: {avg_auc} val_loss: {mlv}, {separate_losses_mean}, {most_separate_losses_mean}')
                model.train()
                criterion.train()
            if iteration % config["delayed_every_n"] == 0:
                sched.step()
                momentum = teacher_momentum if iteration > 0 else 0
                for mp, tp in zip(model.dino.parameters(), model.teacher.parameters()):
                    tp.data.mul(momentum)
                    tp.data.add((1 - momentum) * mp)
    return best_loss_val, loss_curves
