from pathlib import Path
import numpy as np
import torch
from dataprep import pick_from_sides
from matplotlib import pyplot as plt


def main(model, train_dataset, val_dataset, trainf, bad_val_inds, config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    # Create Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],  # The size of batches is defined here
        shuffle=True,
        # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=0,  # Data is usually loaded in parallel by num_workers
        drop_last = True,
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],  # The size of batches is defined here
        shuffle=False,  # During validation, shuffling is not necessary anymore
        num_workers=0,  # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'data/models').mkdir(exist_ok=True, parents=True)

    # Start training
    r, loss_curves = trainf(model, train_dataloader, val_dataloader, device, bad_val_inds, config)
    return r
