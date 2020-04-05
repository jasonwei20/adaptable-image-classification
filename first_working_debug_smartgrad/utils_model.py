import gc
import operator
import random
import time
from pathlib import Path
from typing import (Dict, IO, List, Tuple)

import torchvision.models as models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import (datasets, transforms)

from utils_model_helper import (calculate_confusion_matrix, calculate_confusion_matrix, ImageFolderWithPaths, get_data_transforms, print_model_params, print_data_params, create_model)
from utils_model_helper import (SequentialClassSampler)

from utils_grad_mag import (model_to_grad_as_dict_and_flatten, get_idx_to_weight, check_model_weights, get_new_layer_grad)

def train_smartgrad_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, 
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, 
                 num_epochs: int,
                 log_writer: IO, 
                 train_order_writer: IO, 
                 device: torch.device, 
                 train_batch_size: int,
                 val_batch_size: int,
                 fake_minibatch_size: int, 
                 annealling_factor: float,
                 save_mb_interval: int, 
                 val_mb_interval: int,
                 checkpoints_folder: Path,
                 num_layers: int, 
                 classes: List[str],
                 num_classes: int) -> None:

    grad_layers = list(range(1, 21))

    since = time.time()
    global_minibatch_counter = 0
    # Initialize all the tensors to be used in training and validation.
    # Do this outside the loop since it will be written over entirely at each
    # epoch and doesn't need to be reallocated each time.
    train_all_labels = torch.empty(size=(dataset_sizes["train"], ),
                                   dtype=torch.long).cpu()
    train_all_predicts = torch.empty(size=(dataset_sizes["train"], ),
                                     dtype=torch.long).cpu()
    val_all_labels = torch.empty(size=(dataset_sizes["val"], ),
                                 dtype=torch.long).cpu()
    val_all_predicts = torch.empty(size=(dataset_sizes["val"], ),
                                   dtype=torch.long).cpu()

    for epoch in range(1, num_epochs+1):

        model.train(mode=False) # Training phase.
        train_running_loss, train_running_corrects, epoch_minibatch_counter = 0.0, 0, 0
        idx_to_gt = {}
        
        for idx, (inputs, labels, paths) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs, target=train_labels)
                train_loss.backward(retain_graph=True)

                gt_label = int(train_labels.detach().cpu().numpy()[0])
                idx_to_gt[idx] = gt_label

                ########################
                #### important code ####
                ########################

                #clear the memory
                fake_minibatch_idx = idx % fake_minibatch_size
                fake_minibatch_num = int(idx / fake_minibatch_size)
                if fake_minibatch_idx == 0:
                    minibatch_grad_dict = {}; gc.collect()
                
                #get the per-example gradient magnitude and add to minibatch_grad_dict
                grad_as_dict, grad_flattened = model_to_grad_as_dict_and_flatten(model, grad_layers)
                minibatch_grad_dict[idx] = (grad_as_dict, grad_flattened)

                #every batch, calculate the best ones
                if fake_minibatch_idx == fake_minibatch_size - 1:
                    idx_to_weight_batch = get_idx_to_weight(minibatch_grad_dict, annealling_factor, idx_to_gt)
                    print(idx_to_weight_batch)

                    ##########################
                    # print("\n...............................updating......................................" + str(idx))
                    for layer_num, param in enumerate(model.parameters()):
                        # if layer_num in [0]:#grad_layers:
                        new_grad = get_new_layer_grad(layer_num, idx_to_weight_batch, minibatch_grad_dict)
                        assert param.grad.detach().cpu().numpy().shape == new_grad.detach().cpu().numpy().shape
                        param.grad = new_grad
                            # check_model_weights(idx, model)
                    optimizer.step()
                    # check_model_weights(idx, model)
                    # print("................................done........................................." + str(idx) + '\n\n\n\n')
                    ##########################

            # Update training diagnostics.
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(train_preds == train_labels.data, dtype=torch.double)

            start = idx * train_batch_size
            end = start + train_batch_size
            train_all_labels[start:end] = train_labels.detach().cpu()
            train_all_predicts[start:end] = train_preds.detach().cpu()

            global_minibatch_counter += 1
            epoch_minibatch_counter += 1

            # Write the path of training order if it exists
            if train_order_writer:
                for path in paths: #write the order that the model was trained in
                    train_order_writer.write("/".join(path.split("/")[-2:]) + "\n")

            # Validate the model
            if global_minibatch_counter % val_mb_interval == 0 or global_minibatch_counter == 1:

                # Calculate training diagnostics
                calculate_confusion_matrix( all_labels=train_all_labels.numpy(), all_predicts=train_all_predicts.numpy(),
                                            classes=classes, num_classes=num_classes)
                train_loss = train_running_loss / (epoch_minibatch_counter * train_batch_size)
                train_acc = train_running_corrects / (epoch_minibatch_counter * train_batch_size)

                # Validation phase.
                model.train(mode=False)
                val_running_loss = 0.0
                val_running_corrects = 0

                # Feed forward over all the validation data.
                for idx, (val_inputs, val_labels, paths) in enumerate(dataloaders["val"]):
                    val_inputs = val_inputs.to(device=device)
                    val_labels = val_labels.to(device=device)

                    # Feed forward.
                    with torch.set_grad_enabled(mode=False):
                        val_outputs = model(val_inputs)
                        _, val_preds = torch.max(val_outputs, dim=1)
                        val_loss = criterion(input=val_outputs, target=val_labels)

                    # Update validation diagnostics.
                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    val_running_corrects += torch.sum(val_preds == val_labels.data,
                                                    dtype=torch.double)

                    start = idx * val_batch_size
                    end = start + val_batch_size
                    val_all_labels[start:end] = val_labels.detach().cpu()
                    val_all_predicts[start:end] = val_preds.detach().cpu()

                # Calculate validation diagnostics
                calculate_confusion_matrix( all_labels=val_all_labels.numpy(), all_predicts=val_all_predicts.numpy(),
                                            classes=classes, num_classes=num_classes)
                val_loss = val_running_loss / dataset_sizes["val"]
                val_acc = val_running_corrects / dataset_sizes["val"]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    

                # Remaining things related to training.
                if global_minibatch_counter % save_mb_interval == 0 or global_minibatch_counter == 1:

                    epoch_output_path = checkpoints_folder.joinpath(f"resnet{num_layers}_e{epoch}_mb{global_minibatch_counter}_va{val_acc:.5f}.pt")
                    epoch_output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the model as a state dictionary.
                    torch.save(obj={
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch + 1
                    }, f=str(epoch_output_path))

                log_writer.write(f"{epoch},{global_minibatch_counter},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

                current_lr = None
                for group in optimizer.param_groups:
                    current_lr = group["lr"]

                # Print the diagnostics for each epoch.
                print(f"Epoch {epoch} with "
                    f"mb {global_minibatch_counter} "
                    f"lr {current_lr:.15f}: "
                    f"t_loss: {train_loss:.4f} "
                    f"t_acc: {train_acc:.4f} "
                    f"v_loss: {val_loss:.4f} "
                    f"v_acc: {val_acc:.4f}\n")

        scheduler.step()

        current_lr = None
        for group in optimizer.param_groups:
            current_lr = group["lr"]


def train_smartgrad(   
                    fake_minibatch_size: int, #after how many examples you want to weight by
                    annealling_factor: float,
                    #needed
                    train_folder: Path, 
                    val_folder: Path, 
                    train_order_csv: Path,
                    log_csv: Path, 
                    checkpoints_folder: Path,
                    #auto-generate
                    num_classes: int,
                    device: torch.device, 
                    classes: List[str], 
                    #initialization
                    num_layers: int = 18, 
                    pretrain: bool = False, 
                    resume_checkpoint: bool = False, 
                    resume_checkpoint_path: Path = None, 
                    #learning
                    learning_rate: float = 1e-3,
                    learning_rate_decay: float = 0.85,
                    num_epochs: int = 200, 
                    train_batch_size: int = 1,
                    val_batch_size: int = 256, 
                    #logging
                    val_mb_interval: int = 1000,
                    save_mb_interval: int = 1000,
                    #data augmentation
                    color_jitter_brightness: float = 0, 
                    color_jitter_contrast: float = 0,
                    color_jitter_hue: float = 0, 
                    color_jitter_saturation: float = 0,
                    #probably don't change this
                    weight_decay: float = 1e-4, 
                    num_workers: int = 8,
                    path_mean: List[float] = [0.40853017568588257, 0.4573926329612732, 0.48035722970962524], 
                    path_std: List[float] = [0.28722450137138367, 0.27334490418434143, 0.2799932360649109], 
                    ) -> None:

    assert val_mb_interval % save_mb_interval == 0, "don't save more often than you validate"


    ############################################################################
    # Load the ImageDataset
    imagedataset_start = time.time()
    data_transforms = get_data_transforms(
        color_jitter_brightness=color_jitter_brightness, color_jitter_contrast=color_jitter_contrast,
        color_jitter_hue=color_jitter_hue, color_jitter_saturation=color_jitter_saturation,
        path_mean=path_mean, path_std=path_std)

    print(  f"\nloading train: \t\t\t{train_folder}\n"
            f"loading val: \t\t\t{val_folder}\n")
    image_datasets = {
        "train": ImageFolderWithPaths(root=str(train_folder), transform=data_transforms["train"]),
        "val": ImageFolderWithPaths(root=str(val_folder), transform=data_transforms["val"]), }
    print(f"dataset loading time: \t\t{time.time() - imagedataset_start:.3f} seconds")


    ############################################################################
    # Load the Dataloaders
    dataloaders_start = time.time()
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(dataset=image_datasets['train'],
                                       batch_size=train_batch_size,
                                       sampler=SequentialClassSampler(image_datasets['train'], num_classes),
                                       num_workers=num_workers)
    dataloaders['val'] = torch.utils.data.DataLoader(dataset=image_datasets['val'],
                                       batch_size=val_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)
    print(f"dataloaders loading time: \t{time.time() - dataloaders_start:.3f} seconds\n")

    dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}
    print_data_params(num_classes, classes, dataloaders, train_batch_size, val_batch_size, torch.cuda.is_available())


    ############################################################################
    # Load the Model
    model = create_model(num_classes=num_classes, num_layers=num_layers, pretrain=pretrain)
    model = model.to(device=device)
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)
    if resume_checkpoint:
        ckpt = torch.load(f=resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"model loaded from {resume_checkpoint_path}\n")
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        param_group['weight_decay'] = weight_decay
    print(optimizer)
    print()


    ############################################################################
    # Logging
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_writer = log_csv.open(mode="w")
    log_writer.write("epoch,minibatch,train_loss,train_acc,val_loss,val_acc\n")
    train_order_writer = train_order_csv.open(mode="w") if train_order_csv else None


    ############################################################################
    # Actual training
    train_smartgrad_helper(model=model,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                log_writer=log_writer,
                train_order_writer=train_order_writer,
                train_batch_size=train_batch_size, 
                val_batch_size=val_batch_size,
                fake_minibatch_size=fake_minibatch_size,
                annealling_factor=annealling_factor,
                checkpoints_folder=checkpoints_folder,
                device=device,
                num_layers=num_layers,
                val_mb_interval=val_mb_interval,
                save_mb_interval=save_mb_interval,
                num_epochs=num_epochs,
                classes=classes,
                num_classes=num_classes)