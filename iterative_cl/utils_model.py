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
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from utils_model_helper import (calculate_confusion_matrix, Random90Rotation, calculate_confusion_matrix, ImageFolderWithPaths, get_data_transforms, print_model_params, print_data_params, create_model)

def get_grad_magnitude(model, special_layer_nums = [0, 60, 1, 20, 40, 59]):
    params = list(model.parameters())
    layer_num_to_mag = {}
    total_mag = 0
    for layer_num, param in enumerate(params):
        layer_mag = np.sum(param.grad.detach().cpu().numpy()**2)

        if layer_num not in special_layer_nums:
            total_mag += layer_mag
        elif layer_num in special_layer_nums:
            layer_num_to_mag[layer_num] = layer_mag
    layer_num_to_mag[-1] = total_mag
    return layer_num_to_mag

def get_image_name(image_path):
    return '/'.join(image_path.split('/')[-2:])

def compute_resnet_grad_no_update_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, 
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, 
                 num_epochs: int,
                 log_writer: IO, 
                 train_order_writer: IO, 
                 device: torch.device, 
                 batch_size: int, 
                 checkpoints_folder: Path,
                 num_layers: int, 
                 classes: List[str],
                 num_classes: int):

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

    # grad_writer = grad_csv.open(mode="w")
    # grad_writer.write("image_name,train_loss,layers_-1,layer_0,layer_60,layer_1,layer_20,layer_40,layer_59,conf,correct\n")

    for epoch in range(1, num_epochs+1):

        model.train(mode=True) # Training phase.
        train_running_loss, train_running_corrects, epoch_minibatch_counter = 0.0, 0, 0

        tup_list = []

        for idx, (inputs, labels, paths) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                confs, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs, target=train_labels)
                train_loss.backward(retain_graph=True)
                # optimizer.step()

                train_loss_npy = float(train_loss.detach().cpu().numpy())
                layer_num_to_mag = get_grad_magnitude(model)
                image_name = get_image_name(paths[0])
                conf = float(confs.detach().cpu().numpy())
                train_pred = int(train_preds.detach().cpu().numpy()[0])
                gt_label = int(train_labels.detach().cpu().numpy()[0])
                correct = 0
                if train_pred == gt_label:
                    correct = 1

                output_line = f"{image_name},{train_loss_npy:.4f},{layer_num_to_mag[-1]:.4f},{layer_num_to_mag[0]:.4f},{layer_num_to_mag[60]:.4f},{layer_num_to_mag[1]:.4f},{layer_num_to_mag[20]:.4f},{layer_num_to_mag[40]:.4f},{layer_num_to_mag[59]:.4f},{conf:.4f},{correct}\n"
                # grad_writer.write(output_line)
                tup = (idx, image_name, layer_num_to_mag[-1])
                tup_list.append(tup)

                if idx % 1000 == 0:
                    print(tup)
            
    return tup_list

def compute_resnet_grad_no_update(   
                    #new for this method
                    image_datasets, 
                    #needed
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
                    batch_size: int = 1, 
                    learning_rate: float = 0,
                    learning_rate_decay: float = 0.85,
                    num_epochs: int = 1, 
                    #probably don't change this
                    weight_decay: float = 1e-4, 
                    num_workers: int = 8,
                    ):

    dataloaders_start = time.time()
    dataloaders = { x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                       batch_size=batch_size,
                                       sampler=SequentialSampler(image_datasets[x]),
                                       num_workers=num_workers)
        for x in ("train", "val") }
    print(f"dataloaders loading time, {time.time() - dataloaders_start} seconds")

    dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}
    print_data_params(num_classes, classes, dataloaders, batch_size, torch.cuda.is_available())

    model = create_model(num_classes=num_classes, num_layers=num_layers, pretrain=pretrain)
    model = model.to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    if resume_checkpoint:
        ckpt = torch.load(f=resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"model loaded from {resume_checkpoint_path}\n")

    log_csv.parent.mkdir(parents=True, exist_ok=True)

    log_writer = log_csv.open(mode="w")
    log_writer.write("epoch,minibatch,train_loss,train_acc,val_loss,val_acc\n")

    train_order_writer = train_order_csv.open(mode="w") if train_order_csv else None

    tup_list = compute_resnet_grad_no_update_helper(model=model,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                log_writer=log_writer,
                train_order_writer=train_order_writer,
                batch_size=batch_size,
                checkpoints_folder=checkpoints_folder,
                device=device,
                num_layers=num_layers,
                num_epochs=num_epochs,
                classes=classes,
                num_classes=num_classes)
    
    return tup_list







































def train_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, 
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, 
                 num_epochs: int,
                 log_writer: IO, 
                 train_order_writer: IO, 
                 device: torch.device, 
                 batch_size: int, 
                 checkpoints_folder: Path,
                 num_layers: int, 
                 classes: List[str],
                 minibatch_counter, 
                 num_classes: int) -> None:

    since = time.time()
    global_minibatch_counter = minibatch_counter
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

        model.train(mode=True) # Training phase.
        train_running_loss, train_running_corrects, epoch_minibatch_counter = 0.0, 0, 0

        for idx, (inputs, labels, paths) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs, target=train_labels)
                train_loss.backward()
                optimizer.step()

            # Update training diagnostics.
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(train_preds == train_labels.data, dtype=torch.double)

            this_batch_size = train_labels.detach().cpu().shape[0]
            start = idx * batch_size
            end = start + this_batch_size
            train_all_labels[start:end] = train_labels.detach().cpu()
            train_all_predicts[start:end] = train_preds.detach().cpu()

            global_minibatch_counter += 1
            epoch_minibatch_counter += 1

        # Calculate training diagnostics
        calculate_confusion_matrix( all_labels=train_all_labels.numpy(), all_predicts=train_all_predicts.numpy(),
                                    classes=classes, num_classes=num_classes)
        train_loss = train_running_loss / (epoch_minibatch_counter * batch_size)
        train_acc = train_running_corrects / (epoch_minibatch_counter * batch_size)

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

            this_batch_size = val_labels.detach().cpu().shape[0]
            start = idx * batch_size
            end = start + this_batch_size
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

    # Print training information at the end.
    print(f"\ntraining complete in "
          f"{(time.time() - since) // 60:.2f} minutes")
    
    return epoch_output_path, global_minibatch_counter


def train_resnet_iterative_cl(   
                    image_datasets,
                    best_indices,
                    minibatch_counter,
                    #needed
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
                    batch_size: int = 128, 
                    learning_rate: float = 1e-3,
                    learning_rate_decay: float = 0.85,
                    num_epochs: int = 1, 
                    #probably don't change this
                    weight_decay: float = 1e-4, 
                    num_workers: int = 8,
                    ) :

    dataloaders_start = time.time()
    dataloaders = { "train": torch.utils.data.DataLoader(dataset=image_datasets["train"],
                                       batch_size=batch_size,
                                       sampler=SubsetRandomSampler(best_indices),
                                       num_workers=num_workers),
                    "val": torch.utils.data.DataLoader(dataset=image_datasets["val"],
                                       batch_size=batch_size,
                                       num_workers=num_workers),
                                       }
    print(f"dataloaders loading time, {time.time() - dataloaders_start} seconds")

    dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}
    print_data_params(num_classes, classes, dataloaders, batch_size, torch.cuda.is_available())

    model = create_model(num_classes=num_classes, num_layers=num_layers, pretrain=pretrain)
    model = model.to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    if resume_checkpoint:
        ckpt = torch.load(f=resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"model loaded from {resume_checkpoint_path}\n")

    print_model_params( batch_size=batch_size,
                        checkpoints_folder=checkpoints_folder,
                        learning_rate=learning_rate,
                        learning_rate_decay=learning_rate_decay,
                        log_csv=log_csv,
                        train_order_csv=train_order_csv,
                        num_epochs=num_epochs,
                        num_layers=num_layers,
                        pretrain=pretrain,
                        resume_checkpoint=resume_checkpoint,
                        resume_checkpoint_path=resume_checkpoint_path,
                        save_mb_interval=-1,
                        val_mb_interval=-1,
                        train_folder=Path("placeholder"),
                        weight_decay=weight_decay)

    log_csv.parent.mkdir(parents=True, exist_ok=True)

    log_writer = log_csv.open(mode="w")
    log_writer.write("epoch,minibatch,train_loss,train_acc,val_loss,val_acc\n")

    train_order_writer = train_order_csv.open(mode="w") if train_order_csv else None

    return train_helper(model=model,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                log_writer=log_writer,
                train_order_writer=train_order_writer,
                batch_size=batch_size,
                checkpoints_folder=checkpoints_folder,
                device=device,
                num_layers=num_layers,
                num_epochs=num_epochs,
                classes=classes,
                minibatch_counter=minibatch_counter, 
                num_classes=num_classes)