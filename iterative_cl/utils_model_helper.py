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
import matplotlib.pyplot as plt


from utils import (get_image_paths, get_subfolder_paths)

def append_zeros(s, length=5):
    while len(s) < length:
        s = '0' + s
    return s

def plot_class_histograms(tup_list, output_png_path, _classes=['0', '90', '180', '270'], num_bins=100, heuristic_idx=2):

    fig, axs = plt.subplots(len(_classes) + 1)
    fig.tight_layout(pad=2)
    fig.set_figheight(10)

    _class_to_tup_list = {k: [] for k in _classes}
    for tup in tup_list:
        image_name = tup[1]
        image_class = image_name.split('/')[0]
        _class_to_tup_list[image_class].append(tup)

    for i, _class in enumerate(list(sorted(_class_to_tup_list.keys()))):
        _class_tup_list = _class_to_tup_list[_class]
        x = [tup[heuristic_idx] for tup in _class_tup_list]
        n, bins, patches = axs[i].hist(x, num_bins)
        axs[i].title.set_text("Class: " + _class)
        axs[i].set_xlim([0, 200])

    #plot the combined
    x = [tup[heuristic_idx] for tup in tup_list]
    n, bins, patches = axs[len(_classes)].hist(x, num_bins)
    axs[len(_classes)].title.set_text("Class: combined")
    axs[len(_classes)].set_xlim([0, 200])

    fig.savefig(output_png_path, dpi=300)


###########################################
#             MISC FUNCTIONS              #
###########################################

def get_highest_mag_indices(tup_list, frac=0.2, heuristic_idx=2, _classes=['0', '90', '180', '270']):

    _class_to_tup_list = {k: [] for k in _classes}
    for tup in tup_list:
        image_name = tup[1]
        image_class = image_name.split('/')[0]
        _class_to_tup_list[image_class].append(tup)
    
    best_combined_tup_list = []
    num_best_per_class = int(frac*len(tup_list)/len(_classes))
    for _class_tup_list in _class_to_tup_list.values():
        ordered_class_tup_list = sorted(_class_tup_list, key=lambda x: x[heuristic_idx])
        best_class_tup_list = ordered_class_tup_list[-num_best_per_class:]
        best_combined_tup_list += best_class_tup_list
    
    return best_combined_tup_list

def get_lowest_mag_indices(tup_list, frac=0.2, heuristic_idx=2, _classes=['0', '90', '180', '270']):

    _class_to_tup_list = {k: [] for k in _classes}
    for tup in tup_list:
        image_name = tup[1]
        image_class = image_name.split('/')[0]
        _class_to_tup_list[image_class].append(tup)
    
    worst_combined_tup_list = []
    num_worst_per_class = int(frac*len(tup_list)/len(_classes))
    for _class_tup_list in _class_to_tup_list.values():
        ordered_class_tup_list = sorted(_class_tup_list, key=lambda x: x[heuristic_idx])
        worst_class_tup_list = ordered_class_tup_list[:num_worst_per_class]
        worst_combined_tup_list += worst_class_tup_list
    
    return worst_combined_tup_list

def get_random_indices(tup_list, frac=0.2, heuristic_idx=2):

    random.shuffle(tup_list)
    num_worst = int(frac*len(tup_list))
    random_indices = tup_list[:num_worst]
    return random_indices

def create_model(num_layers: int, num_classes: int,
                 pretrain: bool) -> torchvision.models.resnet.ResNet:

    assert num_layers in (
        18, 34, 50, 101, 152
    ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
    model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
    model = model_constructor(num_classes=num_classes)

    if pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)
    return model

class Random90Rotation:
    def __init__(self, degrees: Tuple[int] = None) -> None:
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_data_transforms(color_jitter_brightness: float,
                        color_jitter_contrast: float,
                        color_jitter_saturation: float,
                        color_jitter_hue: float, path_mean: List[float],
                        path_std: List[float]
                        ) -> Dict[str, torchvision.transforms.Compose]:
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.Resize((224, 224)),
            # transforms.ColorJitter(brightness=color_jitter_brightness,
            #                        contrast=color_jitter_contrast,
            #                        saturation=color_jitter_saturation,
            #                        hue=color_jitter_hue),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ])
    }

def print_data_params( num_classes, classes, dataloaders, batch_size, is_available):

    print(f"{num_classes} classes: \t\t\t{classes}\n"
          f"num train images: \t\t{len(dataloaders['train']) * batch_size}\n"
          f"num val images: \t\t{len(dataloaders['val']) * batch_size}\n"
          f"CUDA is_available: \t\t{torch.cuda.is_available()}\n")

def print_model_params( train_folder: Path, num_epochs: int, num_layers: int,
                        learning_rate: float, batch_size: int, weight_decay: float,
                        learning_rate_decay: float, resume_checkpoint: bool,
                        resume_checkpoint_path: Path, save_mb_interval: int, val_mb_interval: int,
                        checkpoints_folder: Path, pretrain: bool,
                        log_csv: Path, train_order_csv: Path) -> None:

    print(f"train_folder: \t\t\t{train_folder}\n"
          f"output checkpoints_folder: \t{checkpoints_folder}\n"
          f"log_csv: \t\t\t{log_csv}\n"
          f"train_order_csv: \t\t{train_order_csv}\n"
          f"resume_checkpoint: \t\t{resume_checkpoint}\n"
          f"resume_checkpoint_path: \t{resume_checkpoint_path}\n"
          f"num_epochs: \t\t\t{num_epochs}\n"
          f"num_layers: \t\t\t{num_layers}\n"
          f"learning_rate: \t\t\t{learning_rate}\n"
          f"batch_size: \t\t\t{batch_size}\n"
          f"weight_decay: \t\t\t{weight_decay}\n"
          f"learning_rate_decay: \t\t{learning_rate_decay}\n"
          f"save_mb_interval: \t\t{save_mb_interval}\n"
          f"val_mb_interval: \t\t{val_mb_interval}\n"
          f"pretrain: \t\t\t{pretrain}\n\n")

def calculate_confusion_matrix(all_labels: np.ndarray,
                               all_predicts: np.ndarray, classes: List[str],
                               num_classes: int) -> None:
    """
    Prints the confusion matrix from the given data.
    Args:
        all_labels: The ground truth labels.
        all_predicts: The predicted labels.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    remap_classes = {x: classes[x] for x in range(num_classes)}

    # Set print options.
    # Sources:
    #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
    #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    actual = pd.Series(pd.Categorical(
        pd.Series(all_labels).replace(remap_classes), categories=classes),
                       name="Actual")

    predicted = pd.Series(pd.Categorical(
        pd.Series(all_predicts).replace(remap_classes), categories=classes),
                          name="Predicted")

    cm = pd.crosstab(index=actual, columns=predicted, normalize="index")

    cm.style.hide_index()
    print(cm)