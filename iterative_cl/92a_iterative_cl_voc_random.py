import torch
import time
from pathlib import Path

from utils import (get_classes, get_log_csv_name, get_log_csv_train_order, get_log_image_output_folder)
from utils_model import compute_resnet_grad_no_update, train_resnet_iterative_cl
from utils_model_helper import get_data_transforms, ImageFolderWithPaths, get_random_indices, plot_class_histograms, append_zeros

exp_id = "92a"

#manual
# train_folder = Path("/home/brenta/scratch/jason/data/voc/voc_train_tiny")
train_folder = Path("/home/brenta/scratch/jason/data/voc/voc_trainval_full/train")
val_folder = Path("/home/brenta/scratch/jason/data/voc/voc_trainval_full/val")
checkpoints_folder = Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_" + exp_id)
log_folder = Path("/home/brenta/scratch/jason/logs/voc/vanilla/exp_" + exp_id)
learning_rate = 1e-3
lr_decay_per_epoch = 0.9

#self-generated
log_csv = get_log_csv_name(log_folder=log_folder)
log_folder.mkdir(parents=True, exist_ok=True)
train_order_csv = get_log_csv_train_order(log_folder=log_folder)
image_output_folder = get_log_image_output_folder(log_folder=log_folder)
image_output_folder.mkdir(parents=True, exist_ok=True)
classes = get_classes(train_folder)
num_classes = len(classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################################################
# Load the ImageDataset
imagedataset_start = time.time()
color_jitter_brightness = 0 
color_jitter_contrast = 0
color_jitter_hue = 0 
color_jitter_saturation = 0
path_mean = [0.40853017568588257, 0.4573926329612732, 0.48035722970962524] 
path_std = [0.28722450137138367, 0.27334490418434143, 0.2799932360649109] 

data_transforms = get_data_transforms(
    color_jitter_brightness=color_jitter_brightness, color_jitter_contrast=color_jitter_contrast,
    color_jitter_hue=color_jitter_hue, color_jitter_saturation=color_jitter_saturation,
    path_mean=path_mean, path_std=path_std)

print(  f"\nloading train: \t\t\t{train_folder}\n"
        f"loading val: \t\t\t{val_folder}\n")
image_datasets = {
    "train": ImageFolderWithPaths(root=str(train_folder), transform=data_transforms["train"]),
    "val": ImageFolderWithPaths(root=str(val_folder), transform=data_transforms["val"]), }
print(f"dataset loading time, {time.time() - imagedataset_start} seconds")




############################################################################
# MAIN
############################################################################

minibatch_counter = 0
resume_checkpoint_path = Path("/home/brenta/scratch/jason/checkpoints/voc/exp_53/resnet18_e10_mb400_va0.61554.pt")

for i in range(100):

    ############################################################################
    # Run data with mb size 1 and get (index, magnitudes) tuples 
    tup_list = compute_resnet_grad_no_update(   
                        image_datasets = image_datasets, 
                        train_order_csv = train_order_csv,
                        log_csv = log_csv, 
                        checkpoints_folder = checkpoints_folder,
                        num_classes = num_classes,
                        device = device, 
                        classes = classes, 
                        resume_checkpoint = True, 
                        resume_checkpoint_path = resume_checkpoint_path 
                        )          

    ############################################################################
    # Find best (index, magnitude) tuples
    best_combined_tup_list = get_random_indices(tup_list)
    best_indices = [x[0] for x in best_combined_tup_list]

    # Print predicted distributions
    tup_list_output_png_path = str(image_output_folder.joinpath("distributions_tup_list" + append_zeros(str(i)) + ".png"))
    plot_class_histograms(tup_list, tup_list_output_png_path)
    chosen_tup_list_output_png_path = str(image_output_folder.joinpath("distributions_chosen_tup_list" + append_zeros(str(i)) + ".png"))
    plot_class_histograms(best_combined_tup_list, chosen_tup_list_output_png_path)

    ############################################################################
    # Calculate and update with best (index, magnitude) tuples

    epoch_output_path, minibatch_counter = train_resnet_iterative_cl(   
                        image_datasets = image_datasets, 
                        minibatch_counter = minibatch_counter, 
                        best_indices = best_indices, 
                        train_order_csv = train_order_csv,
                        log_csv = log_csv, 
                        checkpoints_folder = checkpoints_folder,
                        num_classes = num_classes,
                        device = device, 
                        classes = classes, 
                        resume_checkpoint = True, 
                        resume_checkpoint_path = resume_checkpoint_path,
                        learning_rate=learning_rate
                        )   
    
    print(f"\n saved to {epoch_output_path}\n")
    resume_checkpoint_path = epoch_output_path
    learning_rate *= lr_decay_per_epoch