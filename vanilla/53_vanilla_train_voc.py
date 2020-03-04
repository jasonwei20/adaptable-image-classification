import torch
from pathlib import Path

from utils import (get_classes, get_log_csv_name, get_log_csv_train_order)
from utils_model import train_resnet

exp_num = 53

#manual
train_folder = Path("/home/brenta/scratch/jason/data/voc/voc_trainval_full/train")
val_folder = Path("/home/brenta/scratch/jason/data/voc/voc_trainval_full/val")
checkpoints_folder = Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_" + str(exp_num))
log_folder = Path("/home/brenta/scratch/jason/logs/voc/vanilla/exp_" + str(exp_num))

#self-generated
log_csv = get_log_csv_name(log_folder=log_folder)
train_order_csv = get_log_csv_train_order(log_folder=log_folder)
classes = get_classes(train_folder)
num_classes = len(classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_resnet(   train_folder = train_folder, 
                val_folder = val_folder, 
                checkpoints_folder = checkpoints_folder,
                train_order_csv = None,
                log_csv = log_csv, 
                classes = classes,
                num_classes = num_classes,
                device = device, 
                save_mb_interval = 50,
                val_mb_interval = 50,
                )