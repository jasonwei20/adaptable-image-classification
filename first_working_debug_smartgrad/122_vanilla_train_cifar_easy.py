import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
import numpy as np
np.random.seed(0)
from pathlib import Path

from utils import (get_classes, get_log_csv_name, get_log_csv_train_order)
from utils_model import train_smartgrad

exp_num = 122

# train_folder = Path("/home/brenta/scratch/jason/data/cifar/small_mammals_train_single_per_class/")
# train_folder = Path("/home/brenta/scratch/jason/data/cifar/small_mammals_train_two_per_class/")
# val_folder = Path("/home/brenta/scratch/jason/data/cifar/small_mammals/test/")
# resume_checkpoint_path = Path("/home/brenta/scratch/jason/checkpoints/cifar/vanilla/exp_117/resnet18_e7_mb50_va0.27750.pt")

train_folder = Path("/home/brenta/scratch/jason/data/cifar/super_easy_task/train/")
val_folder = Path("/home/brenta/scratch/jason/data/cifar/super_easy_task/test/")
resume_checkpoint_path = Path("/home/brenta/scratch/jason/checkpoints/cifar/vanilla/exp_122/resnet18_e1_mb1_va0.50000.pt")

checkpoints_folder = Path("/home/brenta/scratch/jason/checkpoints/cifar/vanilla/exp_" + str(exp_num))
log_folder = Path("/home/brenta/scratch/jason/logs/cifar/vanilla/exp_" + str(exp_num))
log_csv = get_log_csv_name(log_folder=log_folder)
train_order_csv = get_log_csv_train_order(log_folder=log_folder)
classes = get_classes(train_folder)
num_classes = len(classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_smartgrad(train_folder = train_folder, 
                val_folder = val_folder, 
                checkpoints_folder = checkpoints_folder,
                train_order_csv = None,
                log_csv = log_csv, 
                learning_rate = 0.001, 
                classes = classes,
                num_classes = num_classes,
                device = device, 
                train_batch_size = 1,
                val_batch_size = 256,
                save_mb_interval = 500, #asdf
                val_mb_interval = 500, #asdf
                num_epochs = 10, #asdf
                weight_decay = 0.0, #asdf
                fake_minibatch_size = 16, #asdf
                resume_checkpoint = True, #asdf
                resume_checkpoint_path = resume_checkpoint_path,
                annealling_factor = 0.0000001
                )