import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import (Dict, List)


checkpoint_folder_dict = {  Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_92a"): "Random (Baseline)",
                            Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_92b"): "Highest 20 Percent by Gradient",
                            Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_92c"): "Highest 10 Percent + Random Sample 10 Percent",
                            Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_92d"): "Lowest 20 Percent by Graident", }

def get_image_names(folder: Path) -> List[Path]:
    """
    Find the names and paths of all of the images in a folder.
    Args:
        folder: Folder containing images (assume folder only contains images).
    Returns:
        A list of the names with paths of the images in a folder.
    """
    return sorted([
        Path(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)

def checkpoint_folder_to_val_accs(checkpoint_folder):

    tup_list = []
    checkpoint_names = get_image_names(checkpoint_folder)
    for checkpoint_name in checkpoint_names:
        checkpoint_str = str(checkpoint_name)[:-3]
        parts = checkpoint_str.split('_')
        epoch_num = int(parts[1][1:])
        mb_num = int(parts[2][2:])
        val_acc = float(parts[3][2:])
        tup = (mb_num, val_acc)
        tup_list.append(tup)
    tup_list = sorted(tup_list, key=lambda x:x[0])
    mb_num_list = [x[0] for x in tup_list]
    val_acc_list = [x[1] for x in tup_list]
    return mb_num_list, val_acc_list

def plot_val_accs(output_path, checkpoint_folder_dict):

    fig, ax = plt.subplots()
    plt.ylim([-0.02, 1.02])

    for checkpoint_folder in checkpoint_folder_dict:
        mb_num_list, val_acc_list = checkpoint_folder_to_val_accs(checkpoint_folder)
        plt.plot( mb_num_list, val_acc_list, label=checkpoint_folder_dict[checkpoint_folder] )
        print(mb_num_list)
        print(val_acc_list)

    plt.legend(loc="lower right")
    plt.title("CL performance on predicting image rotations (ResNet18, VOC RotNet)")
    plt.xlabel("Minibatch Updates")
    plt.ylabel("Validation Accuracy")
    plt.savefig(output_path, dpi=400)


if __name__ == "__main__":
    
    plot_val_accs("outputs/test_voc.png", checkpoint_folder_dict)