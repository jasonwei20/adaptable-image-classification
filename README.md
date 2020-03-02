# Readable and adaptable image classification with ResNet

This code optimizes readability and adaptability to new experiments. 

# Setup (to do)
- Installations (follow FAIR self-supervised benchmark)
- Conda environments

# Usage
```sh
CUDA_VISIBLE_DEVICES=0 python vanilla/53_vanilla_train_voc.py
```

## Usage tips 
1. You should use experiment ID to keep track of what configurations were run. For instance, for your third experiment, you can create a new file `3_train_imagenet.py`. You can name your [screen](https://www.tecmint.com/screen-command-examples-to-manage-linux-terminals/)s accordingly with the experiment ID as well. 
2. You should use aliases in your `~/.bashrc`.

Credits to Joseph DiPalma and Naofumi Tomita for major edits to the [Deepslide](https://github.com/BMIRDS/deepslide) repository that this code is based on.
