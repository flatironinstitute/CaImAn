#!/usr/bin/env python
"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Changjia Cai, Eric Thompson, Manuel Paez
"""

import os

from caiman.paths import caiman_datadir

class Config:
    # Paths
    DATA_DIR = os.path.expanduser(os.environ.get(
        "CAIMAN_VOLPY_TRAINING_DATA",
        os.path.join(caiman_datadir(), "volpy_training_data"),
    ))
    MODEL_SAVE_DIR = os.path.expanduser(os.environ.get(
        "CAIMAN_VOLPY_MODEL_DIR",
        os.path.join(caiman_datadir(), "model"),
    ))
    ALLOW_OVERWRITE = False

    # Model and Training Hyperparameters
    NUM_CLASSES = 1 + 1  # Background + Neuron
    MODEL_VARIANT = "maskrcnn_resnet50_fpn_v2"
    BATCH_SIZE = 2 
    NUM_EPOCHS = 100
    MAX_LR = 0.005
    BASE_LR = 0.000001
    STEP_SIZE_UP = 3
    STEP_SIZE_DOWN = 7

    # Data Loading, Splitting, and Inference
    RANDOM_SPLIT = False # True for random split, False for fixed split from map below.
    NUM_TEST_RANDOM = 8
    NUM_TORCH_WORKERS = 4
    RANDOM_SEED = 42
    DATASET_REGION_MAP = {
            'HPC': [0, 1, 2, 3],
            'L1': [12, 13, 14],
            'TEG': [21],
            'Train': [4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 22, 23]
        }

    # Inference settings used by the current FPN-v2 checkpoints.
    INFERENCE_THRESHOLD = 0.5
    MASK_THRESHOLD = 0.5
    BOX_NMS_THRESHOLD = 0.5
    MASK_EROSION_RADIUS = 1

    # Logging and Saving Frequency
    PRINT_FREQ = 1 
    SAVE_FREQ = 5

    def to_dict(self):
        """Return public configuration values."""
        return {
            name: getattr(self, name)
            for name in dir(self)
            if name.isupper() and not callable(getattr(self, name))
        }

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")
