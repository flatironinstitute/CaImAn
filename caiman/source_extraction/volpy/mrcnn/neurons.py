#!/bin/python
"""
Mask R-CNN
Train on the segmentation of neurons.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Changjia Cai, and Manuel Paez 
"""

import os
import sys
import numpy as np
from skimage.color import gray2rgb
import skimage.draw
from skimage.draw import polygon2mask
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import tv_tensors
from torch.optim.lr_scheduler import CyclicLR
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T
from torchvision.ops.boxes import masks_to_boxes
from tqdm import tqdm
from typing import List, Dict, Any

from caiman.source_extraction.volpy.mrcnn.config import Config
from caiman.source_extraction.volpy.mrcnn.model import get_model_instance_segmentation, mrcnn_inference
from caiman.source_extraction.volpy.mrcnn.utils import ScaleImage, create_mask, collate_fn, data_transform, nf_match_neurons_in_binary_masks, normalize_image

# Dataset
class NeuronsDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading neuron images and their corresponding instance segmentation masks.

    This dataset is designed for object detection/instance segmentation tasks where each image
    has multiple object instances (neurons), each defined by a mask.
    """
    def __init__(self, root, transforms):
        """
        Args:
            root (str): The root directory of the dataset, which should contain
                        'images' and 'masks' subdirectories.
            transforms (Callable, optional): A function/transform that takes in an image
                                             and a target and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.image_filenames = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.mask_filenames = list(sorted(os.listdir(os.path.join(self.root, "masks"))))

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding target at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            A tuple containing:
            - image (tv_tensors.Image): The image tensor.
            - target (Dict[str, Any]): A dictionary containing the masks, bounding boxes,
                                       labels, and other metadata.
        """
        image_id = idx

        # Image: (C x H x W)
        image_path = os.path.join(self.root, "images", self.image_filenames[idx])
        image = np.load(image_path)['img'] # mean/mean/corr channels  (h w c)
        image = torch.from_numpy(image).permute(2,0,1) # convert to tensor and get into pytorch order C x H x W
        image = ScaleImage()(image)   # scale so it is in 0,1 range
        image = tv_tensors.Image(image)

        # Masks: N x H x W mask array (N masks)
        mask_path = os.path.join(self.root, "masks", self.mask_filenames[idx])
        masks_loaded = np.load(mask_path, allow_pickle=True)
        masks = masks_loaded['mask']
        # first create boolean mask stack
        all_masks = []
        for mask_ind, mask_dict in enumerate(masks): # [mask_ind]
            mask = create_mask(image[1].shape, mask_dict)
            all_masks.append(mask)
        all_masks = np.array(all_masks)
        # then convert to binary uint8 tensor stack
        all_masks = torch.from_numpy(all_masks.astype(np.uint8))

        boxes = masks_to_boxes(all_masks)
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # tensor of areas

        # there is only one class, so labels are all ones
        num_objs = len(masks)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # let's just say nstances are not crowd: all instances will be used for evaluation
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap up everything into a dictionary describing target
        target = {}
        target["image_id"] = image_id
        target["masks"] = tv_tensors.Mask(all_masks)
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        target["labels"] = labels
        target["area"] = box_areas
        target["iscrowd"] = iscrowd

        # run augmentation, if transforms exist
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target
    
    def __len__(self):
        return len(self.image_filenames)
        
    def print_image_filenames(self):
        for image_filename in self.image_filenames:
            print(image_filename)

    def print_mask_filenames(self):
        for mask_filename in self.mask_filenames:
            print(mask_filename)
    
def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader,
                    device: torch.device, epoch: int) -> float:
    """
    Trains the model for one epoch and returns the average training loss.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run training on.
        epoch (int): The current epoch number, used for display purposes.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    train_epoch_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1} [train]"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_epoch_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return train_epoch_loss / len(data_loader)

def validate(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
            device: torch.device, epoch: int):
    """
    Calculates the validation loss for one epoch.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.
        epoch (int): The current epoch number, used for display purposes.

    Returns:
        float: The average validation loss for the epoch.
    """
    model.train()
    val_epoch_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1} [val]"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_epoch_loss += losses.item()
    return val_epoch_loss / len(data_loader)

def perform_final_evaluation(model: nn.Module, config, device: torch.device, plot_results: bool = False):
    """
    Runs inference on the validation set, calculates F1 scores, and reports results.

    Args:
        model (nn.Module): The trained model to evaluate.
        config (MockConfig): A configuration object with necessary paths and parameters.
        device (torch.device): The device (CPU/GPU) to run evaluation on.
        plot_results (bool): If True, enables plotting within the matching function.
    """
    model.eval() # Set the model to evaluation mode

    # Validation Data
    val_indices_path = os.path.join(config.MODEL_SAVE_DIR, 'validation_indices.npy')
    if not os.path.exists(val_indices_path):
        print(f"Validation indices not found at {val_indices_path}.")
        return

    val_indices = np.load(val_indices_path)
    print(f"\nLoaded {len(val_indices)} validation indices for final evaluation.")

    full_dataset = NeuronsDataset(config.DATA_DIR, data_transform(train=False))
    dataset_val = torch.utils.data.Subset(full_dataset, val_indices)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                 num_workers=config.NUM_TORCH_WORKERS, collate_fn=collate_fn)

    # Initialize Score Tracking
    f1_scores_by_region = {region: [] for region in config.DATASET_REGION_MAP if region != 'Train'}
    all_f1_scores = []

    print("\nCalculating F1 scores...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader_val, desc="Evaluation")):
            vp_im = images[0]
            vp_target = targets[0]
            original_idx = val_indices[i]

            # Prepare image and ground truth masks
            summary_mn = gray2rgb(normalize_image(vp_im[1, :, :].cpu().numpy()))
            data_masks = vp_target['masks']

            # Run inference to get predicted masks
            _, _, binarized_masks = mrcnn_inference(model, img=vp_im.to(device), thresh=config.INFERENCE_THRESHOLD,
                                                    eval_transform=data_transform(train=False), device=device)

            # Compare GT and Predicted Masks
            try:
                _, _, _, _, performance = nf_match_neurons_in_binary_masks(
                    data_masks.cpu().numpy().astype(np.float64),
                    binarized_masks.astype(np.float64),
                    plot_results=plot_results,
                    Cn=summary_mn,
                    labels=['GT', 'VolPy'], colors=['red', 'yellow']
                )

                f1_score = performance['f1_score']
                print(f"F1 score for validation image index {original_idx}: {f1_score:.4f}")
                all_f1_scores.append(f1_score)

                # Find the region for the current index and append the score
                region_name = next((r for r, inds in config.DATASET_REGION_MAP.items() if original_idx in inds), None)
                if region_name and region_name in f1_scores_by_region:
                    f1_scores_by_region[region_name].append(f1_score)

            except Exception as e:
                print(f"Could not calculate F1 score for image index {original_idx}: {e}")
                all_f1_scores.append(0)

    # Final Reporting Loop
    print(f'\nOverall Average F1 score: {np.mean(all_f1_scores):.4f}\n')
    print("Average F1 scores by region:")
    for region, scores in f1_scores_by_region.items():
        if scores:
            avg_score = np.mean(scores)
            print(f'Average F1 score for {region}: {avg_score:.4f}')

def train_validate(config, plot_results=False):
    """ Main function to run the training and validation pipeline."""
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    dataset_train = NeuronsDataset(config.DATA_DIR, data_transform(train=True))
    dataset_val = NeuronsDataset(config.DATA_DIR, data_transform(train=False))

    if config.RANDOM_SPLIT:
        print("Using random split for train/validation sets.")
        indices = list(range(len(dataset_train_instance)))
        np.random.shuffle(indices)
        train_indices = indices[:-config.NUM_TEST_RANDOM]
        val_indices = indices[-config.NUM_TEST_RANDOM:]
    else:
        print("Using fixed split based on DATASET_REGION_MAP.")
        train_indices = config.DATASET_REGION_MAP['Train']
        val_indices = [idx for region, inds in config.DATASET_REGION_MAP.items() if region != 'Train' for idx in inds]
    
    val_indices_path = os.path.join(config.MODEL_SAVE_DIR, 'validation_indices.npy')
    np.save(val_indices_path, val_indices)
    print(f"Validation indices for this run have been saved to {val_indices_path}")

    dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

    data_loader_train = DataLoader(dataset_train, batch_size=config.BATCH_SIZE, shuffle=True,
                                   num_workers=config.NUM_TORCH_WORKERS, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                 num_workers=config.NUM_TORCH_WORKERS, collate_fn=collate_fn)

    model = get_model_instance_segmentation(config.NUM_CLASSES)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.MAX_LR, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = CyclicLR(optimizer, base_lr=config.BASE_LR, max_lr=config.MAX_LR,
                            step_size_up=config.STEP_SIZE_UP, step_size_down=config.STEP_SIZE_DOWN,
                            mode="triangular2")

    all_train_losses, all_val_losses, all_lrs = [], [], []
    print(f"**TRAIN {config.NUM_EPOCHS} epochs. PRINT every {config.PRINT_FREQ} epoch(s). "
          f"SAVE every {config.SAVE_FREQ} epoch(s).**")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        val_loss = validate(model, data_loader_val, device, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_lrs.append(current_lr)

        lr_scheduler.step()

        if (epoch + 1) % config.PRINT_FREQ == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        if (epoch + 1) % config.SAVE_FREQ == 0:
            model_path = os.path.join(config.MODEL_SAVE_DIR, f'mrcnn_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"\tModel saved to {model_path}")

    history = {'train_loss': all_train_losses, 'val_loss': all_val_losses, 'lr': all_lrs}
    torch.save(history, os.path.join(config.MODEL_SAVE_DIR, 'volpy_train_history.pt'))
    print("\nDONE!")

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def run_inference(config, plot_results=True):
    """Loads a trained model and runs inference on the validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model_instance_segmentation(config.NUM_CLASSES)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'mrcnn_epoch_{config.NUM_EPOCHS}.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
        
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    val_indices_path = os.path.join(config.MODEL_SAVE_DIR, 'validation_indices.npy')
    if not os.path.exists(val_indices_path):
        print(f"Validation indices not found at {val_indices_path}.")
        return 
        
    val_indices = np.load(val_indices_path)

    full_dataset = NeuronsDataset(config.DATA_DIR, data_transform(train=False))
    dataset_val = torch.utils.data.Subset(full_dataset, val_indices)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=config.NUM_TORCH_WORKERS, collate_fn=collate_fn)
    
    perform_final_evaluation(model, config, device, plot_results=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or run inference with Mask R-CNN.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help="Select mode: 'train' or 'infer'")
    parser.add_argument('--random_split', action='store_true', help="Use a random train/validation split. Overrides config.")
    parser.add_argument('--plot_results', action='store_true',
                        help="Display plots for each validation image during inference.")
    
    args = parser.parse_args()

    config = Config()
    if args.random_split:
        config.RANDOM_SPLIT = True

    if args.mode == 'train':
        # Call evaluation at the end of training, passing the plotting flag
        train_and_validate(config, plot_results=args.plot_results) 
    elif args.mode == 'infer':
        # Call inference, passing the plotting flag
        run_inference(config, plot_results=args.plot_results)
