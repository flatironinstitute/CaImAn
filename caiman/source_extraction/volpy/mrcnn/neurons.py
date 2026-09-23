#!/bin/python
"""
Mask R-CNN
Train on the segmentation of neurons.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Changjia Cai, and Manuel Paez 
"""

import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
from skimage.color import gray2rgb
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import tv_tensors
from torch.optim.lr_scheduler import CyclicLR
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes
from tqdm import tqdm

from caiman.source_extraction.volpy.mrcnn.config import Config
from caiman.source_extraction.volpy.mrcnn.model import (
    get_model_instance_segmentation,
    load_mrcnn_weights,
    mrcnn_inference,
)
from caiman.source_extraction.volpy.mrcnn.utils import (
    collate_fn,
    create_mask,
    data_transform,
    nf_match_neurons_in_binary_masks,
    normalize_image,
    prepare_mrcnn_image,
)


def _atomic_torch_save(value, path):
    """Write a torch artifact atomically so interrupted writes do not replace a good file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(path)}.', suffix='.tmp', dir=directory
    )
    os.close(fd)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _check_output_directory(path, allow_overwrite=False, num_epochs=None, save_freq=None):
    """Create a writable output directory and protect existing training artifacts."""
    os.makedirs(path, exist_ok=True)
    artifact_names = {
        'mrcnn_latest.pt',
        'validation_indices.npy',
        'volpy_train_history.pt',
    }
    if num_epochs is not None:
        if save_freq is None or num_epochs < 1 or save_freq < 1:
            raise ValueError("num_epochs and save_freq must both be positive")
        artifact_names.update(
            f'mrcnn_epoch_{epoch}.pt'
            for epoch in range(save_freq, num_epochs + 1, save_freq)
        )
        artifact_names.add(f'mrcnn_epoch_{num_epochs}.pt')
        existing_artifacts = sorted(artifact_names.intersection(os.listdir(path)))
    else:
        existing_artifacts = sorted(
            name for name in os.listdir(path)
            if name in artifact_names or (name.startswith('mrcnn_epoch_') and name.endswith('.pt'))
        )
    if existing_artifacts and not allow_overwrite:
        raise FileExistsError(
            f"Training artifacts already exist in {os.path.abspath(path)}: "
            f"{existing_artifacts}. Choose a new MODEL_SAVE_DIR or set "
            "config.ALLOW_OVERWRITE = True explicitly."
        )
    fd, probe_path = tempfile.mkstemp(prefix='.volpy-write-test-', dir=path)
    os.close(fd)
    os.remove(probe_path)

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

        image_filenames = sorted(os.listdir(os.path.join(self.root, "images")))
        mask_filenames = sorted(os.listdir(os.path.join(self.root, "masks")))

        def sample_stem(filename):
            stem = os.path.splitext(filename)[0]
            return stem[:-5] if stem.endswith('_mask') else stem

        images_by_stem = {sample_stem(name): name for name in image_filenames}
        masks_by_stem = {sample_stem(name): name for name in mask_filenames}
        if images_by_stem.keys() != masks_by_stem.keys():
            missing_masks = sorted(images_by_stem.keys() - masks_by_stem.keys())
            missing_images = sorted(masks_by_stem.keys() - images_by_stem.keys())
            raise ValueError(
                "VolPy image/mask filenames do not match. "
                f"Missing masks for {missing_masks}; missing images for {missing_images}"
            )

        sample_stems = sorted(images_by_stem)
        self.image_filenames = [images_by_stem[stem] for stem in sample_stems]
        self.mask_filenames = [masks_by_stem[stem] for stem in sample_stems]

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding target at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            A tuple containing:
            - image (tv_tensors.Image): The image tensor.
            - target (dict): A dictionary containing the masks, bounding boxes,
                                       labels, and other metadata.
        """
        image_id = idx

        # Image: (C x H x W)
        image_path = os.path.join(self.root, "images", self.image_filenames[idx])
        image = np.load(image_path)['img'] # mean/mean/corr channels  (h w c)
        image = prepare_mrcnn_image(image)

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
    was_training = model.training
    model.train()
    # Torchvision detection models only return losses in training mode. Keep
    # BatchNorm frozen so validation data cannot update running statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    val_epoch_loss = 0
    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1} [val]"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_epoch_loss += losses.item()
    finally:
        model.train(was_training)
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
                                                    mask_threshold=config.MASK_THRESHOLD,
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
    if config.NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be at least 1")
    if config.SAVE_FREQ < 1:
        raise ValueError("SAVE_FREQ must be at least 1")
    _check_output_directory(
        config.MODEL_SAVE_DIR,
        getattr(config, 'ALLOW_OVERWRITE', False),
        num_epochs=config.NUM_EPOCHS,
        save_freq=config.SAVE_FREQ,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving training artifacts to: {os.path.abspath(config.MODEL_SAVE_DIR)}")

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    print("Loading datasets...")
    dataset_train = NeuronsDataset(config.DATA_DIR, data_transform(train=True))
    dataset_val = NeuronsDataset(config.DATA_DIR, data_transform(train=False))

    if config.RANDOM_SPLIT:
        print("Using random split for train/validation sets.")
        indices = np.random.default_rng(config.RANDOM_SEED).permutation(len(dataset_train)).tolist()
        train_indices = indices[:-config.NUM_TEST_RANDOM]
        val_indices = indices[-config.NUM_TEST_RANDOM:]
    else:
        print("Using fixed split based on DATASET_REGION_MAP.")
        train_indices = config.DATASET_REGION_MAP['Train']
        val_indices = [idx for region, inds in config.DATASET_REGION_MAP.items() if region != 'Train' for idx in inds]
    
    val_indices_path = os.path.join(config.MODEL_SAVE_DIR, 'validation_indices.npy')
    temporary_indices_path = f'{val_indices_path}.tmp.npy'
    np.save(temporary_indices_path, val_indices)
    os.replace(temporary_indices_path, val_indices_path)
    print(f"Validation indices for this run have been saved to {val_indices_path}")

    dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

    shuffle_generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=config.NUM_TORCH_WORKERS,
        collate_fn=collate_fn,
    )
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
    history_path = os.path.join(config.MODEL_SAVE_DIR, 'volpy_train_history.pt')
    latest_path = os.path.join(config.MODEL_SAVE_DIR, 'mrcnn_latest.pt')
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

        completed_epoch = epoch + 1
        history = {'train_loss': all_train_losses, 'val_loss': all_val_losses, 'lr': all_lrs}
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': completed_epoch,
            'config': config.to_dict(),
            'torch_version': str(torch.__version__),
            'torchvision_version': str(torchvision.__version__),
            'train_indices': train_indices,
            'validation_indices': val_indices,
            'history': history,
        }
        _atomic_torch_save(checkpoint, latest_path)
        _atomic_torch_save(history, history_path)
        print(f"\tLatest checkpoint saved to {latest_path}")

        if completed_epoch % config.SAVE_FREQ == 0 or completed_epoch == config.NUM_EPOCHS:
            model_path = os.path.join(config.MODEL_SAVE_DIR, f'mrcnn_epoch_{epoch+1}.pt')
            _atomic_torch_save(checkpoint, model_path)
            print(f"\tModel saved to {model_path}")

    history = {'train_loss': all_train_losses, 'val_loss': all_val_losses, 'lr': all_lrs}
    print(f"\nDONE! Final checkpoint: {model_path}")

    if plot_results:
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
    return model, history

def run_inference(config, plot_results=True):
    """Loads a trained model and runs inference on the validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model_instance_segmentation(config.NUM_CLASSES, pretrained=False)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'mrcnn_epoch_{config.NUM_EPOCHS}.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
        
    print(f"Loading model from {model_path}")
    load_mrcnn_weights(model, model_path, device)
    model.to(device)

    val_indices_path = os.path.join(config.MODEL_SAVE_DIR, 'validation_indices.npy')
    if not os.path.exists(val_indices_path):
        print(f"Validation indices not found at {val_indices_path}.")
        return 
        
    val_indices = np.load(val_indices_path)

    full_dataset = NeuronsDataset(config.DATA_DIR, data_transform(train=False))
    dataset_val = torch.utils.data.Subset(full_dataset, val_indices)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=config.NUM_TORCH_WORKERS, collate_fn=collate_fn)
    
    perform_final_evaluation(model, config, device, plot_results=plot_results)

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
        train_validate(config, plot_results=args.plot_results)
    elif args.mode == 'infer':
        # Call inference, passing the plotting flag
        run_inference(config, plot_results=args.plot_results)
