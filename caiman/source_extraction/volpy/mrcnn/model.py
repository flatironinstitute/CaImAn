#!/usr/bin/env python
"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import pickle
import warnings

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

# Model (Pre-trained on the COCO Dataset)
def get_model_instance_segmentation(num_classes, pretrained=True):
    """
    Loads a pre-trained Mask R-CNN model and modifies its classification
    and mask prediction heads for a custom number of classes.

    Args:
        num_classes (int): The number of classes for the custom dataset,
                           including the background class.

    Returns:
        torch.nn.Module: The modified Mask R-CNN model ready for fine-tuning.
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=3 if pretrained else None,
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 
                                                        hidden_layer, 
                                                        num_classes)
    return model


def load_mrcnn_weights(model, weights_path, device):
    """Load legacy state dictionaries and metadata-bearing checkpoints."""
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        checkpoint = torch.load(weights_path, map_location=device)
    except pickle.UnpicklingError:
        warnings.warn(
            "This legacy checkpoint requires pickle loading. Only load model files "
            "from a trusted source.",
            RuntimeWarning,
            stacklevel=2,
        )
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        raise TypeError(f"Unsupported Mask R-CNN checkpoint type: {type(checkpoint).__name__}")
    model.load_state_dict(state_dict)
    return checkpoint

def mrcnn_inference(model, 
                    img, 
                    eval_transform, 
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                    thresh=0.5,
                    mask_threshold=0.5,
                    return_scores=False):
    """
    inference using Mask R-CNN network
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = eval_transform(img)
        x = x.to(device)
        predictions = model([x, ])
        pred = predictions[0]
    
    predicted_masks, predicted_boxes, predicted_scores = thresholded_predictions(
        pred, threshold=thresh
    )
    binarized_masks = (predicted_masks >= mask_threshold).to(torch.uint8).cpu().numpy()

    if return_scores:
        return predicted_masks, predicted_boxes, binarized_masks, predicted_scores
    return predicted_masks, predicted_boxes, binarized_masks

def thresholded_predictions(pred, threshold=0.7):
    """
    Get masks and boxes for those above threshold
    """
    keep = pred['scores'] >= threshold
    masks = pred['masks'][keep, 0]
    boxes = pred['boxes'][keep]
    scores = pred['scores'][keep]

    return masks, boxes, scores
