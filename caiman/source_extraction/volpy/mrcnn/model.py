#!/usr/bin/env python
"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import numpy as np
import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from caiman.source_extraction.volpy.mrcnn.utils import ScaleImage, data_transform

# Model (Pre-trained on the COCO Dataset)
def get_model_instance_segmentation(num_classes):
    """
    Loads a pre-trained Mask R-CNN model and modifies its classification
    and mask prediction heads for a custom number of classes.

    Args:
        num_classes (int): The number of classes for the custom dataset,
                           including the background class.

    Returns:
        torch.nn.Module: The modified Mask R-CNN model ready for fine-tuning.
    """
    # load an instance segmentation model pre-trained on COCO, fpn_v2 provides better performance
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=3)

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

def mrcnn_inference(model, 
                    img, 
                    eval_transform, 
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                    thresh=0.5):
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
    
    predicted_masks, predicted_boxes = thresholded_predictions(pred, threshold=thresh) 
    binarized_masks = (0.5+predicted_masks).detach().cpu().numpy().astype(np.uint8) 
    return predicted_masks, predicted_boxes, binarized_masks

def thresholded_predictions(pred, threshold=0.7):
    """
    Get masks and boxes for those above threshold
    """
    numels = len(torch.where(pred['scores'] >= threshold)[0])
    masks = pred['masks'][:numels].squeeze()
    boxes = pred['boxes'][:numels]
    
    return masks, boxes 