#!/usr/bin/env python

import numpy as np
import torch
import matplotlib.pyplot as plt

import caiman as cm
from caiman.source_extraction.volpy import utils as volpy_utils
from caiman.source_extraction.volpy.mrcnn.config import Config
from caiman.source_extraction.volpy.mrcnn.model import (
    mrcnn_inference,
    thresholded_predictions,
)
from caiman.source_extraction.volpy.mrcnn.neurons import (
    _atomic_torch_save,
    _check_output_directory,
    validate,
)
from caiman.source_extraction.volpy.mrcnn.utils import (
    nf_match_neurons_in_binary_masks,
    prepare_mrcnn_image,
)
from caiman.utils.utils import download_demo, download_model


def _prediction(count, height=8, width=9):
    scores = torch.tensor([0.9, 0.6, 0.1])[:count]
    return {
        'scores': scores,
        'masks': torch.ones((count, 1, height, width)),
        'boxes': torch.zeros((count, 4)),
    }


def test_thresholded_predictions_preserves_instance_dimension():
    for count, expected in ((0, 0), (1, 1), (3, 2)):
        masks, boxes, scores = thresholded_predictions(_prediction(count), threshold=0.5)
        assert masks.shape == (expected, 8, 9)
        assert boxes.shape == (expected, 4)
        assert scores.shape == (expected,)


def test_prepare_mrcnn_image_supports_2d_and_constant_images():
    image = prepare_mrcnn_image(np.full((8, 9), 3.0, dtype=np.float32))
    assert image.shape == (3, 8, 9)
    assert image.dtype == torch.float32
    assert torch.count_nonzero(image) == 0


def test_default_split_matches_reference_training_protocol():
    assert Config.DATASET_REGION_MAP == {
        'HPC': [0, 1, 2, 3],
        'L1': [12, 13, 14],
        'TEG': [21],
        'Train': [4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 22, 23],
    }


def test_binary_mask_evaluation_draws_visible_contours(monkeypatch):
    mask = np.zeros((1, 20, 20), dtype=np.uint8)
    mask[0, 5:10, 6:11] = 1
    contour_inputs = []
    original_contour = plt.contour

    def record_contour(values, *args, **kwargs):
        contour_inputs.append((np.asarray(values), kwargs.get('levels')))
        return original_contour(values, *args, **kwargs)

    monkeypatch.setattr(plt, 'contour', record_contour)
    monkeypatch.setattr(plt, 'show', lambda: None)
    nf_match_neurons_in_binary_masks(
        mask,
        mask,
        plot_results=True,
        Cn=np.zeros((20, 20)),
        labels=['GT', 'VolPy'],
        colors=['red', 'yellow'],
    )

    assert len(contour_inputs) == 2
    assert all(np.array_equal(values, mask[0]) for values, _ in contour_inputs)
    assert all(levels == [0.5] for _, levels in contour_inputs)
    plt.close('all')


def test_mrcnn_inference_keeps_legacy_three_value_contract():
    class FakeModel(torch.nn.Module):
        def forward(self, images):
            return [_prediction(1, images[0].shape[-2], images[0].shape[-1])]

    image = prepare_mrcnn_image(np.arange(72, dtype=np.float32).reshape(8, 9))
    masks, boxes, binary_masks = mrcnn_inference(
        FakeModel(),
        image,
        eval_transform=lambda value: value,
        device=torch.device('cpu'),
        thresh=0.5,
    )
    assert masks.shape == (1, 8, 9)
    assert boxes.shape == (1, 4)
    assert binary_masks.shape == (1, 8, 9)


def test_validation_does_not_update_batch_norm_statistics():
    class LossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = torch.nn.BatchNorm2d(1)

        def forward(self, images, targets):
            value = self.batch_norm(images[0].unsqueeze(0))
            return {'loss': value.sum()}

    model = LossModel()
    running_mean = model.batch_norm.running_mean.clone()
    data_loader = [([torch.ones((1, 2, 2))], [{}])]
    validate(model, data_loader, torch.device('cpu'), epoch=0)
    assert torch.equal(model.batch_norm.running_mean, running_mean)


def test_atomic_torch_save_replaces_artifact(tmp_path):
    artifact_path = tmp_path / 'checkpoint.pt'
    _atomic_torch_save({'epoch': 1}, artifact_path)
    _atomic_torch_save({'epoch': 2}, artifact_path)

    assert torch.load(artifact_path, weights_only=True) == {'epoch': 2}
    assert list(tmp_path.iterdir()) == [artifact_path]


def test_training_output_directory_protects_existing_artifacts(tmp_path):
    artifact_path = tmp_path / 'mrcnn_epoch_1.pt'
    artifact_path.touch()

    with np.testing.assert_raises(FileExistsError):
        _check_output_directory(tmp_path)

    _check_output_directory(tmp_path, allow_overwrite=True)


def test_training_output_directory_only_blocks_planned_epochs(tmp_path):
    (tmp_path / 'mrcnn_epoch_200.pt').touch()
    _check_output_directory(tmp_path, num_epochs=100, save_freq=5)

    (tmp_path / 'mrcnn_epoch_100.pt').touch()
    with np.testing.assert_raises(FileExistsError):
        _check_output_directory(tmp_path, num_epochs=100, save_freq=5)


def test_mrcnn_pytorch_demo_inference():
    weights_path = download_model('mask_rcnn')
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))

    rois = volpy_utils.mrcnn_inference_pytorch(
        img=summary_images.transpose([1, 2, 0]),
        size_range=[5, 22],
        weights_path=weights_path,
        display_result=False,
    )

    assert rois.dtype == bool
    assert rois.ndim == 3
    assert rois.shape[1:] == summary_images.shape[1:]
    assert len(rois) == 14
