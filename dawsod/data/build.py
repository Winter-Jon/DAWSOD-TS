# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.data import get_detection_dataset_dicts,build_detection_train_loader


def build_detection_train_source_loader(cfg, mapper=None, *, dataset=None, sampler=None):
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN_SOURCE,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    return build_detection_train_loader(cfg,dataset=dataset)


