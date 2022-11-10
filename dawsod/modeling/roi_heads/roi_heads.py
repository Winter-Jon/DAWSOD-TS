import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou



from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers



from detectron2.modeling.roi_heads import StandardROIHeads,ROI_HEADS_REGISTRY


@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    """
    Convert instance-level annotations to image-level
    """
    if targets is None:
        return None, None, None
    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )
    return gt_classes_img, gt_classes_img_int, gt_classes_img_oh

    

@ROI_HEADS_REGISTRY.register()
class DAWSOD_ROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_weak(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        See :class:`ROIHeads.forward`.
        """
        assert targets
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        del targets

        sampled_proposals = []
        for proposals_per_image in proposals:
            sampled_idxs = torch.randperm(
                len(proposals_per_image), device=proposals_per_image.proposal_boxes.device
            )[:self.batch_size_per_image]
            sampled_proposals.append(proposals_per_image[sampled_idxs])

        losses = self._forward_box_weak(features, sampled_proposals)
        return losses

    def _forward_box_weak(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
    
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
    
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        box_features = self.box_head(box_features)
        cls_predictions, _ = self.box_predictor(box_features)
        del box_features

        num_proposal_per_image = [len(p) for p in proposals]
        # TODO: better implementation to calculate loss using rest inputs with non-valid proposals
        if 0 in num_proposal_per_image:
            return {"loss_mil": 0.0 * cls_predictions.sum()}
    
        objectness_scores = torch.unsqueeze(torch.cat([p.objectness_logits for p in proposals]), dim=1)
    
        cls_scores = F.softmax(cls_predictions[:, :-1], dim=1)
    
        max_cls_ids = torch.unsqueeze(torch.argmax(cls_predictions[:, :-1], dim=1), dim=1)
        objectness_scores = torch.zeros_like(cls_scores).scatter_(
            dim=1, index=max_cls_ids, src=objectness_scores
        )
    
        pred_img_cls_logits = torch.cat(
            [
                torch.sum(cls*F.softmax(obj, dim=0), dim=0, keepdim=True)
                for cls, obj in zip(
                cls_scores.split(num_proposal_per_image, dim=0),
                objectness_scores.split(num_proposal_per_image, dim=0))
            ],
            dim=0,
        )
    
        img_cls_losses = F.binary_cross_entropy(
            torch.clamp(pred_img_cls_logits, min=1e-6, max=1.0 - 1e-6),
            self.gt_classes_img_oh,
            reduction='mean'
        )
        return {"loss_mil": img_cls_losses}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances],protype: Optional[torch.Tensor]=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            # pred_features = predictions[0][:,:-1]
            # cls_scores = self.protype_classifier(pred_features,protype)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


    def forward_weak_pro(
        self,
        src: str,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        protype: List[torch.Tensor],
        targets: Optional[List[Instances]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        See :class:`ROIHeads.forward`.
        """
        assert targets
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        del targets

        sampled_proposals = []
        for proposals_per_image in proposals:
            sampled_idxs = torch.randperm(
                len(proposals_per_image), device=proposals_per_image.proposal_boxes.device
            )[:self.batch_size_per_image]
            sampled_proposals.append(proposals_per_image[sampled_idxs])

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features,True)
        # pred_features = self.box_predictor(box_features)
        # box_features = pred_features[0][:,:-1]
        # del pred_features
        
        num_proposal_per_image = [len(p) for p in proposals]
        objectness_scores = torch.unsqueeze(torch.cat([p.objectness_logits for p in proposals]), dim=1)
        
        cls_scores = self.protype_classifier(box_features,protype)

        # max_cls_ids = torch.unsqueeze(torch.argmax(cls_scores, dim=1), dim=1)
        # objectness_scores = torch.zeros_like(cls_scores).scatter_(
        #     dim=1, index=max_cls_ids, src=objectness_scores
        # )
        objectness_scores = objectness_scores.detach()


        pred_img_cls_logits = torch.cat(
            [
                torch.sum(cls*torch.sigmoid(obj), dim=0, keepdim=True)
                for cls, obj in zip(
                cls_scores.split(num_proposal_per_image, dim=0),
                objectness_scores.split(num_proposal_per_image, dim=0))
            ],
            dim=0,
        )

        # img_cls_losses = F.binary_cross_entropy(
        #     torch.clamp(torch.sigmoid(torch.log(pred_img_cls_logits)), min=1e-6, max=1.0 - 1e-6),
        #     self.gt_classes_img_oh,
        #     reduction='mean'
        # )

        mean = torch.mean(pred_img_cls_logits,dim=1).unsqueeze(1)
        std = torch.std(pred_img_cls_logits,dim=1).unsqueeze(1)
        normed_pred_img_cls_logits = (pred_img_cls_logits -mean) / std

        img_cls_losses = F.binary_cross_entropy(
            torch.clamp(torch.sigmoid(normed_pred_img_cls_logits*3), min=1e-6, max=1.0 - 1e-6),
            self.gt_classes_img_oh,
            reduction='mean'
        )

        # cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(
        #     torch.clamp(torch.sigmoid(pred_img_cls_logits), min=1e-6, max=1.0 - 1e-6),
        #     self.gt_classes_img_oh
        # )
        # img_cls_losses = -torch.log(torch.mean(cos_sim))

        # pred_img_cls_logits = torch.cat(
        #     [
        #         torch.sum(cls*F.softmax(obj, dim=0), dim=0, keepdim=True)
        #         for cls, obj in zip(
        #         cls_scores.split(num_proposal_per_image, dim=0),
        #         objectness_scores.split(num_proposal_per_image, dim=0))
        #     ],
        #     dim=0,
        # )
    
        # img_cls_losses = F.binary_cross_entropy(
        #     torch.clamp(pred_img_cls_logits, min=1e-6, max=1.0 - 1e-6),
        #     self.gt_classes_img_oh,
        #     reduction='mean'
        # )
        return {"loss_cls_w_"+src: img_cls_losses}
        
 
        # losses = self._forward_box_weak(features, sampled_proposals)

    
    def protype_classifier(self,box_feature,protype:torch.Tensor):
        box_feature = box_feature.unsqueeze(0)
        protype = protype.unsqueeze(0)
        protype = protype.clone()
        protype.requires_grad = False
        protype_mean = torch.mean(protype,dim=1,keepdim=True)
        
        feature_dist = torch.cdist(box_feature,protype,p=2)
        feature_dist_mean = torch.cdist(box_feature,protype_mean,p=2)

        feature_dist = feature_dist - feature_dist_mean

        cls_scores = F.softmax((-feature_dist),dim=2)
        cls_scores = cls_scores.squeeze()
        # cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        # feature_sim = cos(box_feature,protype)
        # cls_scores = F.softmax(feature_sim,dim=1)
        return cls_scores
    
    def _update_protype(self,features,targets,protype,ema_rate):
        # gt_classes_features = torch.zeros((self.num_classes,self.box_head.output_shape.channels))
        gt_classes_features = protype
        
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.gt_boxes for x in targets])
        box_features = self.box_head(box_features,True)
        # pred_features = self.box_predictor(box_features)
        # box_features = pred_features[0][:,:-1]
        # del pred_features
        
        gt_classes = [gt.gt_classes.split(1,dim=0) for gt in targets]
        
        gt_classes_tup = ()
        for i in gt_classes:
            gt_classes_tup += i
        feature_dict = {}
        for cls,features in zip(gt_classes_tup,box_features.split(1,dim=0)):
            cls = cls.item()
            if feature_dict.get(cls) == None:
                feature_dict[cls] = [features]
            else:
                feature_dict[cls].append(features)
        
        for cls,feature in feature_dict.items():
            feature = torch.vstack(tuple(feature))
            feature = torch.mean(feature,dim=0)
            feature = feature 
            gt_classes_features[cls] = (1 - ema_rate) * gt_classes_features[cls] + ema_rate * feature
            # gt_classes_features[cls] = feature
        
        return gt_classes_features