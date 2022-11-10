import copy
import os
from typing import OrderedDict
import torch
import time
import contextlib
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)

# from detectron2.modeling import GeneralizedRCNNWithTTA

from dawsod.data.build import build_detection_train_source_loader
from dawsod.evaluation.dawsod_evaluation import DomainAdaptiveDetectionEvaluator

try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:
    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


class DAWSOD_Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.source_data_loader = self.build_train_source_loader(cfg)
        self._source_data_loader_iter = iter(self.source_data_loader)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._trainer._data_loader_iter)

            if all([len(x["instances"]) > 0 for x in data]):
                break

        source_data = next(self._source_data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data+source_data,self.iter)
        losses = sum(loss_dict.values())
    
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            # metrics_dict["data_time"] = data_time
            self._trainer._write_metrics(metrics_dict,data_time)
            # self._trainer._detect_anomaly(losses, loss_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


    @classmethod
    def build_train_source_loader(cls, cfg):
        return build_detection_train_source_loader(cfg)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "cross_domain":
            return DomainAdaptiveDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
        
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list       


    # def build_hooks(self):
    #     cfg = self.cfg.clone()
    #     cfg.defrost()
    #     cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

    #     ret = [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(self.optimizer, self.scheduler),
    #         hooks.PreciseBN(
    #             # Run at the same freq as (but before) evaluation.
    #             cfg.TEST.EVAL_PERIOD,
    #             self.model,
    #             # Build a new data loader to not affect training
    #             self.build_train_loader(cfg),
    #             cfg.TEST.PRECISE_BN.NUM_ITER,
    #         )
    #         if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
    #         else None,
    #     ]

    #     # Do PreciseBN before checkpointer, because it updates the model and need to
    #     # be saved by checkpointer.
    #     # This is not always the best: if checkpointing has a different frequency,
    #     # some checkpoints may have more precise statistics than others.
    #     if comm.is_main_process():
    #         ret.append(
    #             hooks.PeriodicCheckpointer(
    #                 self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
    #             )
    #         )

    #     def test_and_save_results_student():
    #         self._last_eval_results_student = self.test(self.cfg, self.model)
    #         _last_eval_results_student = {
    #             k + "_student": self._last_eval_results_student[k]
    #             for k in self._last_eval_results_student.keys()
    #         }
    #         return _last_eval_results_student

    #     def test_and_save_results_teacher():
    #         self._last_eval_results_teacher = self.test(
    #             self.cfg, self.model_teacher)
    #         _last_eval_results_teacher = {
    #             k + "_teacher": self._last_eval_results_teacher[k]
    #             for k in self._last_eval_results_teacher.keys()
    #         }
    #         return _last_eval_results_teacher

    #     ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
    #                test_and_save_results_student))
    #     ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
    #                test_and_save_results_teacher))

    #     if comm.is_main_process():
    #         # run writers in the end, so that evaluation metrics are written
    #         ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
    #     return ret
     


    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA
    #     # Only support some R-CNN models.
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = GeneralizedRCNNWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res