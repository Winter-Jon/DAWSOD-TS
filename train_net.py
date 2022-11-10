import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, hooks
from detectron2.evaluation import verify_results

from dawsod.config.config import add_extra_config
from dawsod.engine.trainer import DAWSOD_Trainer

# hacky way to register
from dawsod.data.datasets import cdod
from dawsod.modeling.meta_arch.rcnn import DAWSOD_RCNN
from dawsod.engine.trainer import DAWSOD_Trainer
from dawsod.modeling.proposal_generator.rpn import DAWSOD_RPN
from dawsod.modeling.roi_heads.roi_heads import DAWSOD_ROIHeads
from dawsod.modeling.roi_heads.box_head import DAWSOD_FastRCNNConvFCHead


# from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
os.environ["DETECTRON2_DATASETS"] = "/data/users/jiangwentao/Repos/datasets"
os.environ['CUDA_VISIBLE_DEVICES']='6,7'

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_extra_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DAWSOD_Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DAWSOD_Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(DAWSOD_Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = DAWSOD_Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

# def main(args):
#     cfg = setup(args)
#     if cfg.SEMISUPNET.Trainer == "ateacher":
#         Trainer = ATeacherTrainer
#     elif cfg.SEMISUPNET.Trainer == "baseline":
#         Trainer = BaselineTrainer
#     else:
#         raise ValueError("Trainer Name is not found.")

#     if args.eval_only:
#         if cfg.SEMISUPNET.Trainer == "ateacher":
#             model = Trainer.build_model(cfg)
#             model_teacher = Trainer.build_model(cfg)
#             ensem_ts_model = EnsembleTSModel(model_teacher, model)

#             DetectionCheckpointer(
#                 ensem_ts_model, save_dir=cfg.OUTPUT_DIR
#             ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
#             res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

#         else:
#             model = Trainer.build_model(cfg)
#             DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#                 cfg.MODEL.WEIGHTS, resume=args.resume
#             )
#             res = Trainer.test(cfg, model)
#         return res

#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=args.resume)

#     return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )