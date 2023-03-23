import os
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

_C.MODEL = CN()
_C.MODEL.NUM_ACTIONS = 3
_C.MODEL.NUM_COMMANDS = 4
_C.MODEL.BRANCHED = True

_C.DATA = CN()
_C.DATA.ROOT = "/data/AgentHuman"
_C.DATA.TRAIN_DIR = os.path.join(_C.DATA.ROOT, "SeqTrain")
_C.DATA.TRAIN_IMGS_DIR = os.path.join(_C.DATA.TRAIN_DIR, "images")
_C.DATA.TRAIN_TARGETS_DIR = os.path.join(_C.DATA.TRAIN_DIR, "targets")
_C.DATA.VAL_DIR = os.path.join(_C.DATA.ROOT, "SeqVal")
_C.DATA.VAL_IMGS_DIR = os.path.join(_C.DATA.VAL_DIR, "images")
_C.DATA.VAL_TARGETS_DIR = os.path.join(_C.DATA.VAL_DIR, "targets")

_C.DATA.IMG_WIDTH = 200
_C.DATA.IMG_HEIGHT = 88

_C.TRAIN = CN()
_C.TRAIN.LR = 0.00002
_C.TRAIN.NUM_EPOCHS = 30
_C.TRAIN.BATCH_SIZE = 384
_C.TRAIN.DROPOUT = 0.5
_C.TRAIN.OPTIM = "adam"
_C.TRAIN.PRINT_INTERVAL = 500

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 384
_C.VAL.PRINT_INTERVAL = 500


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
