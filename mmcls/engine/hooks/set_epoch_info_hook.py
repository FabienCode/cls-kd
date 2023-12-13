# Copyright (c) OpenMMLab. All rights reserved
import warnings

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmcls.models import BaseRetriever
from mmcls.registry import HOOKS


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model.module.distill_losses.loss_mvkd
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch)
