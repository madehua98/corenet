#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional

from torch.distributed.elastic.multiprocessing import errors
import os
from corenet.options.opts import get_training_arguments
from corenet.train_eval_pipelines import (
    TRAIN_EVAL_PIPELINE_REGISTRY,
    BaseTrainEvalPipeline,
)


@errors.record
def callback(train_eval_pipeline: BaseTrainEvalPipeline) -> None:
    """
    This function will be invoked on each gpu worker process.

    Args:
        train_eval_pipeline: Provides major pipeline components. The class to be used is
            configurable by "--train-eval-pipeline.name" opt. By default, an instance of
            ``train_eval_pipelines.TrainEvalPipeline`` will be passed to this function.
    """
    train_sampler = train_eval_pipeline.train_sampler
    train_eval_pipeline.training_engine.run(train_sampler=train_sampler)  # 分两步，先init了training_engine,然后调用default_trainer.py里面的run


def main_worker(args: Optional[List[str]] = None):
    opts = get_training_arguments(args=args)
    pipeline_name = getattr(opts, "train_eval_pipeline.name")  # default
    train_eval_pipeline = TRAIN_EVAL_PIPELINE_REGISTRY[pipeline_name](opts=opts)
    train_eval_pipeline._prepare_model()  #->corenet/train_eval_pipelines/default_train_eval.py
    launcher = train_eval_pipeline.launcher
    launcher(callback)


if __name__ == "__main__":
    os.environ['TMPDIR'] = '/ML-A800/home/guoshuyue/madehua/tmp'
    main_worker()
