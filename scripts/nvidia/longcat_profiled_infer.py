#!/usr/bin/env python3
"""Add the missing LongCat pipeline profile without modifying LightX2V."""

from lightx2v import infer
from lightx2v.models.runners.longcat_image.longcat_image_runner import (
    LongCatImageRunner,
)
from lightx2v.utils.profiler import ProfilingContext4DebugL1

LongCatImageRunner.run_pipeline = ProfilingContext4DebugL1("RUN pipeline")(LongCatImageRunner.run_pipeline)


if __name__ == "__main__":
    infer.main()
