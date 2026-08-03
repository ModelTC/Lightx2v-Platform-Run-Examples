#!/usr/bin/env python3
"""Run LTX2 while consuming its lazy VAE output without encoding a file."""

from __future__ import annotations

import torch.distributed as dist
from loguru import logger

from lightx2v import infer
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner


def consume_lazy_video_without_saving(self: LTX2Runner) -> dict[str, None]:
    """Drain rank 0's lazy decoder so RUN pipeline includes actual VAE work."""
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_zero:
        chunk_count = 0
        for _chunk in self.gen_video_final:
            chunk_count += 1
        logger.info(
            "LTX2 no-save benchmark consumed {} decoded video chunks",
            chunk_count,
        )
    return {"video": None}


LTX2Runner.process_images_after_vae_decoder = consume_lazy_video_without_saving


if __name__ == "__main__":
    infer.main()
