"""
This example follows the experimental settings of the CIFAR10 experiments in the ICLR 2024 paper,
"Differentially Private Synthetic Data via Foundation Model APIs 1: Images" (https://arxiv.org/abs/2305.15560).

For detailed information about parameters and APIs, please consult the documentation of the Private Evolution library:
https://microsoft.github.io/DPSDA/.
"""

from pe.data.image import Cifar10
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.image import ImprovedDiffusion270M
from pe.embedding.image import Inception
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import SampleImages
from pe.callback import ComputeFID
from pe.logger import ImageFile
from pe.logger import CSVPrint
from pe.logger import LogPrint

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    exp_folder = "results/image/diffusion_model/cifar10_improved_diffusion"

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = Cifar10()
    api = ImprovedDiffusion270M(
        variation_degrees=list(range(0, 42, 2)),
        timestep_respacing="100",
    )
    embedding = Inception(res=32, batch_size=100)
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=8,
        api=api,
    )
    population = PEPopulation(api=api, histogram_threshold=10)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    sample_images = SampleImages()
    compute_fid = ComputeFID(priv_data=data, embedding=embedding)

    image_file = ImageFile(output_folder=exp_folder)
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, sample_images, compute_fid],
        loggers=[image_file, csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[50000] * 21,
        delta=1e-5,
        noise_multiplier=5 * np.sqrt(2),
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
