"""
This example follows the experimental settings of the CelebA with avatar render experiments in the paper,
"Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Model"
(https://arxiv.org/abs/2502.05505).

For detailed information about parameters and APIs, please consult the documentation of the Private Evolution library:
https://microsoft.github.io/DPSDA/.
"""

from pe.data.image import CelebA
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.image import Avatar
from pe.embedding.image import FLDInception
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import SampleImages
from pe.callback import ComputeFID
from pe.callback import DPImageBenchClassifyImages
from pe.logger import ImageFile
from pe.logger import CSVPrint
from pe.logger import LogPrint

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    exp_folder = "results/image/simulator/celeba_avatar"

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    train_data, val_data = CelebA(root_dir="/tmp/data/celeba/").random_split([145064, 17706], seed=0)
    test_data = CelebA(root_dir="/tmp/data/celeba/", split="test")

    api = Avatar(
        res=32,
        variation_degrees=[
            1.0,
            0.8,
            0.6,
            0.4,
            0.2,
            0.1,
            0.08,
            0.06,
        ],
    )
    fld_inception_embedding = FLDInception()
    histogram = NearestNeighbors(
        embedding=fld_inception_embedding,
        mode="L2",
        lookahead_degree=8,
        api=api,
    )
    population = PEPopulation(api=api, histogram_threshold=1)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    sample_images = SampleImages()
    compute_fld_fid = ComputeFID(priv_data=train_data, embedding=fld_inception_embedding)
    dp_image_bench_classify_images_wrn = DPImageBenchClassifyImages(
        model_name="wrn",
        test_data=test_data,
        val_data=val_data,
    )
    dp_image_bench_classify_images_resnet = DPImageBenchClassifyImages(
        model_name="resnet",
        test_data=test_data,
        val_data=val_data,
    )
    dp_image_bench_classify_images_resnext = DPImageBenchClassifyImages(
        model_name="resnext",
        test_data=test_data,
        val_data=val_data,
    )

    image_file = ImageFile(output_folder=exp_folder)
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(train_data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=train_data,
        population=population,
        histogram=histogram,
        callbacks=[
            save_checkpoints,
            sample_images,
            compute_fld_fid,
            dp_image_bench_classify_images_wrn,
            dp_image_bench_classify_images_resnet,
            dp_image_bench_classify_images_resnext,
        ],
        loggers=[image_file, csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[60000] * 8,
        delta=delta,
        epsilon=10.0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
