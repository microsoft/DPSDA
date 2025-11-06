from dotenv import load_dotenv

from pe.data import TabularCSV
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api import TabularAPI
from pe.embedding import TabularEmbedding
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from pe.callback import TabClassifier
from pe.callback import SaveTabToCSV
from pe.callback import ComputeTVD
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True

if __name__ == "__main__":
    dataset_name = "artificial-characters"  # "person-activity" or "artificial-characters"
    exp_folder = f"results/tabular/{dataset_name}"
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    priv_data = TabularCSV(
        csv_path=f"https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/{dataset_name}_train.csv",
        metadata_path=f"https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/{dataset_name}_metadata.json",
    )
    priv_info = priv_data.get_tab_info()

    test_data = TabularCSV(
        csv_path=f"https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/{dataset_name}_test.csv",
        metadata_path=f"https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/{dataset_name}_metadata.json",
    )

    num_iterations = 15

    api = TabularAPI(
        info=priv_info,
        mutation_rate_init=0.5,
        mutation_rate_final=0.01,
        decay_type="polynomial",
        gamma=0.2,
        num_iterations=num_iterations,
    )
    embedding = TabularEmbedding(info=priv_info)
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=0,
        backend="torch",
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=3, next_variation_api_fold=3, keep_selected=True, selection_mode="rank"
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(
        priv_data=priv_data, embedding=embedding, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    tab_classifier = TabClassifier(
        test_data=test_data, model_name="tabicl", filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    save_tab_to_csv = SaveTabToCSV(output_folder=os.path.join(exp_folder, "synthetic_tab"))
    compute_tvd_1way = ComputeTVD(
        priv_data=priv_data, degree=1, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    compute_tvd_2way = ComputeTVD(
        priv_data=priv_data, degree=2, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(priv_data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=priv_data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, compute_fid, tab_classifier, save_tab_to_csv, compute_tvd_1way, compute_tvd_2way],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[1000] * num_iterations,
        delta=delta,
        epsilon=1.0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
