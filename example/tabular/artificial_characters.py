from dotenv import load_dotenv

from pe.data import TabularCSV
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.population import CompositePopulation
from pe.api import TabularAPI
from pe.embedding import TabularEmbedding
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from pe.callback import TabClassifier
from pe.callback import SaveTabToCSV
from pe.callback import ComputeWSD
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True

if __name__ == "__main__":
    exp_folder = "results/tabular/artificial-characters_composite_population"
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    priv_data = TabularCSV(
        csv_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular/real/artificial-characters/artificial-characters_train.csv",
        metadata_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular/real/artificial-characters/artificial-characters_metadata.json",
    )
    priv_info = priv_data.get_tab_info()

    test_data = TabularCSV(
        csv_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular/real/artificial-characters/artificial-characters_test.csv",
        metadata_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular/real/artificial-characters/artificial-characters_metadata.json",
    )

    num_iterations = 15
    num_samples = 1000

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
    population1 = PEPopulation(
        api=api,
        initial_variation_api_fold=0,
        next_variation_api_fold=1,
        keep_selected=False,
        selection_mode="sample",
        histogram_threshold=0,
    )
    population2 = PEPopulation(
        api=api, initial_variation_api_fold=3, next_variation_api_fold=3, keep_selected=True, selection_mode="rank"
    )
    population = CompositePopulation(populations=[population1] * 5 + [population2] * (num_iterations - 5))

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    save_tab_to_csv = SaveTabToCSV(output_folder=os.path.join(exp_folder, "synthetic_tab"))
    tab_classifier = TabClassifier(
        test_data=test_data, model_name="tabicl", filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    compute_wsd_5way = ComputeWSD(
        priv_data=priv_data,
        degree=5,
        num_samples=num_samples,
        seed=42,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
    )
    compute_wsd_6way = ComputeWSD(
        priv_data=priv_data,
        degree=6,
        num_samples=num_samples,
        seed=42,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
    )
    compute_wsd_7way = ComputeWSD(
        priv_data=priv_data,
        degree=7,
        num_samples=num_samples,
        seed=42,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
    )

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(priv_data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=priv_data,
        population=population,
        histogram=histogram,
        callbacks=[
            save_checkpoints,
            save_tab_to_csv,
            tab_classifier,
            compute_wsd_5way,
            compute_wsd_6way,
            compute_wsd_7way,
        ],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[1000] * num_iterations,
        delta=delta,
        epsilon=1.0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
