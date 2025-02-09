"""
This example follows the experimental settings of the GPT-2 PubMed experiments in the ICML 2024 Spotlight paper,
"Differentially Private Synthetic Data via Foundation Model APIs 2: Text" (https://arxiv.org/abs/2403.01749).

The ``model_name_or_path`` parameter can be set to other models on HuggingFace. Note that we use the FastChat
library (https://github.com/lm-sys/FastChat) to manage the conversation template. If the conversation template of your
desired model is not available in FastChat, please register the conversation template in the FastChat library. See the
following link for an example:
https://github.com/microsoft/DPSDA/blob/main/pe/llm/huggingface/register_fastchat/gpt2.py

For detailed information about parameters and APIs, please consult the documentation of the Private Evolution library:
https://microsoft.github.io/DPSDA/.
"""

from pe.data.text import PubMed
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import HuggingfaceLLM
from pe.embedding.text import SentenceTransformer
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    exp_folder = "results/text/pubmed_huggingface"
    current_folder = os.path.dirname(os.path.abspath(__file__))

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = PubMed(root_dir="/tmp/data/pubmed")
    llm = HuggingfaceLLM(max_completion_tokens=448, model_name_or_path="gpt2", temperature=1.0)
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, "variation_api_prompt.json"),
    )
    embedding = SentenceTransformer(model="sentence-t5-base")
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=0,
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=6, next_variation_api_fold=6, keep_selected=True, selection_mode="rank"
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(
        priv_data=data, embedding=embedding, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[2000] * 11,
        delta=delta,
        epsilon=1.0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
