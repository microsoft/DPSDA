"""
This example follows the experimental settings of the GPT-3.5 OpenReview experiments in the ICML 2024 Spotlight paper,
"Differentially Private Synthetic Data via Foundation Model APIs 2: Text" (https://arxiv.org/abs/2403.01749), except
that the model is changed from GPT-3.5 to GPT-4o-mini-2024-07-18 as the original GPT-3.5 model version used in the
paper is no longer available.

To run the code, the following environment variables are required:
* OPENAI_API_KEY: OpenAI API key. You can get it from https://platform.openai.com/account/api-keys. Multiple keys can
    be separated by commas, and a key with the lowest current workload will be used for each request.

We can also switch from OpenAI API to Azure OpenAI API by using :py:class:`pe.llm.AzureOpenAILLM` instead
of :py:class:`pe.llm.OpenAILLM`. In that case, the following environment variables are required:
* ``AZURE_OPENAI_API_KEY``: Azure OpenAI API key. You can get it from https://portal.azure.com/. Multiple keys can
    be separated by commas. The key can also be "AZ_CLI", in which case the Azure CLI will be used to authenticate
    the requests, and the environment variable ``AZURE_OPENAI_API_SCOPE`` needs to be set. See Azure OpenAI
    authentication documentation for more information:
    https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints#microsoft-entra-id-authentication
* ``AZURE_OPENAI_API_ENDPOINT``: Azure OpenAI endpoint. Multiple endpoints can be separated by commas. You can get
    it from https://portal.azure.com/.
* ``AZURE_OPENAI_API_VERSION``: Azure OpenAI API version. You can get it from https://portal.azure.com/.
Assuming $x_1$ API keys and $x_2$ endpoints are provided. When it is desired to use $X$ endpoints/API keys, then
$x_1,x_2$ must be equal to either $X$ or 1, and the maximum of $x_1,x_2$ must be $X$. For the $x_i$ that equals to
1, the same value will be used for all endpoints/API keys. For each request, the API key + endpoint pair with the
lowest current workload will be used.

These environment variables can be set in a .env file in the same directory as this script. For example:
```
OPENAI_API_KEY=your_openai_api_key
```
See https://github.com/theskumar/python-dotenv for more information about the .env file.

For detailed information about parameters and APIs, please consult the documentation of the Private Evolution library:
https://microsoft.github.io/DPSDA/.
"""

from dotenv import load_dotenv

from pe.data.text import OpenReview
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import OpenAILLM
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
    exp_folder = "results/text/openreview_openai_api"
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = OpenReview(root_dir="/tmp/data/openreview")
    llm = OpenAILLM(max_completion_tokens=1000, model="gpt-4o-mini-2024-07-18", temperature=1.2, num_threads=4)
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, "variation_api_prompt.json"),
        min_word_count=25,
        word_count_std=30,
        token_to_word_ratio=5,
        max_completion_tokens_limit=1200,
        blank_probabilities=0.5,
    )
    embedding = SentenceTransformer(model="stsb-roberta-base-v2")
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=0,
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=3, next_variation_api_fold=3, keep_selected=True, selection_mode="rank"
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
