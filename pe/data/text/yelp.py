import os
import pandas as pd
from collections import namedtuple

from .text_csv import TextCSV
from pe.util import download
import gdown
import csv

DownloadInfo = namedtuple("DownloadInfo", ["url", "type"])


class Yelp(TextCSV):
    """The Yelp dataset in the ICML 2024 Spotlight paper, "Differentially Private Synthetic Data via Foundation
    Model APIs 2: Text" (https://arxiv.org/abs/2403.01749)."""

    #: The download information for the Yelp dataset.
    DOWNLOAD_INFO_DICT = {
        "train": DownloadInfo(url="https://drive.google.com/uc?id=1epLuBxCk5MGnm1GiIfLcTcr-tKgjCrc2", type="gdown"),
        "val": DownloadInfo(
            url=(
                "https://raw.githubusercontent.com/AI-secure/aug-pe/bca21c90921bd1151aa7627e676c906165e205a0/data/"
                "yelp/dev.csv"
            ),
            type="direct",
        ),
        "test": DownloadInfo(
            url=(
                "https://raw.githubusercontent.com/AI-secure/aug-pe/bca21c90921bd1151aa7627e676c906165e205a0/data/"
                "yelp/test.csv"
            ),
            type="direct",
        ),
    }

    def __init__(self, root_dir="data", split="train", **kwargs):
        """Constructor.

        :param root_dir: The root directory of the dataset. If the dataset is not there, it will be downloaded
            automatically. Defaults to "data"
        :type root_dir: str, optional
        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        """
        self._processed_data_path = os.path.join(root_dir, f"{split}_processed.csv")
        self._data_path = os.path.join(root_dir, f"{split}.csv")
        self._download(
            download_info=self.DOWNLOAD_INFO_DICT[split],
            data_path=self._data_path,
            processed_data_path=self._processed_data_path,
        )
        super().__init__(
            csv_path=self._processed_data_path,
            label_columns=["business_category", "review_stars"],
            text_column="text",
            **kwargs,
        )

    def _download(self, download_info, data_path, processed_data_path):
        """Download the dataset.

        :param download_info: The download information
        :type download_info: pe.data.text.yelp.DownloadInfo
        :param data_path: The path to the raw data
        :type data_path: str
        :param processed_data_path: The path to the processed data
        :type processed_data_path: str
        :raises ValueError: If the download type is unknown
        """
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if not os.path.exists(processed_data_path):
            if not os.path.exists(data_path):
                if download_info.type == "gdown":
                    gdown.download(url=download_info.url, output=data_path)
                elif download_info.type == "direct":
                    download(url=download_info.url, fname=data_path)
                else:
                    raise ValueError(f"Unknown download type: {download_info.type}")
            data_frame = pd.read_csv(data_path, dtype=str)
            data_frame["label1"] = data_frame["label1"].str.replace("Business Category: ", "")
            data_frame["label2"] = data_frame["label2"].str.replace("Review Stars: ", "")
            data_frame = data_frame.rename(columns={"label1": "business_category", "label2": "review_stars"})
            data_frame.to_csv(processed_data_path, index=False, quoting=csv.QUOTE_ALL)
