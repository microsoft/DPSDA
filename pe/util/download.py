import requests
from tqdm import tqdm


def download(url: str, fname: str, chunk_size=1024):
    """
    From:
    https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
