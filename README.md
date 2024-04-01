# Differentially Private Synthetic Data via Foundation Model APIs

This repo is a Python library to **generate differentially private (DP) synthetic data without the need of any ML model training**. It is based on the following papers that proposed a new DP synthetic data framework that only utilizes the blackbox inference APIs of foundation models (e.g., Stable Diffusion).

* Differentially Private Synthetic Data via Foundation Model APIs 1: Images  
	[[paper (ICLR 2024)]](https://openreview.net/forum?id=YEhQs8POIo) [[paper (arxiv)](https://arxiv.org/abs/2305.15560)]  
    **Authors:** [[Zinan Lin](https://zinanlin.me/)], [[Sivakanth Gopi](https://www.microsoft.com/en-us/research/people/sigopi/)], [[Janardhan Kulkarni](https://www.microsoft.com/en-us/research/people/jakul/)], [[Harsha Nori](https://www.microsoft.com/en-us/research/people/hanori/)], [[Sergey Yekhanin](http://www.yekhanin.org/)]


#### Potential Use Cases
Given a private dataset, this tool can generate a new DP synthetic dataset that is statistically similar to the private dataset, while ensuring a rigorous privacy guarantee called Differential Privacy. The DP synthetic dataset can replace real data in various use cases where privacy is a concern, for example:
* Sharing them with other parties for collaboration and research.
* Using them in downstream algorithms (e.g., training ML models) in the normal non-private pipeline.
* Inspecting the data directly for easier product debugging or development.


#### Supported Data Types
This repo currently supports the following data types and foundation models.

| Foundation Model APIs | Data Type | Size of Generated Images (`--image_size`) |
|--------|--------|--------|
|    [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) |   Images  | Preferably 512x512 |
|    [improved diffusion](https://github.com/openai/improved-diffusion)    |   Images | 64x64 |
|    [DALLE2](https://platform.openai.com/docs/api-reference/images)    |    Images     | 256x256, 512x512, or 1024x1024 |



## Quick Examples

See the [docker file](docker/Dockerfile) for the environment.

#### CIFAR10 Images
```sh
pushd data; python get_cifar10.py; popd  # Download CIFAR10 dataset
pushd models; ./get_models.sh; popd  # Download the pre-trained improved diffusion model
./scripts/main_improved_diffusion_cifar10_conditional.sh  # Run DP generation
```

#### Camelyon17 Images
```sh
pushd data; python get_camelyon17.py; popd  # Download Camelyon17 dataset
pushd models; ./get_models.sh; popd  # Download the pre-trained improved diffusion model
./scripts/main_improved_diffusion_camelyon17_conditional.sh  # Run DP generation
```


#### Cat Images

* Download the dataset from [https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou](https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou), and put them under `data/cookie` and `data/doudou`.
* For Cat Cookie:
```
./scripts/main_stable_diffusion_cookie.sh  # Run DP generation
```
* For Cat Doudou:
```sh
./scripts/main_stable_diffusion_doudou.sh  # Run DP generation
``` 

> After running the above scripts, the synthetic data will be at `<result_folder>/<Private Evolution iteration>/samples.npz`. The `samples` key in the npz file contains `N` generated images in UINT8 format of shape `N x height x width x 3`. For conditional generation with `K` classes, the samples with index in `[i*N/K, (i+1)*N/K-1]` correspond to the samples of the `i+1`-th class. 

See [scripts folder](scripts) for more examples.


## Detailed Usage

`main.py` is the main script for generation. Please refer to `python main.py --help` for detailed descriptions of the arguments. 

For each foundation model API (e.g., Stable Diffusion, improved diffusion), there could be more arguments. Please use `--api_help` argument, e.g., `python main.py --api stable_diffusion --data_folder data --api_help`, to see detailed descrptions of the API-specific arguments.

See Appendices H, I, J of the [paper](https://arxiv.org/abs/2305.15560) for examples/guidelines of parameter selection.

## Generate DP Synthetic Data for Your Own Dataset
Please put all images in a folder (which can contain any nested folder structure), and the naming of the image files should be `<class label without '_' character>_<the remaining part of the filename>.<jpg/jpeg/png/gif>`. Pass the path of this folder to `--data_folder` argument.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Responsible Uses

This project uses foundation model APIs to create [synthetic image data](https://en.wikipedia.org/wiki/Synthetic_data) with [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) guarantees. Differential privacy (DP) is a formal framework that ensures the output of an algorithm does not reveal too much information about its inputs. Without a formal privacy guarantee, a synthetic data generation algorithm may inadvertently reveal sensitive information about its input datapoints.

Using synthetic data in downstream applications can carry risk. Synthetic data may not always reflect the true data distribution, and can cause harms in downstream applications. Both the dataset and algorithms behind the foundation model APIs may contain various types of bias, leading to potential allocation, representation, and quality-of-service harms. Additionally, privacy violations can still occur if the ε and δ privacy parameters are set inappropriately, or if multiple copies of a sample exist in the seed dataset. It is important to consider these factors carefully before any potential deployments.  
