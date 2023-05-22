# Differentially Private Synthetic Data via Foundation Model APIs

This repo is a Python library to **generate differentially private (DP) synthetic data without the need of any ML model training**. It is based on the following papers that proposed a new DP synthetic data framework which only utilizes the blackbox inference APIs of  foundation models (e.g., Stable Diffusion).

* Differentially Private Synthetic Data via Foundation Model APIs 1: Images  
	[[paper (arxiv)](#)]  
    **Authors:** [[Zinan Lin](https://zinanlin.me/)], [[Sivakanth Gopi](https://www.microsoft.com/en-us/research/people/sigopi/)], [[Janardhan Kulkarni](https://www.microsoft.com/en-us/research/people/jakul/)], [[Harsha Nori](https://www.microsoft.com/en-us/research/people/hanori/)], [[Sergey Yekhanin](http://www.yekhanin.org/)]


#### Supported Data Types
This repo currently supports the following data types and foundation models.

| Data Type | Foundation Model APIs |
|--------|--------|
|    Images    |    [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)    |
|    Images    |    [improved diffusion](https://github.com/openai/improved-diffusion)    |
|    Images    |    [DALLE2](https://platform.openai.com/docs/api-reference/images)    |


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

See [scripts folder](scripts) for more examples.


## Detailed Usage

`main.py` is the main script for generation. Please refer to `python main.py --help` for detailed descriptions of the arguments. For each foundation model API (e.g., Stable Diffusion, improved diffusion), there could be more arguments. Please use `--api_help` argument, e.g., `python main.py --api stable_diffusion --data_folder data --api_help`, to see detailed descrptions of the API-specific arguments.

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
