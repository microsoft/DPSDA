# Private Evolution: Differentially Private Synthetic Data via Foundation Model APIs

This repo is a Python library to **generate differentially private (DP) synthetic data without the need of any ML model training**. It is based on the following papers that proposed a new DP synthetic data framework that only utilizes the blackbox inference APIs of foundation models (e.g., Stable Diffusion, GPT models).

* **Differentially Private Synthetic Data via Foundation Model APIs 1: Images**  
    [[paper (ICLR 2024)]](https://openreview.net/forum?id=YEhQs8POIo) [[paper (arxiv)](https://arxiv.org/abs/2305.15560)]  
    **Authors:** [[Zinan Lin](https://zinanlin.me/)], [[Sivakanth Gopi](https://www.microsoft.com/en-us/research/people/sigopi/)], [[Janardhan Kulkarni](https://www.microsoft.com/en-us/research/people/jakul/)], [[Harsha Nori](https://www.microsoft.com/en-us/research/people/hanori/)], [[Sergey Yekhanin](https://www.microsoft.com/en-us/research/people/yekhanin/)]
* **Differentially Private Synthetic Data via Foundation Model APIs 2: Text**  
    [[paper (ICML 2024 Spotlight)]](https://proceedings.mlr.press/v235/xie24g.html) [[paper (arxiv)](https://arxiv.org/abs/2403.01749)] [[website](https://alphapav.github.io/augpe-dpapitext)]  
    **Authors:** [[Chulin Xie](https://alphapav.github.io/)], [[Zinan Lin](https://zinanlin.me/)], [[Arturs Backurs](https://www.mit.edu/~backurs/)], [[Sivakanth Gopi](https://www.microsoft.com/en-us/research/people/sigopi/)], [[Da Yu](https://dayu11.github.io/)], [[Huseyin Inan](https://www.microsoft.com/en-us/research/people/huinan/)], [[Harsha Nori](https://www.microsoft.com/en-us/research/people/hanori/)], [[Haotian Jiang](https://jhtdavid96.wixsite.com/jianghaotian)], [[Huishuai Zhang](https://huishuai-git.github.io/)], [[Yin Tat Lee](https://yintat.com/)], [[Bo Li](https://aisecure.github.io/)], [[Sergey Yekhanin](https://www.microsoft.com/en-us/research/people/yekhanin/)]
* **Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Models**  
    [[paper (arxiv)](https://arxiv.org/abs/2502.05505)]  
    **Authors:** [[Zinan Lin](https://zinanlin.me/)], [[Tadas Baltrusaitis](https://www.microsoft.com/en-us/research/people/tabaltru/)], [[Sergey Yekhanin](https://www.microsoft.com/en-us/research/people/yekhanin/)]

Please refer to [this repo](https://github.com/fjxmlzn/private-evolution-papers) for the full list of Private Evolution papers and code repositories related to PE.

## Documentation
Please refer to the [documentation](https://microsoft.github.io/DPSDA/) for more details, including the installation instructions, usage, and examples.

## News

* `2/11/2025`: **Image generation with simulator APIs** based on the paper [`Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Models`](https://arxiv.org/abs/2502.05505) has been released in this library!
* `1/8/2025`: **Text generation with foundation model APIs** based on the paper [`Differentially Private Synthetic Data via Foundation Model APIs 2: Text`](https://arxiv.org/abs/2403.01749) has been integrated into the library! If you want to reproduce the results in the [paper](https://arxiv.org/abs/2403.01749), please refer to [our original codebase](https://github.com/AI-secure/aug-pe).
* `11/21/2024`: The refactored codebase for **image generation with foundation model APIs** based on the paper [`Differentially Private Synthetic Data via Foundation Model APIs 1: Images`](https://arxiv.org/abs/2305.15560) has been released! It is completely refactored to be more modular and easier to use and extend. The code originally published with the [paper](https://arxiv.org/abs/2305.15560) has been moved to the [deprecated](https://github.com/microsoft/DPSDA/tree/deprecated) branch in this repository, which is no longer maintained.

## Citations

If you use this library in your research or work, please cite the following papers:

https://github.com/microsoft/DPSDA/blob/91ccad6e65febd12c84448e3f79d498a20eda7b3/doc/source/getting_started/pe1.bib#L1-L6

https://github.com/microsoft/DPSDA/blob/91ccad6e65febd12c84448e3f79d498a20eda7b3/doc/source/getting_started/pe2.bib#L1-L6

https://github.com/microsoft/DPSDA/blob/91ccad6e65febd12c84448e3f79d498a20eda7b3/doc/source/getting_started/pe3.bib#L1-L6



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

This project uses foundation model APIs to create [synthetic data](https://en.wikipedia.org/wiki/Synthetic_data) with [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) guarantees. Differential privacy (DP) is a formal framework that ensures the output of an algorithm does not reveal too much information about its inputs. Without a formal privacy guarantee, a synthetic data generation algorithm may inadvertently reveal sensitive information about its input datapoints.

Using synthetic data in downstream applications can carry risk. Synthetic data may not always reflect the true data distribution, and can cause harms in downstream applications. Both the dataset and algorithms behind the foundation model APIs may contain various types of bias, leading to potential allocation, representation, and quality-of-service harms. Additionally, privacy violations can still occur if the ε and δ privacy parameters are set inappropriately, or if multiple copies of a sample exist in the seed dataset. It is important to consider these factors carefully before any potential deployments.  
