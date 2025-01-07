# A Poor Vision Transformer with Competitive Performance on Small-scale Datasets by Low-Rank Attensions and Low-Dimensional Patch Tokens

Our work real the redundancy among ViTs and prove that low-rank attention and low-dimensional patch tokens can achieve competitive performance on small-scale datasets.

> **Abstract:** 
> Vision Transformers (ViTs) have demonstrated remarkable success on large-scale datasets, but their performance on smaller datasets often falls short of convolutional neural networks (CNNs). This paper explores the design and optimization of Tiny ViTs for small datasets, using CIFAR-10 as a benchmark. We systematically evaluate the impact of data augmentation, patch token initialization, low-rank compression, and multi-class token strategies on model performance. Our experiments reveal that low-rank compression of queries in Multi-Head Latent Attention (MLA) incurs minimal performance loss, indicating redundancy in ViTs. Additionally, introducing multiple CLS tokens improves global representation capacity, boosting accuracy. These findings provide a comprehensive framework for optimizing Tiny ViTs, offering practical insights for efficient and effective designs. Code is available at \url{https://github.com/erow/PoorViTs}.



# main results

|         | Baseline | Augmentation |        |         | Patch |       |  MLA |       |       |       |      |  DDP  |        |              |
|---------|:--------:|:------------:|:------:|:-------:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:------:|:------------:|
|         |          |      -aa     | -mixup | -cutmix |  sin  | Witen |  qkv |   kv  |   qk  |   q   |   k  | bs256 | bs1024 | Lion(bs1024) |
| Runtime |   4658   |     4684     |  4656  |   4651  |  4687 |  4676 | 4805 |  4756 |  4785 |  4716 | 4711 |  3582 |  1371  |     1545     |
| Val/Acc |   93.65  |     92.14    |  93.68 |  92.77  | 93.43 | 93.28 | 91.8 | 92.43 | 92.42 | 93.32 | 92.5 | 93.44 |  92.09 |     92.38    |


## Citation
If you use our work, please consider citing:
```bibtex 

```
<hr>


## References
Our code is build on the repositories of [Train Vision Transformer on Small-scale Datasets](https://github.com/hananshafi/vits-for-small-scale-datasets) and [Vision Transformer for Small-Size Datasets](https://github.com/aanna0701/SPT_LSA_ViT). We thank them for releasing their code.

<hr>

  
