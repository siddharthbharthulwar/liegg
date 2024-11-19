*"LieGG: Studying Learned Lie Group Generators "* (2022), Moskalev A., Sepliarskaia A., Sosnovik I., Smeulders A.,  *Advances in Neural Information Processing Systems (NeurIPS)*, [[arXiv]](https://arxiv.org/abs/2210.04345)

```bibtex
@inproceedings{moskalev2022liegg,
  title={LieGG: Studying Learned Lie Group Generators},
  author={Moskalev, Artem and Sepliarskaia, Anna and Sosnovik, Ivan and Smeulders, Arnold},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

*Abstract*: Symmetries built into a neural network have appeared to be very beneficial for a wide range of tasks as it saves the data to learn them. We depart from the position that when symmetries are not built into a model a priori, it is advantageous for robust networks to learn symmetries directly from the data to fit a task function. In this paper, we present a method to extract symmetries learned by a neural network and to evaluate the degree to which a network is invariant to them. With our method, we are able to explicitly retrieve learned invariances in a form of the generators of corresponding Lie-groups without prior knowledge of symmetries in the data. We use the proposed method to study how symmetrical properties depend on a neural network's parameterization and configuration. We found that the ability of a network to learn symmetries generalizes over a range of architectures. However, the quality of learned symmetries depends on the depth and the number of parameters.

👾 Check the <a href="https://www.youtube.com/watch?v=BUQ5VNEdrVk">video</a> on the paper. 👾

<p align="center">
  <img width="550" alt="Invariance for Lie algebras" src="./etc/liegg.jpg">
</p>

Acknowledgements
-------

The Robert Bosch GmbH is acknowledged for financial support.

License
-------

Licensed under an MIT license.