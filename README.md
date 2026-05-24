# CBP

This code contains Pytorch implementation of [CBP](https://ieeexplore.ieee.org/abstract/document/11488056/):

> CBP employs counterfactual inference to exclusively estimate the impact of bias and personalization, especially in the context of repurchasing behavior. This helps balance the reduction of bias with the enhancement of individual user preferences in next-basket prediction.

## Environments

torch 1.10.1+cuda 11.2.
python 3.6.13.
numpy 1.23.4.
scipy 1.5.4.
scikit-learn 0.23.2.
RTX3090.

We suggest you create a new environment with: conda create -n CBP python=3.6



## Dataset

* Tafeng: https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Gowalla: https://snap.stanford.edu/data/loc-Gowalla.html
* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis



## Running the CBP

```python
$ python main.py --dataset TaFeng --lr 0.01 --l2 0.00001 --alpha 0.1 --beta 0.2 --batch_size 100 --dim 32
$ python main.py --dataset Dunnhumby --lr 0.0001 --l2 0.001 --alpha 0.2 --beta 0.6 --batch_size 100 --dim 32
$ python main.py --dataset Gowalla --lr 0.0001 --l2 0.00001 --alpha 0.1 --beta 0.8 --batch_size 100 --dim 32
$ python main.py --dataset Instacart --lr 0.01 --l2 0.00001 --alpha 0.3 --beta 0.4 --batch_size 100 --dim 32
```

## Reference

```
@article{deng2026confounder,
  title={Confounder Balance in Next Basket Prediction},
  author={Deng, Zhiying and Li, Jianjun and Liu, Wei and Zhao, Juan},
  journal={IEEE Transactions on Cybernetics},
  year={2026},
  publisher={IEEE}
}
```

