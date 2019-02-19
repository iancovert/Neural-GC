# Neural Granger Causality

The `Neural-GC` repository contains code for a neural network based approach to discovering Granger causality in nonlinear time series. The methods are described in [this paper](https://arxiv.org/abs/1802.05842).

## Installation

To install the code, please clone the repository. All you need is `Python 3`, `PyTorch (>= 0.4.0)`, `numpy` and `scipy`.

## Demos

See examples in the Jupyter notebooks `cmlp_lagged_var_demo.ipynb`, `clstm_lorenz_demo.ipynb`, and `crnn_lorenz_demo.ipynb`.

## How it works

The models implemented in this repository, termed cMLP, cLSTM, and cRNN, are neural networks that model multivariate time series by forecasting each sequence separately. During training, sparse penalties on the first hidden layer's weight matrix set groups of parameters to zero, which can be interpreted as discovering Granger causality.

The cMLP model can be trained with three different penalties: group lasso, group sparse group lasso, and hierarchical. The cLSTM and cRNN models both use a group lasso penalty, and differ from one another only in the type of RNN cell they use.

Training models with non-convex loss functions and non-smooth penalties requires a specialized optimization strategy, and we use the generalized iterative shrinkage and thresholding algorithm ([GISTA](https://arxiv.org/abs/1303.4434)), which differs from the better known iterative shrinkage and thresholding algorithm (ISTA) only in its use of a line search criterion. Our implementation begins by performing ISTA steps without checking the line search criterion, and switches to a line search when the objective function fails to decrease sufficiently between loss checks.

## Other information

- Selecting the right degree of regularization can be difficult and time consuming. To see results for many regularization strengths, you may eventually want to run parallel training jobs or use a warm start strategy.
- Pretraining (training without regularization) followed by GISTA can lead to a different result than training directly with GISTA. Given the non-convex objective function, this is unsurprising, because the initialization from pretraining is very different than a random initialization. You may need to experiment to find what works best for you.
- If you want to train a debiased model with the sparsity pattern you've learned, use the `cMLPSparse`, `cLSTMSparse`, and `cRNNSparse` classes.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Alex Tank
- Nicholas Foti
- Ali Shojaie
- Emily Fox

## References

- Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily Fox. Neural Granger Causality for Nonlinear Time Series. *arXiv preprint arXiv:1802.05842*, 2018.