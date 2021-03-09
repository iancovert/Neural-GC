# Neural Granger Causality

The `Neural-GC` repository contains code for a deep learning-based approach to discovering Granger causality networks in multivariate time series. The methods implemented here are described in [this paper](https://arxiv.org/abs/1802.05842).

## Installation

To install the code, please clone the repository. All you need is `Python 3`, `PyTorch (>= 0.4.0)`, `numpy` and `scipy`.

## Usage

See examples of how to apply our approach in the notebooks `cmlp_lagged_var_demo.ipynb`, `clstm_lorenz_demo.ipynb`, and `crnn_lorenz_demo.ipynb`.

## How it works

The models implemented in this repository, called the cMLP, cLSTM and cRNN, are neural networks that model multivariate time series by forecasting each time series separately. During training, sparse penalties on the input layer's weight matrix set groups of parameters to zero, which can be interpreted as discovering Granger non-causality.

The cMLP model can be trained with three different penalties: group lasso, group sparse group lasso, and hierarchical. The cLSTM and cRNN models both use a group lasso penalty, and they differ from one another only in the type of RNN cell they use.

Training models with non-convex loss functions and non-smooth penalties requires a specialized optimization strategy, and we use a proximal gradient descent approach (ISTA). Our paper finds that ISTA provides comparable performance to two other approaches: proximal gradient descent with a line search (GISTA), which guarantees convergence to a local minimum, and Adam, which converges faster (although it requires an additional thresholding parameter).

## Other information

- Selecting the right regularization strength can be difficult and time consuming. To get results for many regularization strengths, you may want to run parallel training jobs or use a warm start strategy.
- Pretraining (training without regularization) followed by ISTA can lead to a different result than training directly with ISTA. Given the non-convex objective function, this is unsurprising, because the initialization from pretraining is very different than a random initialization. You may need to experiment to find what works best for you.
- If you want to train a debiased model with the learned sparsity pattern, use the `cMLPSparse`, `cLSTMSparse`, and `cRNNSparse` classes.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Alex Tank
- Nicholas Foti
- Ali Shojaie
- Emily Fox

## References

- Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily Fox. "Neural Granger Causality." *Transactions on Pattern Analysis and Machine Intelligence*, 2021.