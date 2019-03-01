import torch
import torch.nn as nn
from copy import deepcopy


class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        super(LSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.linear = nn.Conv1d(hidden, 1, 1)

    def init_hidden(self, batch):
        '''Initialize hidden states for LSTM cell.'''
        device = self.lstm.weight_ih_l0.device
        return (torch.zeros(1, batch, self.hidden, device=device),
                torch.zeros(1, batch, self.hidden, device=device))

    def forward(self, X, hidden=None, truncation=None):
        self.lstm.flatten_parameters()
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        if truncation is None:
            X, hidden = self.lstm(X, hidden)
        else:
            T = X.shape[1]
            X_forward = []
            for t in range(0, T, truncation):
                lstm_out, hidden = self.lstm(X[:, t:min(t+truncation, T)],
                                             hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                X_forward.append(lstm_out)
            X = torch.cat(X_forward, dim=1)

        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class cLSTM(nn.Module):
    def __init__(self, num_series, hidden):
        '''cLSTM model.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        '''
        super(cLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks = [LSTM(num_series, hidden) for _ in range(num_series)]

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X, i=None, hidden=None, truncation=None):
        '''Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          i: index of the time series to forecast.
          hidden: hidden states for LSTM cell.
          truncation: for truncated backpropagation through time.

        Returns:
          pred: predictions from one LSTM or all LSTMs.
          hidden: hidden states from one LSTM or all LSTMs.
        '''
        if i is None:
            if hidden is None:
                hidden = [None for _ in range(self.p)]
            pred = [self.networks[i](X, hidden[i], truncation)
                    for i in range(self.p)]
            pred, hidden = zip(*pred)
            pred = torch.cat(pred, dim=2)
        else:
            pred, hidden = self.networks[i](X, hidden, truncation)
            pred = pred[:, :, 0]

        return pred, hidden

    def GC(self, threshold=True):
        '''Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class cLSTMSparse(nn.Module):
    def __init__(self, num_series, sparsity, hidden):
        '''cLSTM model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          hidden: number of units in LSTM cell.
        '''
        super(cLSTMSparse, self).__init__()
        self.p = num_series
        self.hidden = hidden
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(LSTM(num_inputs, hidden))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X, i=None, hidden=None, truncation=None):
        '''Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          i: index of the time series to forecast.
          hidden: hidden states for LSTM cell.
          truncation: for truncated backpropagation through time.

        Returns:
          pred: predictions from one LSTM or all LSTMs.
          hidden: hidden states from one LSTM or all LSTMs.
        '''
        if i is None:
            if hidden is None:
                hidden = [None for _ in range(self.p)]
            pred = [self.networks[i](X[:, :, self.sparsity[i]], hidden[i],
                                     truncation)
                    for i in range(self.p)]
            pred, hidden = zip(*pred)
            pred = torch.cat(pred, dim=2)
        else:
            X_subset = X[:, :, self.sparsity[i]]
            pred, hidden = self.networks[i](X_subset, hidden, truncation)
            pred = pred[:, :, 0]

        return pred, hidden


def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.lstm.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr * 0.1)))
              * torch.clamp(norm - (lr * lam), min=0.0))


def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.lstm.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer.'''
    return lam * torch.sum(network.linear.weight ** 2)


def train_model_gista(clstm, X, lam, lam_ridge, lr, max_iter, check_every=1000,
                      r=0.8, lr_min=1e-12, sigma=0.5, monotone=False, m=10,
                      truncation=None, switch_tol=1e-3, verbose=1):
    '''Train cLSTM model with GISTA.

    Args:
      clstm: clstm model.
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      truncation: for truncated backpropagation through time.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''
    p = clstm.p
    clstm_copy = deepcopy(clstm)
    loss_fn = nn.MSELoss(reduction='mean')

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p):
        net = clstm.networks[i]
        pred, _ = net(X[:, :-1], truncation=truncation)
        mse = loss_fn(pred, X[:, 1:, i:i+1])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = False

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr
            net = clstm.networks[i]
            net_copy = clstm_copy.networks[i]

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(),
                                             net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it)

                # Check line search criterion.
                pred, _ = net_copy(X[:, :-1], truncation=truncation)
                mse = loss_fn(pred, X[:, 1:, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam)
                    loss = smooth + nonsmooth
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                clstm.networks[i], clstm_copy.networks[i] = net_copy, net

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(clstm.GC().float())))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    return train_loss_list, train_mse_list


def train_model_adam(clstm, X, lr, niter, check_every, truncation=None,
                     verbose=0):
    '''Train model with Adam.'''
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(clstm.parameters(), lr=lr)
    train_loss_list = []

    for it in range(niter):
        # Calculate gradients.
        pred, _ = clstm(X[:, :-1], truncation=truncation)
        loss = loss_fn(pred, X[:, 1:])
        loss.backward()

        optimizer.step()
        clstm.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            train_loss_list.append(loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % loss)

    return train_loss_list
