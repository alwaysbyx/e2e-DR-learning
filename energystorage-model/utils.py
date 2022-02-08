import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from itertools import accumulate
import cvxpy as cp
import gurobipy
import numpy as np


class OptLayer(nn.Module):
    """Identical to the example in deep implicity layer tutorial
    http://implicit-layers-tutorial.org/differentiable_optimization/
    """

    def __init__(
        self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts
    ):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts

        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [
            ineq(*variables, *parameters) <= 0 for ineq in inequalities
        ]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(
            cp.Minimize(objective(*variables, *parameters)),
            self.cp_inequalities + self.cp_equalities,
        )

    def forward(self, *batch_params):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(batch_params[0].shape[0]):
            # solve the optimization problem and extract solution + dual variables
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()
                self.problem.solve(**self.cvxpy_opts)
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [
                    torch.tensor(c.dual_value).type_as(params[0])
                    for c in self.cp_inequalities
                ]
                nu = [
                    torch.tensor(c.dual_value).type_as(params[0])
                    for c in self.cp_equalities
                ]

            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z, lam, nu] for a in b])

            def mat(x):
                sz = [0] + list(
                    accumulate([a.numel() for b in [z, lam, nu] for a in b])
                )
                val = [x[a:b] for a, b in zip(sz, sz[1:])]
                return (
                    [val[i].view_as(z[i]) for i in range(len(z))],
                    [val[i + len(z)].view_as(lam[i]) for i in range(len(lam))],
                    [val[i + len(z) + len(lam)].view_as(nu[i]) for i in range(len(nu))],
                )

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                L = (
                    self.objective(*z, *params)
                    + sum((u * v).sum() for u, v in zip(lam, g))
                    + sum((u * v).sum() for u, v in zip(nu, dnu))
                )
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i] * g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            # compute residuals and re-engage autograd tape
            y = vec(z, lam, nu)
            y = y - vec(
                *kkt(
                    [z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params
                )
            )

            # compute jacobian and backward hook
            J.append(
                autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y)
            )
            y.register_hook(
                lambda grad, b=batch: torch.solve(grad[:, None], J[b].transpose(0, 1))[
                    0
                ][:, 0]
            )
# +torch.diag(torch.as_tensor(np.full(J[b].shape[0], 1e-10)))
            out.append(mat(y)[0])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        return out[0] if len(out) == 1 else tuple(out)


class PolytopeProjection(nn.Module):
    """This function is only used in main_eta.py.
    Using this layer, we train c1, c2, E1, E2, and eta, other parameters are infered from historical data."""

    def __init__(self, P1, P2, T):
        super().__init__()
        self.c1 = nn.Parameter(10 * torch.ones(1))
        self.c2 = nn.Parameter(10 * torch.ones(1))
        self.E1 = nn.Parameter(1 * torch.ones(1))
        self.E2 = nn.Parameter(-1 * torch.ones(1))
        self.eta = nn.Parameter(0.9 * torch.ones(1))
        eps = 1e-5

        obj = (
            lambda d, p, price, c1, c2, E1, E2, eta: -price @ (d - p)
            + c1 * cp.sum(d)
            + cp.sum_squares(cp.sqrt(c2) * d)
            if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 * torch.sum(d) + c2 * torch.sum(d ** 2)
        )

        ineq1 = (
            lambda d, p, price, c1, c2, E1, E2, eta: p
            - torch.ones(T, dtype=torch.double) * P1
        )
        ineq2 = (
            lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(T, dtype=torch.double)
            * 0
            - p
        )
        ineq3 = (
            lambda d, p, price, c1, c2, E1, E2, eta: d
            - torch.ones(T, dtype=torch.double) * P2
        )
        ineq4 = (
            lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(T, dtype=torch.double)
            * 0
            - d
        )
        ineq5 = (
            lambda d, p, price, c1, c2, E1, E2, eta: torch.tril(
                torch.ones(T, T, dtype=torch.double)
            )
            @ (eta * p - d / eta)
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
            - torch.ones(T, dtype=torch.double) * E1
        )
        ineq6 = lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(
            T, dtype=torch.double
        ) * E2 - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))


        self.layer = OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            [
                cp.Parameter(T,),
                cp.Parameter(1),
                cp.Parameter(1),
                cp.Parameter(1),
                cp.Parameter(1),
                cp.Parameter(1),
            ],
            obj,
            [ineq1, ineq2, ineq3, ineq4, ineq5, ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )

    def forward(self, price):
        return self.layer(
            price,
            self.c1.expand(price.shape[0], *self.c1.shape),
            self.c2.expand(price.shape[0], *self.c2.shape),
            self.E1.expand(price.shape[0], *self.E1.shape),
            self.E2.expand(price.shape[0], *self.E2.shape),
            self.eta.expand(price.shape[0], *self.eta.shape),
        )


class PolytopeProjectionETA(nn.Module):
    """This function is only used in main_eta.py.
    Using this layer, we only train c1 and c2, other parameters are infered from historical data.
    
    """

    def __init__(self, P1, P2, E1, E2, eta, T):
        super().__init__()
        self.c1 = nn.Parameter(10 * torch.ones(1))
        self.c2 = nn.Parameter(100 * torch.ones(1))
        eps = 1e-5

        obj = (
            lambda d, p, price, c1, c2: -price @ (d - p)
            + c1 * cp.sum(d)
            + cp.sum_squares(cp.sqrt(c2) * d)
            if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 * torch.sum(d) + c2 * torch.sum(d ** 2)
        )

        ineq1 = lambda d, p, price, c1, c2: p - torch.ones(T, dtype=torch.double) * P1
        ineq2 = lambda d, p, price, c1, c2: torch.ones(T, dtype=torch.double) * 0 - p
        ineq3 = lambda d, p, price, c1, c2: d - torch.ones(T, dtype=torch.double) * P2
        ineq4 = lambda d, p, price, c1, c2: torch.ones(T, dtype=torch.double) * 0 - d
        ineq5 = (
            lambda d, p, price, c1, c2: torch.tril(torch.ones(T, T, dtype=torch.double))
            @ (eta * p - d / eta)
            - torch.ones(T, dtype=torch.double) * E1
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
        )
        ineq6 = lambda d, p, price, c1, c2: torch.ones(
            T, dtype=torch.double
        ) * E2 - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))

        self.layer = OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            [cp.Parameter(T,), cp.Parameter(1), cp.Parameter(1)],
            obj,
            [ineq1, ineq2, ineq3, ineq4, ineq5, ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )

    def forward(self, price):
        return self.layer(
            price,
            self.c1.expand(price.shape[0], *self.c1.shape),
            self.c2.expand(price.shape[0], *self.c2.shape),
        )


def data_generator(
    c1_value,
    c2_value,
    upperbound_p,
    lowerbound_p,
    upperbound_e,
    lowerbound_e,
    initial_e,
    efficiency,
    price_hist,
    N=1,
    T=288,
):
    """Generate data from the following optimization problem, some of the args could be writing in kwargs

    Args:
        c1_value (float): linear discharge cost
        c2_value (float): quadratic discharge cost
        upperbound_p (float): power upper bound
        lowerbound_p (float): power lower bound
        upperbound_e (float): energy upper bound
        lowerbound_e (float): energy lower bound
        initial_e (float): inital energy level
        efficiency (float): energy storage efficiency
        price_hist (Dataframe): real-time price
        N (int, optional):  Defaults to 1.
        T (int, optional): Defaults to 288.

    Returns:
        [dataframes]: dataframes of price, discharge and charge
    """
    #

    df_price = np.zeros((N, T))
    df_d = np.zeros((N, T))
    df_p = np.zeros((N, T))
    index = 0
    dates = np.random.choice(365,N,replace=False)
    for i in dates:
        price = np.array(price_hist.RTP[i * 288 : (i + 1) * 288])
        price = np.mean(price.reshape(24,-1),axis=1)
        # price = np.array(price_hist.DAP[i * 288 : (i + 1) * 288 : 12])

        c1 = c1_value
        c2 = c2_value
        P1 = upperbound_p
        P2 = lowerbound_p
        E1 = upperbound_e
        E2 = lowerbound_e
        e0 = initial_e
        eta = efficiency
        # eps = leak

        # define the variable and objective function
        p = cp.Variable(T)
        d = cp.Variable(T)
        f = -price @ (d - p) + c1 * cp.sum(d) + c2 * cp.sum_squares(d)
        # define constraints
        cons = [
            p <= np.ones(T,) * P1,
            p >= np.ones(T,) * P2,
            d <= np.ones(T,) * P1,
            d >= np.ones(T,) * P2,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) <= np.ones(T,) * (E1 - e0),
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) >= np.ones(T,) * (E2 - e0),
            # np.tril(np.ones((T, T))) @ (eta * p - d / eta) - np.arange(eps, (T+1)*eps, eps) <= np.ones(T,) * (E1 - e0),
            # np.tril(np.ones((T, T))) @ (eta * p - d / eta) - np.arange(eps, (T+1)*eps, eps) >= np.ones(T,) * (E2 - e0),
        ]

        cp.Problem(cp.Minimize(f), cons).solve(
            solver="GUROBI",
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=1000000000,
        )

        df_price[index, :] = price.T
        df_d[index, :] = d.value
        df_p[index, :] = p.value

        index = index + 1

    return df_price, df_d, df_p

def data_generator_val(
    c1_value,
    c2_value,
    upperbound_p,
    lowerbound_p,
    upperbound_e,
    lowerbound_e,
    efficiency,
    price_hist,
    N=1,
    T=288,
):
    """Generate data from the following optimization problem, some of the args could be writing in kwargs

    Args:
        c1_value (float): linear discharge cost
        c2_value (float): quadratic discharge cost
        upperbound_p (float): power upper bound
        lowerbound_p (float): power lower bound
        upperbound_e (float): energy upper bound
        lowerbound_e (float): energy lower bound
        initial_e (float): inital energy level
        efficiency (float): energy storage efficiency
        price_hist (Dataframe): real-time price
        N (int, optional):  Defaults to 1.
        T (int, optional): Defaults to 288.

    Returns:
        [dataframes]: dataframes of price, discharge and charge
    """
    #

    df_d = np.zeros((N, T))
    df_p = np.zeros((N, T))
    index = 0
    dates = np.random.choice(365,N,replace=False)
    for i in range(N):
        price = price_hist[i]

        # price = np.array(price_hist.DAP[i * 288 : (i + 1) * 288 : 12])

        c1 = c1_value
        c2 = c2_value
        P1 = upperbound_p
        P2 = lowerbound_p
        E1 = upperbound_e
        E2 = lowerbound_e
        eta = efficiency
        # eps = leak

        # define the variable and objective function
        p = cp.Variable(T)
        d = cp.Variable(T)
        f = -price @ (d - p) + c1 * cp.sum(d) + c2 * cp.sum_squares(d)
        # define constraints
        cons = [
            p <= np.ones(T,) * P1,
            p >= np.ones(T,) * P2,
            d <= np.ones(T,) * P1,
            d >= np.ones(T,) * P2,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) <= np.ones(T,) * E1 ,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) >= np.ones(T,) * E2 ,
        ]

        cp.Problem(cp.Minimize(f), cons).solve(
            solver="GUROBI",
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=1000000000,
        )

        df_d[index, :] = d.value
        df_p[index, :] = p.value

        index = index + 1

    return df_d, df_p

# def get_batch(X,Y,M):
#     N = len(Y)
#     valid_indices = np.array(range(N))
#     batch_indices = np.random.choice(valid_indices,size=M,replace=False)
#     batch_xs = X[batch_indices,:]
#     batch_ys = Y[batch_indices]
#     return batch_xs, batch_ys

# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]
