import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from numpy import linalg 
from itertools import accumulate

def solve(price,u,M,L,k,T=12):
    x = cp.Variable((k,T))
    obj = 3*cp.sum(price@x.T) - cp.sum(u@x.T)
    problem = cp.Problem(cp.Minimize(obj), [x <= L, x >= 0, cp.sum(x,axis=1) == M])
    problem.solve()
    return x.value

class Layer(nn.Module):
    def __init__(self,K,T=12):
        super().__init__()
        
        self.u = nn.Parameter(3*torch.ones(K,T).double())
        self.M = nn.Parameter(torch.Tensor([8,7,9]).double())
        self.L = nn.Parameter(5*torch.ones(K,T).double())
   
        obj = (lambda x1,x2,x3, price, u, M, L: K*cp.sum(price@cp.vstack([x1,x2,x3]).T) - cp.sum(u@cp.vstack([x1,x2,x3]).T)
               if isinstance(x1, cp.Variable) else K*torch.sum(price@torch.vstack([x1,x2,x3]).T) - torch.sum(u@torch.vstack([x1,x2,x3]).T))
        ineq1 = lambda x1,x2,x3, price, u, M, L:  x1 - L[0]
        ineq2 = lambda x1,x2,x3, price, u, M, L:  x2 - L[1]
        ineq3 = lambda x1,x2,x3, price, u, M, L:  x3 - L[2]
        ineq4 = lambda x1,x2,x3, price, u, M, L:  -x1
        ineq5 = lambda x1,x2,x3, price, u, M, L:  -x2
        ineq6 = lambda x1,x2,x3, price, u, M, L:  -x3
        eq1 = lambda x1,x2,x3, price, u, M, L:  cp.sum(cp.vstack([x1,x2,x3]),axis=1) - M if isinstance(x1, cp.Variable) else torch.sum(torch.vstack([x1,x2,x3]),axis=1) - M                                                 
        self.layer = OptLayer([cp.Variable(T), cp.Variable(T), cp.Variable(T)], [cp.Parameter(T), cp.Parameter((K,T)), cp.Parameter(K), cp.Parameter((K,T))],
                              obj, [ineq1,ineq2,ineq3,ineq4,ineq5,ineq6], [eq1])

    def forward(self, price):
        return self.layer(price,
                          self.u.expand(price.shape[0], *self.u.shape),
                          self.M.expand(price.shape[0], *self.M.shape),
                          self.L.expand(price.shape[0], *self.L.shape))


class OptLayer(nn.Module):
    def __init__(self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts
        
        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(cp.Minimize(objective(*variables, *parameters)), 
                                  self.cp_inequalities + self.cp_equalities)
        
    def forward(self, *batch_params):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(batch_params[0].shape[0]):
            # solve the optimization problem and extract solution + dual variables
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i,p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()
                self.problem.solve(**self.cvxpy_opts)
                #print(batch, type(self.variables[0].value))
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_inequalities]
                nu = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_equalities]

            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z,lam,nu] for a in b])

            def mat(x):
                sz = [0] + list(accumulate([a.numel() for b in [z,lam,nu] for a in b]))
                val = [x[a:b] for a,b in zip(sz, sz[1:])]
                return ([val[i].view_as(z[i]) for i in range(len(z))],
                        [val[i+len(z)].view_as(lam[i]) for i in range(len(lam))],
                        [val[i+len(z)+len(lam)].view_as(nu[i]) for i in range(len(nu))])

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                L = (self.objective(*z, *params) + 
                     sum((u*v).sum() for u,v in zip(lam,g)) + sum((u*v).sum() for u,v in zip(nu,dnu)))
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i]*g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            # compute residuals and re-engage autograd tape
            y = vec(z, lam, nu)
            y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))

            # compute jacobian and backward hook
            J.append(autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y))
            y.register_hook(lambda grad,b=batch : torch.solve(grad[:,None], J[b].transpose(0,1))[0][:,0])
            
            
            tmp = torch.stack([x for x in mat(y)[0]])
            o = torch.sum(tmp,axis=0)
            out.append([o])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        return out[0] if len(out) == 1 else tuple(out) 