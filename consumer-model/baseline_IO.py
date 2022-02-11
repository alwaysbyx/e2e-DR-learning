import numpy as np
import cvxpy as cp
import xpress as xp
import scipy.io as scio

data = scio.loadmat('dataset/K3pi1N50.mat')
z = data['data']
price = data['price']

N = 20
T = 12
K = 3
z = data['data'][:N]
price = data['price'][:N]
m = xp.problem()

e = [[xp.var(name="e_{0}_{1}".format(i,t), vartype=xp.continuous, lb=0) for t in range(T)] for i in range(N)]
x = [[[xp.var(name="x_{0}_{1}_{2}".format(i,k,t), vartype=xp.continuous, lb=0) for t in range(T)] for k in range(K)] for i in range(N)]
u = [[xp.var(name="u_{0}_{1}".format(k,t), vartype=xp.continuous, lb=0) for t in range(T)] for k in range(K)]
L = [[xp.var(name="L_{0}_{1}".format(k,t), vartype=xp.continuous, lb=0) for t in range(T)] for k in range(K)]
M = [xp.var(name="M_{0}".format(k), vartype=xp.continuous, lb=0) for k in range(K)]
beta = [[[xp.var(name="beta_{0}_{1}_{2}".format(i,k,t), vartype=xp.continuous, lb=0) for t in range(T)] for k in range(K)] for i in range(N)]
alpha = [[xp.var(name="alpha_{0}_{1}".format(i,k), vartype=xp.continuous) for k in range(K)]for i in range(N)]

m.addVariable(e)
m.addVariable(x)
m.addVariable(u)
m.addVariable(L)
m.addVariable(M)
m.addVariable(beta)
m.addVariable(alpha)

m.addConstraint((e[i][t] >= z[i][t] - xp.Sum([x[i][k][t] for k in range(K)]) for i in range(N) for t in range(T)),
                (e[i][t] <= z[i][t] - xp.Sum([x[i][k][t] for k in range(K)]) for i in range(N) for t in range(T)),
                (xp.Sum([x[i][k][t] for t in range(T)]) <= M[k]+0.1 for i in range(N) for k in range(K)),
                (xp.Sum([x[i][k][t] for t in range(T)]) >= M[k]-0.1 for i in range(N) for k in range(K)),
                (x[i][k][t] <= L[k][t] for t in range(T) for k in range(K) for i in range(N)),
                (alpha[i][k] + beta[i][k][t] >= u[k][t] - price[i][t] for i in range(N) for k in range(K) for t in range(T)),
                (xp.Sum([u[k][t]*x[i][k][t] - price[i][t]*x[i][k][t] - L[k][t]*beta[i][k][t] for k in range(K) for t in range(T)]) == xp.Sum([M[k]*alpha[i][k] for k in range(K)]) for i in range(N))
                )

m.setObjective(xp.Sum([e[i][t] for i in range(N) for t in range(T)]), sense=xp.minimize)

print("status: ", m.getProbStatus())
m.nlpoptimize()

print("solution:", m.getSolution())