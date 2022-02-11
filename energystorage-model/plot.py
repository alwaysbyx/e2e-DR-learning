import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

## plot parameter MAE/MAPE 
# error = pd.DataFrame(columns=("c1", "c2", "E1", "E2", "eta")) #initialize relative error dataframe
# for i in range(10):
#     result = np.load("./Results/data%d/data.npz"%(i+1))
#     learning = pd.read_csv("./Results/data%d/learning.csv"%(i+1))
#     # result_N20 = np.load("./Results/data4/result4_N20.npz")

#     theta = result['paras'] #parameter ground truth
#     hattheta = learning.iloc[-1] #parameter prediction

#     e_c1 = np.log(abs(theta[0][0]-hattheta['c1'])/theta[0][0])
#     e_c2 = np.log(abs(theta[0][1]-hattheta['c2'])/theta[0][1])
#     e_E1 = np.log(abs(theta[0][2]-4*hattheta['E1'])/theta[0][2])
#     e_E2 = np.log(abs(theta[0][2]+4*hattheta['E2'])/theta[0][2])
#     e_eta = np.log(abs(theta[0][3]-hattheta['eta'])/theta[0][3])
#     # e_c1 = abs(theta[0][0]-hattheta['c1'])
#     # e_c2 = abs(theta[0][1]-hattheta['c2'])
#     # e_E1 = abs(theta[0][2]-4*hattheta['E1'])/4
#     # e_E2 = abs(theta[0][2]+4*hattheta['E2'])/4
#     # e_eta = abs(theta[0][3]-hattheta['eta'])
#     error.loc[i] = [e_c1, e_c2, e_E1, e_E2, e_eta]
# # error.to_csv('absolute_error.csv')
# fig = px.box(error, points="all",width=800, height=400)
# fig.update_layout(
#     xaxis = dict(
#         tickmode = 'array',
#         tickvals = [0, 1, 2, 3, 4],
#         ticktext = [r'$c_1$',r'$c_2$',r'$\overline{E_m}$',r'$\underline{E_m}$',r'$\eta$'],
#         )
# )
# fig.update_layout(
#     xaxis_title="Paramters",
#     yaxis_title="log-MAE",
#     font=dict(
#         family="Courier New, monospace",
#         size=20,
#     )
# )
# fig.show()

## plot prediction error
MLP = pd.read_csv("./Results/MLP_val_loss.csv")
OptNet = pd.read_csv("./Results/OptNet_val_loss.csv")

# y1 = np.log(MLP.iloc[:,1:7].mean().values)
y1 =np.log(MLP.iloc[:,1:7].quantile(.5).values).tolist()
y1_upper =np.log(MLP.iloc[:,1:7].quantile(.8).values).tolist()
y1_lower = np.log(MLP.iloc[:,1:7].quantile(.2).values).tolist()
# y2 = np.log(OptNet.iloc[:,1:7].mean().values)
y2 =np.log(OptNet.iloc[:,1:7].quantile(.5).values).tolist()
y2_upper =np.log(OptNet.iloc[:,1:7].quantile(.8).values).tolist()
y2_lower = np.log(OptNet.iloc[:,1:7].quantile(.2).values).tolist()
x =[1,100,200,300,400,500]

fig = go.Figure([
    go.Scatter(
        x=x,
        y=y1,
        line=dict(color='rgb(255, 149, 0)'),
        mode='lines',
        name="NN"
    ),
    go.Scatter(
        x=x,
        y=y2,
        line=dict(color='rgb(0, 128, 255)'),
        mode='lines',
        name="OptNet"
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y1_upper+y1_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(255, 149, 0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y2_upper+y2_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0, 128, 255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
])
fig.update_xaxes(range=[0, 500])

# fig.update_layout(
#     xaxis = dict(
#         tickmode = 'array',
#         tickvals = [0, 1, 2, 3, 4],
#         ticktext = [r'$c_1$',r'$c_2$',r'$\overline{E_m}$',r'$\underline{E_m}$',r'$\eta$'],
#         )
# )
# fig.update_layout(
#     xaxis_title="Paramters",
#     yaxis_title="log-MAE",
#     font=dict(
#         family="Courier New, monospace",
#         size=20,
#     )
# )
fig.show()