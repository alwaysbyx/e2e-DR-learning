import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

## plot parameter MAE/MAPE 
# error = pd.DataFrame(columns=("c1", "c2", "E1", "E2", "eta")) #initialize relative error dataframe
# for i in range(10):
#     result = np.load("./Results/data%d/data.npz"%(i+1))
#     learning = pd.read_csv("./Results/data%d/learning.csv"%(i+1))
#     # result_N20 = np.load("./Results/data4/result4_N20.npz")

#     theta = result['paras'] #parameter ground truth
#     hattheta = learning.iloc[-1] #parameter prediction

#     e_c1 = abs(theta[0][0]-hattheta['c1'])/theta[0][0]
#     e_c2 = abs(theta[0][1]-hattheta['c2'])/theta[0][1]
#     e_E1 = abs(theta[0][2]-4*hattheta['E1'])/theta[0][2]
#     e_E2 = abs(theta[0][2]+4*hattheta['E2'])/theta[0][2]
#     e_eta = abs(theta[0][3]-hattheta['eta'])/theta[0][3]
#     # e_c1 = abs(theta[0][0]-hattheta['c1'])
#     # e_c2 = abs(theta[0][1]-hattheta['c2'])
#     # e_E1 = abs(theta[0][2]-4*hattheta['E1'])/4
#     # e_E2 = abs(theta[0][2]+4*hattheta['E2'])/4
#     # e_eta = abs(theta[0][3]-hattheta['eta'])
#     error.loc[i] = [e_c1, e_c2, e_E1, e_E2, e_eta]
# # error.to_csv('absolute_error.csv')
# fig = px.box(error)
# fig.update_layout(
#     xaxis = dict(
#         tickmode = 'array',
#         tickvals = [0, 1, 2, 3, 4],
#         ticktext = [r'$\Huge{c_1}$',r'$\Huge{c_2}$',r'$\Huge{\overline{E_m}}$',r'$\Huge{\underline{E_m}}$',r'$\Huge{\eta}$'],
#         ),
#     yaxis = dict(
#         type ='log',
#         tickvals = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
#         ),
# )
# fig.update_layout(
#     xaxis_title="Paramters",
#     yaxis_title="log-MAPE",
#     font=dict(
#         size=40,
#         color = '#3a4142'
#     ),
#     template = "plotly_white"
# )
# pio.write_image(fig, 'relative_error.png', width=1600, height=900)

## plot prediction error
MLP = pd.read_csv("./Results/MLP_val_loss.csv")
OptNet = pd.read_csv("./Results/OptNet_val_loss.csv")

# y1 = np.log(MLP.iloc[:,1:7].mean().values)
y1 = MLP.iloc[:,1:7].quantile(.5).values.tolist()
y1_upper = MLP.iloc[:,1:7].quantile(.8).values.tolist()
y1_lower = MLP.iloc[:,1:7].quantile(.2).values.tolist()
# y2 = np.log(OptNet.iloc[:,1:7].mean().values)
y2 = OptNet.iloc[:,1:7].quantile(.5).values.tolist()
y2_upper = OptNet.iloc[:,1:7].quantile(.8).values.tolist()
y2_lower = OptNet.iloc[:,1:7].quantile(.2).values.tolist()
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
fig.update_layout(
    # autosize=False,
    # width=800,
    # height=600,
    yaxis = dict(
        type ='log',
        tickvals = [100, 1, 0.01, 0.0001, 0.000001, 0.00000001, 0.0000000001],
    ),
    xaxis_title="Iterations",
    yaxis_title="log-Validation Loss",
    font=dict(
        size=40,
        color = '#3a4142'
    ),
    template = "plotly_white"
)
pio.write_image(fig, 'validation_loss.png', width=1600, height=900)

## plot dispatch
# data = np.load("./Results/data1/data.npz")
# result = np.load("./Results/data1/result.npz")
# mlp_predict = np.load("./Results/MLP_predict_0.npy")
# y1 = (result['p_valid'][0]-result['d_valid'][0]).tolist()
# y2 = (result['p_pred'][0]-result['d_pred'][0]).tolist()
# y3 = mlp_predict[0].tolist()
# p_upper = data["price"][100].tolist()
# p_lower = np.zeros(24).tolist()
# x = list(range(1,25))
# fig = make_subplots(specs=[[{"secondary_y": True}]])

# fig.add_trace(
#     go.Scatter(
#         x=x,
#         y=y3,
#         line=dict(color='rgb(255, 149, 0)'),
#         mode='lines',
#         name="NN",
#     ),
#     secondary_y=False,
# )

# fig.add_trace(
#     go.Scatter(
#         x=x,
#         y=y2,
#         line=dict(color='rgb(0, 128, 255)'),
#         mode='lines',
#         name="OptNet"
#     ),
#     secondary_y=False,
# )

# fig.add_trace(
#     go.Scatter(
#         x=x,
#         y=y1,
#         line=dict(color='rgb(255, 0, 0)',dash='dash'),
#         mode='lines',
#         name="True"
#     ),
#     secondary_y=False,
# )

# fig.add_trace(
#     go.Scatter(
#         x=x+x[::-1], # x, then x reversed
#         y=p_upper+p_lower[::-1], # upper, then lower reversed
#         fill='toself',
#         fillcolor='rgba(0, 255, 255,0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         hoverinfo="skip",
#         name = 'price'
#     ),
#     secondary_y=True,
# )

# fig.update_layout(
#     xaxis_title="Hour",
#     yaxis_title="Dispatch (MW)",
#     font=dict(
#         size=40,
#         color = '#3a4142'
#     ),
#     legend=dict(yanchor="top",
#                 y=1.10,
#                 xanchor="left",
#                 x=0.01,
#                 orientation="h"),
#     template = "plotly_white"
# )
# fig.update_yaxes(tickmode = 'linear', dtick = 0.1, range = [-0.5,0.5], secondary_y=False)
# fig.update_yaxes(tickmode = 'linear', dtick = 10, range = [0,100], title_text="Price ($/MWh)", secondary_y=True)

# # fig.show()

# pio.write_image(fig, 'ES_dispatch.png', width=1600, height=900)
