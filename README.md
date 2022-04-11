# e2e-DR-learning
## Energy storage model
- ESID-data: NYISO 2019 real-time electricity price data.
- Results: Contain 10 random generated data: data1-data10. Figures and post-processing data also in this folder.
- data_generation.py: generate random ground truth parameters. Random select dates in real-time price data to generate true dispatch data for training and validation.
- main.py: using OptNet to learn parameters with training data, check validation loss with learned parameters.
- MLP.py: baseline, using two-layer forward ReLU network, return validation losses of differet data and iteration numbers.
- plot.py: make figures for energy storage models.
- post_processing.py: using differernt iteration numbers OptNet learned parameters to calculate validation loss.
- utils.py: functions used in other scripts.


## Building model
- dataset: contain price and ambient data, from *Fernández-Blanco, R., Morales, J. M., & Pineda, S. (2020). Forecasting the Price-Response of a Pool of Buildings via Homothetic Inverse Optimization* (https://github.com/groupoasys/homothetic)
- baseline_NN.py: Neural network as baseline to predict the power consumation
- gradient_method.py: Our approach
- utils.py: contain the solve function and etc.

## Electricity consumer model
- dataset: contain all the input data in the experiments
- baseline_IO.py: reproducing code for *Inverse optimization approach to the identification of electricity consumer models*
- baseline_NN.py: two forward ReLU NN for behavior prediction
- gradient_method.py: our model for DR identification
