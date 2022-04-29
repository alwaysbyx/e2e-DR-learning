# e2e-DR-learning
# Install Package
Install the package
```python
pip install -r requirements.txt
```

# Energy storage model
- ESID-data: NYISO 2019 real-time electricity price data.
- Results: Contain 10 random generated data: data1-data10. Figures and post-processing data also in this folder.
- data_generation.py: generate random ground truth parameters. Random select dates in real-time price data to generate true dispatch data for training and validation.
- main.py: using OptNet to learn parameters with training data, check validation loss with learned parameters.
- MLP.py: baseline, using three-layer forward ReLU network, return validation losses of differet data and iteration numbers.
- plot.py: make figures for energy storage models.
- post_processing.py: using differernt iteration numbers OptNet learned parameters to calculate validation loss.
- utils.py: functions used in other scripts.


# Building model
- dataset: contain price and ambient data, from *Fernández-Blanco, R., Morales, J. M., & Pineda, S. (2020). Forecasting the Price-Response of a Pool of Buildings via Homothetic Inverse Optimization* (https://github.com/groupoasys/homothetic)
- baseline_NN.py: Neural network as baseline to predict the power consumation
- gradient_method.py: Our approach
- utils.py: contain the solve function and etc.

## Usage
You can use `--num` to determine the number of experiments, `--choice` to determine whether the model learns the indoor temperature, `--save` to determine whether to save the identified model parameters.
```python
python gradient_method.py # default experiment setting
python gradient_method.py --num 10 --save True --seed 1234 --choice 2
```

# Electricity consumer model
- dataset: contain all the input data in the experiments
- baseline_IO.py: reproducing code for *Inverse optimization approach to the identification of electricity consumer models*
- baselines.py: RNN and MLP model
- gradient_method.py: our model for DR identification

## Usage
You can use `--K` to determine the number of loads, `--noise` to determine the noise level, `--save` to determine whether to save the identified model parameters.
```python
python gradient_method.py # default experiment setting
python gradient_method.py --K 3 --noise 2 --save True
```
For baseline comparison, you can use the following method to save the identified model parameters.
```python
python baseline_IO.py --K --noise 2 --save True
```
