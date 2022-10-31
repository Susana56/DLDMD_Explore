"""
    Author:
        Jay Lago, SDSU, 2021
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as tfa
import tensorflow.keras.initializers as tfi
from scipy.integrate import solve_ivp
import pickle
import datetime as dt
import os
import sys
sys.path.insert(0, '../../')

conv = True
if conv:
    import DLDMD_TRY_CNN as dl
else:
    import DLDMD as dl

import LossDLDMD as lf
import Data as dat
import Training as tr

# ==============================================================================
# Setup
# ==============================================================================
DEVICE = '/GPU:3'
GPUS = tf.config.experimental.list_physical_devices('GPU')
if GPUS:
    try:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    DEVICE = '/CPU:0'

tf.keras.backend.set_floatx('float64')  # !! Set precision for the entire model here
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Num GPUs available: {}".format(len(GPUS)))
print("Training at precision: {}".format(tf.keras.backend.floatx()))
print("Training on device: {}".format(DEVICE))

# ==============================================================================
# 638 Project Params
# ==============================================================================
gaussian_scale = 0
num_epoch = 100
latent_dim = 8

# ==============================================================================
# Initialize hyper-parameters and Koopman model
# ==============================================================================
# General parameters
hyp_params = dict()
hyp_params['conv'] = conv
hyp_params['gaussian_scale'] = gaussian_scale
hyp_params['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp_params['experiment'] = 'lorenz63'
hyp_params['plot_path'] = f'./training_results/{hyp_params["experiment"]}_{hyp_params["sim_start"]}_{latent_dim}_{np.log10(gaussian_scale)}' + \
    ("_conv" if conv == True else "_dense")
hyp_params['model_path'] = f'./trained_models/{hyp_params["experiment"]}_{hyp_params["sim_start"]}_{latent_dim}_{np.log10(gaussian_scale)}' + \
    ("_conv" if conv == True else "_dense")
hyp_params['device'] = DEVICE
hyp_params['precision'] = tf.keras.backend.floatx()
hyp_params['seed'] = 1984
hyp_params['num_init_conds'] = 1500
hyp_params['num_train_init_conds'] = 1000
hyp_params['num_val_init_conds'] = 1024//10
hyp_params['num_test_init_conds'] = 2048//10
hyp_params['time_final'] = 10
hyp_params['delta_t'] = 0.05
hyp_params['num_time_steps'] = int(hyp_params['time_final']/hyp_params['delta_t'])
hyp_params['num_pred_steps'] = hyp_params['num_time_steps']
hyp_params['save_every'] = 10   # Model save frequency (epochs)
hyp_params['plot_every'] = 10   # Model diagnostic plot frequency (epochs)

# Autoencoder training parameters
hyp_params['max_epochs'] = num_epoch
hyp_params['optimizer'] = 'adam'
hyp_params['batch_size'] = 64
hyp_params['phys_dim'] = 3
hyp_params['latent_dim'] = latent_dim
hyp_params['hidden_activation'] = tfa.elu
hyp_params['bias_initializer'] = tfi.zeros
hyp_params['num_en_layers'] = 2
hyp_params['num_en_neurons'] = 128
hyp_params['kernel_init_enc'] = tfi.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['kernel_init_dec'] = tfi.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['ae_output_activation'] = tfa.linear

# Loss Function Parameters
hyp_params['a1'] = tf.constant(1, dtype=hyp_params['precision'])  # Reconstruction
hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])  # DMD
hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])  # Prediction
hyp_params['a4'] = tf.constant(1e-9, dtype=hyp_params['precision'])  # L-2 on weights

# Learning rate
hyp_params['lr'] = 1e-4

# Set seed for rng
tf.random.set_seed(hyp_params['seed'])

# Initialize the Koopman model and loss
myMachine = dl.DLDMD_TRY_CNN(hyp_params)
myLoss = lf.LossDLDMD(hyp_params)

# ==============================================================================
# Generate / load data
# ==============================================================================
data_fname = 'data_lorenz.pkl'
if False:#os.path.exists(data_fname):
    # Load data from file
    data = pickle.load(open(data_fname, 'rb'))
    data = tf.cast(data, dtype=hyp_params['precision'])
else:
    # Create new data
    # data = dat.data_maker_lorenz63(
        # x1min=-15, x1max=15, x2min=-20, x2max=20, x3min=0, x3max=40,
        # num_ic=hyp_params['num_init_conds'],
        # dt=hyp_params['delta_t'],
        # tf=hyp_params['time_final'],
        # seed=hyp_params['seed'],
    # )
    def lorenz(t, x):
        return np.array([10*(x[1] - x[0]), x[0]*(28-x[2]) - x[1], x[0]*x[1] -8*x[2]/3])
    icx = np.random.uniform(-15, 15, hyp_params['num_init_conds'])
    icy = np.random.uniform(-20, 20, hyp_params['num_init_conds'])
    icz = np.random.uniform(-5, 5, hyp_params['num_init_conds'])
    tspan = np.array([0, hyp_params['time_final']])
    dts = np.arange(0, hyp_params['time_final'], hyp_params['delta_t'])
    X = np.zeros(shape=(hyp_params['num_init_conds'], hyp_params['phys_dim'], hyp_params['num_time_steps']))
    for ii, ic in enumerate(zip(icx, icy, icz)):
        tmp = solve_ivp(lorenz, t_span=tspan, y0=ic, method='RK45', t_eval=dts, vectorized=True)
        X[ii, :, :] = tmp.y
    data = tf.transpose(X, perm=[0, 2, 1])
    data += np.random.normal(loc=0, scale=gaussian_scale, size=data.shape)
    data = tf.cast(data, dtype=hyp_params['precision'])
    # Save data to file
    pickle.dump(data, open(data_fname, 'wb'))

# Scale
data = (data - np.mean(data)) / (np.max(data) - np.min(data))

# Create training and validation datasets from the initial conditions
shuffled_data = tf.random.shuffle(data)
ntic = hyp_params['num_train_init_conds']
nvic = hyp_params['num_val_init_conds']
train_data = shuffled_data[:ntic, :, :]
val_data = shuffled_data[ntic:ntic+nvic, :, :]
test_data = shuffled_data[ntic+nvic:, :, :]
pickle.dump(train_data, open('data_train.pkl', 'wb'))
pickle.dump(val_data, open('data_val.pkl', 'wb'))
pickle.dump(test_data, open('data_test.pkl', 'wb'))
train_data = tf.data.Dataset.from_tensor_slices(train_data)
val_data = tf.data.Dataset.from_tensor_slices(val_data)
test_data = tf.data.Dataset.from_tensor_slices(test_data)

# Batch and prefetch the validation data to the GPUs
val_set = val_data.batch(hyp_params['batch_size'], drop_remainder=True)
val_set = val_set.prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# Train the model
# ==============================================================================
results = tr.train_model(
    hyp_params=hyp_params, train_data=train_data,
    val_set=val_set, model=myMachine, loss=myLoss,
)
print(results['model'].summary())
exit()
