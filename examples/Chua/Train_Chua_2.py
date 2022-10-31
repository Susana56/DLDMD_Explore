"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
import numpy as np
import pickle
import datetime as dt
import scipy
from scipy.integrate import solve_ivp
import os
import sys
sys.path.insert(0, '../../')
import DLDMD as dl
import LossDLDMD as lf
import Training as tr



# ==============================================================================
# Setup
# ==============================================================================
NUM_SAVES = 4        # Number of times to save the model throughout training
NUM_PLOTS = 10      # Number of diagnostic plots to generate while training
DEVICE = '/GPU:0'
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
# ==============================================================================
# ==============================================================================
# Initialize hyper-parameters and Koopman model
# ==============================================================================
# General parameters
hyp_params = dict()
hyp_params['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp_params['experiment'] = 'Chua_2'
hyp_params['plot_path'] = './training_results/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']+ '_' + str(np.log10(gaussian_scale)) + '_noise_scale'
hyp_params['model_path'] = './trained_models/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']+ '_' + str(np.log10(gaussian_scale)) + '_noise_scale'
hyp_params['device'] = DEVICE
hyp_params['precision'] = tf.keras.backend.floatx()
hyp_params['num_init_conds'] = 1500          # take 0 out of the following 4
hyp_params['num_train_init_conds'] = 1000
hyp_params['num_val_init_conds'] = 300
hyp_params['num_test_init_conds'] = 200
hyp_params['time_final'] = 20     # Should I change these
hyp_params['delta_t'] = 1e-2
hyp_params['c_1'] = 15.6     # commonly used parameters for the chaotic Chua attractor
hyp_params['c_2'] = 1
hyp_params['c_3'] = 25.58
hyp_params['M_0'] = -1.143
hyp_params['M_1'] = -0.714
hyp_params['num_time_steps'] = int(hyp_params['time_final']/hyp_params['delta_t'])
hyp_params['num_pred_steps'] = hyp_params['num_time_steps']
hyp_params['max_epochs'] = 100
hyp_params['save_every'] = hyp_params['max_epochs'] // NUM_SAVES
hyp_params['plot_every'] = hyp_params['max_epochs'] // NUM_PLOTS

# Universal network layer parameters (AE & Aux)
hyp_params['optimizer'] = 'adam'
hyp_params['batch_size'] = 256       # make smaller, try running + see error with memory
hyp_params['phys_dim'] = 3          # To get the correct input expected in the DMDLD
hyp_params['latent_dim'] = 8       # dimension gets projected into, will start small and increase over time
hyp_params['hidden_activation'] = tf.keras.activations.relu
hyp_params['bias_initializer'] = tf.keras.initializers.Zeros

# Encoding/Decoding Layer Parameters
hyp_params['num_en_layers'] = 3
hyp_params['num_en_neurons'] = 128
hyp_params['kernel_init_enc'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['kernel_init_dec'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['ae_output_activation'] = tf.keras.activations.linear

# Loss Function Parameters
hyp_params['a1'] = tf.constant(1, dtype=hyp_params['precision'])        # Reconstruction
hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])        # DMD
hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])        # Prediction
hyp_params['a4'] = tf.constant(1e-14, dtype=hyp_params['precision'])    # L-2 on weights

# Learning rate
hyp_params['lr'] = 1e-4

# Initialize the Koopman model and loss
myMachine = dl.DLDMD(hyp_params)
myLoss = lf.LossDLDMD(hyp_params)

# Professor C Matlab code for chatic attractor
# ==============================================================================
# Generate / load data
# ==============================================================================
data_fname = 'Chua_data.pkl'
if False: #os.path.exists(data_fname):
    # Load data from file
    data = pickle.load(open(data_fname, 'rb'))
    data = tf.cast(data, dtype=hyp_params['precision'])
else:
    c_1 = hyp_params['c_1']
    c_2 = hyp_params['c_2']
    c_3 = hyp_params['c_3']
    M_0 = hyp_params['M_0']
    M_1 = hyp_params['M_1']
    def Chua(t, x):
        f_x = M_1 * x[0] + 0.5 * (M_0 - M_1) * (abs(x[0] + 1) - abs(x[0] - 1))
        return[c_1 * (x[1] - x[0] - f_x), c_2*(x[0] - x[1] + x[2]), -c_3*x[1]]
    icx = np.random.uniform(-2, 2, hyp_params['num_init_conds'])
    icy = np.random.uniform(-0.4, 0.4, hyp_params['num_init_conds'])
    icz = np.random.uniform(-3, 3, hyp_params['num_init_conds'])
    tspan = np.array([0, hyp_params['time_final']])
    dts = np.arange(0, hyp_params['time_final'], hyp_params['delta_t'])
    X = np.zeros(shape=(hyp_params['num_init_conds'], 3, hyp_params['num_time_steps']))
    for ii, ic in enumerate(zip(icx, icy, icz)):
        tmp = solve_ivp(Chua, t_span=tspan, y0=ic, method='LSODA', t_eval=dts)
        X[ii, :, :] = tmp.y
    data = tf.transpose(X, perm=[0, 2, 1])
    data += np.random.normal(loc=0, scale=gaussian_scale, size=data.shape)
    data = tf.cast(data, dtype=hyp_params['precision'])
    pickle.dump(data, open(data_fname, 'wb'))

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
results = tr.train_model(hyp_params=hyp_params, train_data=train_data,
                         val_set=val_set, model=myMachine, loss=myLoss)
print(results['model'].summary())
exit()
