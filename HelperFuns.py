"""
    Author:
        Jay Lago, NIWC/SDSU, 2021
"""
from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from scipy import fft as ft

font = {'family': 'DejaVu Sans', 'size': 18}
matplotlib.rc('font', **font)


def diagnostic_plot(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag):
    if hyp_params['experiment'] == 'pendulum':
        plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag)
    elif hyp_params['experiment'] == 'duffing' or \
            hyp_params['experiment'] == 'van_der_pol':
        plot_3d_latent(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag)
    elif hyp_params['experiment'] in {'Rossler', 'lorenz', 'lorenz63'}:
        plot_full_3d(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag)
    else:
        print("[ERROR] unknown experiment, create new diagnostic plots...")
    frequency_plots(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag)

def frequency_plots(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag):
    skip = 20
    enc = y_pred[0].numpy()[::skip]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]


    times = np.arange(0, hyp_params['time_final'], hyp_params['delta_t'])

    fig, axes = plt.subplots(hyp_params['latent_dim'], 2, sharex='col', figsize=(10, hyp_params['latent_dim']*10/3))
    #st = fig.suptitle(f"Epoch: {epoch}/{hyp_params['max_epochs']}, Noise Mag: {noise_mag:.3f}")

    enc_hat = ft.rfft(enc, axis=1)
    enc_freqs = ft.rfftfreq(enc.shape[1], d=times[1] - times[0])

    axes[-1,0].set_xlabel("$t$", size=15)
    axes[-1,-1].set_xlabel("$\\xi$", size=15)

    axes[0,0].set_title(f"Lifted dimension components, {epoch}/{hyp_params['max_epochs']}", size=15)
    axes[0,1].set_title(f"Noise Mag: {noise_mag}, Lifted dimension FTs", size=15)

    for ii, (ax, dim) in enumerate(zip(axes[:,0], np.transpose(enc, axes=[2,1,0]))):
        ax.plot(times, dim)
        ax.set_ylabel(f"$\\widetilde{{y}}_{ii+1}$")
        ax.set_xlim(*times[[0,-1]])
        ax.grid(True)

    for ii, (ax, dim_hat) in enumerate(zip(axes[:,1], np.transpose(enc_hat, axes=[2,1,0]))):
        ax.plot(enc_freqs, np.abs(dim_hat))
        ax.set_xlim(0, enc_freqs[-1]/5)
        ax.set_ylabel(f"$\\widehat{{\\widetilde{{y}}}}_{ii+1}$")
        ax.grid(True)

    this_plot = save_path[:-4]
    fig.tight_layout()
    fig.savefig(f"{this_plot}_freq.png", dpi=100)
    plt.close(fig)

def plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(40, 20))
    ax = ax.flat
    skip = 5

    # Validation batch
    for ii in np.arange(0, y_true.shape[0], skip):
        ax[0].plot(y_true[ii, :, 0], y_true[ii, :, 1], '-')
    ax[0].scatter(y_true[::skip, 0, 0], y_true[::skip, 0, 1])
    ax[0].grid()
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ax[1].plot(enc_adv_dec[ii, :, 0], enc_adv_dec[ii, :, 1], '-')
    ax[1].scatter(enc_adv_dec[::skip, 0, 0], enc_adv_dec[::skip, 0, 1])
    ax[1].grid()
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    for ii in np.arange(0, enc.shape[0], skip):
        ax[2].plot(enc[ii, :, 0], enc[ii, :, 1], '-')
    ax[2].scatter(enc[::skip, 0, 0], enc[::skip, 0, 1])
    ax[2].grid()
    ax[2].set_xlabel("y1")
    ax[2].set_ylabel("y2")
    ax[2].axis("equal")
    ax[2].set_title("Encoded (y)")

    # Encoded-decoded time series
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ax[3].plot(enc_dec[ii, :, 0], enc_dec[ii, :, 1], '-')
    ax[3].scatter(enc_dec[::skip, 0, 0], enc_dec[::skip, 0, 1])
    ax[3].grid()
    ax[3].set_xlabel("x1")
    ax[3].set_ylabel("x2")
    ax[3].set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ax[4].plot(enc_adv[ii, :, 0], enc_adv[ii, :, 1], '-')
    ax[4].scatter(enc_adv[::skip, 0, 0], enc_adv[::skip, 0, 1])
    ax[4].grid()
    ax[4].set_xlabel("y1")
    ax[4].set_ylabel("y2")
    ax[4].axis("equal")
    ax[4].set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax[5].plot(val_loss, color='k', linewidth=lw, label='total')
    ax[5].set_title("Total Loss")
    ax[5].grid()
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("$log_{10}(L)$")
    ax[5].legend(loc="upper right")

    ax[6].plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax[6].set_title("Recon Loss")
    ax[6].grid()
    ax[6].set_xlabel("Epoch")
    ax[6].set_ylabel("$log_{10}(L_{recon})$")
    ax[6].legend(loc="upper right")

    ax[7].plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax[7].set_title("Prediction Loss")
    ax[7].grid()
    ax[7].set_xlabel("Epoch")
    ax[7].set_ylabel("$log_{10}(L_{pred})$")
    ax[7].legend(loc="upper right")

    ax[8].plot(loss_comps[:, 2], color='g', linewidth=lw, label='dmd')
    ax[8].set_title("DMD")
    ax[8].grid()
    ax[8].set_xlabel("Epoch")
    ax[8].set_ylabel("$log_{10}(L_{dmd})$")
    ax[8].legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}, Noise Mag: {mag:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1],
            mag=noise_mag))

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_3d_latent(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    font = {'family': 'DejaVu Sans', 'size': 10}
    matplotlib.rc('font', **font)

    skip = 3
    fig = plt.figure(figsize=(40, 20))

    # Validation batch
    ax = fig.add_subplot(3, 3, 1)
    for ii in np.arange(0, y_true.shape[0], skip):
        ii = int(ii)
        x1 = y_true[ii, :, 0]
        x2 = y_true[ii, :, 1]
        ax.plot(x1, x2)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.grid()
    ax.set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(3, 3, 2)
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv_dec[ii, :, 0]
        x2 = enc_adv_dec[ii, :, 1]
        ax.plot(x1, x2)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.grid()
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    for ii in np.arange(0, enc.shape[0], skip):
        ii = int(ii)
        x1 = enc[ii, :, 0]
        x2 = enc[ii, :, 1]
        x3 = enc[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    # ax.grid()
    ax.set_title("Encoded (y)")

    # Encoded-decoded time series
    ax = fig.add_subplot(3, 3, 4)
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_dec[ii, :, 0]
        x2 = enc_dec[ii, :, 1]
        ax.plot(x1, x2)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.grid()
    ax.set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv[ii, :, 0]
        x2 = enc_adv[ii, :, 1]
        x3 = enc_adv[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='dmd')
    ax.set_title("DMD")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{dmd})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"Epoch: {epoch}/{hyp_params['max_epochs']}, Learn Rate: {hyp_params['lr']:.5f}," +
        f"Val. Loss: {val_loss[-1]:.3f}, Noise Mag: {noise_mag}", 
        size=25)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_full_3d(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss, noise_mag):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    font = {'family': 'DejaVu Sans', 'size': 10}
    matplotlib.rc('font', **font)

    skip = 3
    fig = plt.figure(figsize=(40, 20))

    # Validation batch
    ax = fig.add_subplot(3, 3, 1, projection='3d')
    for ii in np.arange(0, y_true.shape[0], skip):
        ii = int(ii)
        x1 = y_true[ii, :, 0]
        x2 = y_true[ii, :, 1]
        x3 = y_true[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(3, 3, 2, projection='3d')
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv_dec[ii, :, 0]
        x2 = enc_adv_dec[ii, :, 1]
        x3 = enc_adv_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    for ii in np.arange(0, enc.shape[0], skip):
        ii = int(ii)
        x1 = enc[ii, :, 0]
        x2 = enc[ii, :, 1]
        x3 = enc[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded (y)")

    # Encoded-decoded time series
    ax = fig.add_subplot(3, 3, 4, projection='3d')
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_dec[ii, :, 0]
        x2 = enc_dec[ii, :, 1]
        x3 = enc_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.grid()
    ax.set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv[ii, :, 0]
        x2 = enc_adv[ii, :, 1]
        x3 = enc_adv[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='dmd')
    ax.set_title("DMD")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{dmd})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"Epoch: {epoch}/{hyp_params['max_epochs']}, Learn Rate: {hyp_params['lr']:.5f}," +
        f"Val. Loss: {val_loss[-1]:.3f}, Noise Mag: {noise_mag}", 
        size=25)

    fig.tight_layout()

    plt.savefig(save_path)
    plt.close()