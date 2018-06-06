import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle as pkl
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Experiments for plotting sgd trajectory')
parser.add_argument('--root_dir', type=str, help='', required=True)
parser.add_argument('--epoch_index', type=int, help='', required=True)
args = parser.parse_args()
plot_list = ['interp','angle','dist']


for plot in plot_list:
    if plot=='interp':
        save_dir = args.root_dir + '/interpolation_'+str(args.epoch_index)+'/train_loss.pkl'
    elif plot=='angle':
        save_dir = args.root_dir + '/cos_mg_list.pkl'
    elif plot == 'dist':
        save_dir = args.root_dir + '/param_dist_list.pkl'

    vec = pkl.load(open(save_dir, 'rb'))
    fig, ax = plt.subplots(1, figsize=(4.5,2))
    if plot=='interp':
        ax.plot(np.arange(0, 50, 0.1), vec[0:500])
    else:
        ax.plot(np.arange(0, 50, 1), vec[0:50])
    ax.set_xlabel('Iteration', fontsize=9.5)
    if plot=='interp':
        ax.set_ylabel('Training Loss', fontsize=9)
    elif plot=='angle':
        ax.set_ylabel('cos($g_{t-1}, g_{t}$)', fontsize=9)
    elif plot=='dist':
        ax.set_ylabel('Parameter Distance', fontsize=9)
    spacing=1
    minorLocator = MultipleLocator(spacing)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    # Set grid to use minor tick locations.
    ax.grid(which='minor')
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.2)
    plt.tight_layout()
    plt.show()