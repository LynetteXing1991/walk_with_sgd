# PLOTTING HESSIAN Spectral norm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Experiments for plotting hessian spectral norm')
parser.add_argument('--files', nargs='+', help='list of pickle files containing spectral norm', required=False)
args = parser.parse_args()
spectral_norm_list = []
spectral_norm_epochs=[]
for file in args.files:
    spectral_norm_list.append ( pickle.load(open(file, 'rb')))
    epoch = int(file.split('_')[-1].split('.')[0])
    spectral_norm_epochs.append(epoch)

fig, ax = plt.subplots(1, figsize=(4.5, 2))

ax.plot(spectral_norm_epochs, spectral_norm_list, 'b', alpha=0.7)
ax.set_ylabel('$\lambda_{\max}(\mathbf{H})$', color='b')
ax.set_xlabel('Epoch', fontsize=9.5)
ax.legend(loc='best', prop={'size': 7})
ax.grid(which='both')
ax.grid(which='minor', alpha=0.9)
ax.grid(which='major', alpha=0.9)
plt.tight_layout()
plt.show()
