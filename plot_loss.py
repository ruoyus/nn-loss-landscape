'''
   usage:  $ python plot_loss.py --surf_file train_loss.pkl 
   Other choices: python plot_loss.py --surf_file train_acc.pkl --y_max 100
                  python plot_loss.py --surf_file valid_acc.pkl --y_max 100
   optional: --folder datasets/cifar10/trained_nets --show 
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle as pkl
import numpy as np
import os
import time 
import argparse

# save_dir = f'{args.working_folder}/train_loss.pkl'
# plot 1d curve, from data in .pkl file 
def plot_1d_loss_pkl(surf_file, working_folder, y_max=5,  max_len = 1500, show=False):
    save_dir = f'{working_folder}/{surf_file}'  # e.g. working_folder/train_loss.pkl
    print(f' target file for plotting: {save_dir}')
    if not os.path.exists(save_dir):
        print(f'-------- {save_dir} does not exist -------')
    vec = pkl.load(open(save_dir, 'rb'))  # original code 
    _, ax = plt.subplots(1, figsize=(4.5,2))

    plot_len = len(vec[0: max_len ])   # check at most max_len points in the file 
    ax.plot(np.arange(0, plot_len , 1), vec[0: max_len ] , 'b-', label='Training loss', linewidth=1)

    print( 'length of saved file at', save_dir, ':', len(vec ) )
    print('length of loss vector: ', plot_len)
        
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Training Loss', fontsize=9)
    ax.set_ylim(0, y_max)  # set the upper limit of the plotted loss 

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
    # plt.show()

    save_fig_dir = f'{args.working_folder}/training_plots/'
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)
        print('created new folder with name', save_fig_dir  )
    fig_name = f'{save_fig_dir}{surf_file}.pdf'
    plt.savefig(fig_name,dpi=300, bbox_inches='tight', format='pdf')   # (   'figure_%s.png' % (plt_index ))
    if show: plt.show()
    plt.close()

    # save all values in a txt file in the same folder 
    print(f'------ store all loss values into {surf_file}.txt')
    file = open(f'{save_fig_dir}/{surf_file}.txt', 'w') 
    ep_id = 1
    for k in vec:
        file.write(f'epoch {ep_id} loss:   {k}' + '\n') # + str(k)
        ep_id += 1
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for plotting loss during training')
    parser.add_argument('--surf_file', '--f', default='train_loss.pkl', help='The pkl file contains loss values')
    parser.add_argument('--working_folder', '--folder', default= 'datasets/cifar10/trained_nets')
    parser.add_argument('--y_max', default=5, type=float, help='ymax value')
    parser.add_argument('--check_len', default=500, type=float, help='length of plot')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    args = parser.parse_args()

    start_time = time.time()      # starting timing 
    print('starting plot')

    plot_1d_loss_pkl(args.surf_file, args.working_folder, args.y_max, args.check_len,  args.show  )
    
    end_time = time.time()
    process_time = end_time - start_time
    print("Total process time = %f" % process_time)