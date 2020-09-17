"""
    2D plotting funtions
"""
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns

np.set_printoptions(threshold=np.inf)  # print all entries of a matrix 

# truncate an array by: replacing nan by max; truncate anything larger than sth to max
def truncate_array(Z, des_height = np.inf):
    Z_true_min = np.nanmin(Z)
    Z_true_max = np.nanmax(Z)
    Z[np.isnan(Z)]= Z_true_max
    print(f'old Z: min value is {Z_true_min}, max value is {Z_true_max}, desired height is {des_height}')
    np.clip(Z, Z_true_min, Z_true_min + des_height , Z )
    print(f'new Z: min value is {np.min(Z)}, max value is {np.max(Z)}, desired height is {des_height}')

# main plot function
def plot_2d_contour(surf_file, surf_name='train_loss',  vmin=0.1, vmax=10, vlevel=0.5, show=False, des_height = np.inf ):
    """Plot 2D contour map and 3D surface.
       vmin and vmax: the minimal and maximal allowable value to show. 
       vlevel: gap in the countour (the smaller, the denser the plot)
    """
    print('-------current directory is:', os.getcwd()  )
    print( 'surf_file is saved as:', surf_file)
    print('-----------------------' )
    
    f = h5py.File(surf_file, 'r')

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err':
        Z = 100 - np.array(f[surf_name][:])
    else:
        print('%s is not found in %s' % (surf_name, surf_file))

    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])

    mod_len  = -1  # default: 0 or -1; if want to draw a subset, change this quantity to desired length. 
    if mod_len > 0:
        x = x[5: 5 + mod_len - 1 ]
        y = y[5: 5 + mod_len - 1 ]    
        print(Z.shape)
        Z = Z[5: 5 + mod_len - 1 , 5: 5 + mod_len - 1 ]
        print(Z.shape)

    X, Y = np.meshgrid(x, y)
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    Z_orig = Z # Z[35: 40, 0:30]
    # print(Z_orig[0:3, 0:3])
    print('Original Z values: max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z_orig ), surf_name, np.min(Z_orig )))
    truncate_array(Z, des_height ) 
    print('After truncation, max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z ), surf_name, np.min(Z )))
    print(Z[0:3,0:3])

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    plot_dir = os.path.dirname(os.path.abspath(surf_file))

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(plot_dir + '/' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(plot_dir + '/' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(plot_dir + '/' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(plot_dir + '/' + surf_name + '_2dheat.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,  linewidth=0, antialiased=False)  # cmap='viridis',
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=20,azim= -25)  # change the view. elev: rotate around y-axis; azim: rotate around z-axis
    ax.set_xlabel( 'x (model 1 to 2)' )
    ax.set_ylabel('y (model 1 to 3)')
    ax.set_zlabel('surf_name')
    fig.savefig(plot_dir + '/' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()
    if show: plt.show()


def plot_trajectory(proj_file, dir_file, show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    assert exists(proj_file), 'Projection file does not exist.'
    f = h5py.File(proj_file, 'r')
    fig = plt.figure()
    plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')
    f.close()

    if exists(dir_file):
        f2 = h5py.File(dir_file, 'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x * 100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y * 100), fontsize='xx-large')
        f2.close()

    fig.savefig(proj_file + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    if show: plt.show()


def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])

    fig = plt.figure()
    CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

    # plot trajectories
    pf = h5py.File(proj_file, 'r')
    plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file, 'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x * 100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y * 100), fontsize='xx-large')
    df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pf.close()
    if show: plt.show()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig', show=False):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """

    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])

    # Plot 2D heatmaps with color bar using seaborn
    abs_ratio = np.absolute(np.divide(Z1, Z2))
    print(abs_ratio)

    fig = plt.figure()
    sns_plot = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=.5, cbar=True,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_abs_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # Plot 2D heatmaps with color bar using seaborn
    ratio = np.divide(Z1, Z2)
    print(ratio)
    fig = plt.figure()
    sns_plot = sns.heatmap(ratio, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')
    f.close()
    if show: plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--des_height', '--des_ht', default=np.inf, type=float, help='max allowable height of 3d plot')

    args = parser.parse_args()

    if exists(args.surf_file) and exists(args.proj_file) and exists(args.dir_file):
        plot_contour_trajectory(args.surf_file, args.dir_file, args.proj_file,
                                args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
    elif exists(args.proj_file) and exists(args.dir_file):
        plot_trajectory(args.proj_file, args.dir_file, args.show)
    elif exists(args.surf_file):
        print(f"----surf_file {args.surf_file} exists, will continue to plot using function plot_2d_counter------")
        plot_2d_contour(args.surf_file, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show, args.des_height)
    else: # surf_file does not exist
        print("------ERROR: Desired surf file does NOT exist! Please check the file directory! ---")

