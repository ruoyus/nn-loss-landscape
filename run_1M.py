

'''
This script is the one-shot script to run the interpolation code plot_surface.py for mulitiple model files.
Restriction: operate on the models with different epochs, with name epoch_k_sd.pt, where k =0,1,2,..,50 (or 100).

Typical command: (we want to replace it by something else)

rm -r -f datasets/cifar10/trained_nets/test_plot_2d  &&
CUDA_VISIBLE_DEVICES=2  nohup   python plot_surface.py --name test_plot_2d --model resnet56 --dataset cifar10
 --x=-1:1:21 --y=-1:1:21 --plot --model_file datasets/cifar10/trained_nets/epoch_45_sd.pt 
--cuda --ngpu 1 --threads 8 --batch_size 512 > resnet_2d_0908.out & 2>&1

'''

import os
import argparse

parser = argparse.ArgumentParser(description='Experiments of "visualize the landscape of neural nets')
parser.add_argument('--epoch_index', '--ep', type=int, default = 10,  help='' )  # model file, with name epoch_2_sd.pt
parser.add_argument('--model_file', '--mod', default = '', help='' )  # general model file 
parser.add_argument('--steps', '--s', type=int, default= 4, help='num of interpolated points')
parser.add_argument('--gpu_index', '--g', default= '', help='gpu index')
parser.add_argument('--date_run', '--d', default= '0916', help='date to run')
parser.add_argument('--working_folder', '--folder', default= 'datasets/cifar10/trained_nets', 
                     help='folder that contains saved models, and the folder of plots')
args = parser.parse_args()

# set up the name of the two varying folder/file: log_file, plot_folder
def set_up_folder_name_1M ( model_file , date_run ):
    """
    Produce log_file, plot_folder based on model_file (file name ending with _sd.pt)
    If model_file is epoch_1_sd.pt, date_run is 0913: plot_folder 'plot_2d_epoch_1', log file 'log_0913_2d_epoch_1'
    """
    # if model_file == '': model_pure = f'ep{ep_ind}'
    model_pure = model_file[:model_file.rfind('_sd.pt')]
    plot_folder = f'plot_2d_{model_pure}'
    log_file = f'log_{date_run}_2d_{model_pure}'  # .out  This is the log file, recording the printing
    return log_file, plot_folder 

def set_up_command_1M_epoch( gpu_index, ep_ind , working_folder, steps, date_run):
    """
        Produce command, based on epoch_index. Assume model_file is in the form of epoch_2_sd.pt
    """
    model_file = f'epoch_{ep_ind}_sd.pt'   # Can change according to your file name
    cmd_2d_run = set_up_command_1M_model( gpu_index, working_folder, model_file , steps, date_run)
    return cmd_2d_run 

def set_up_command_1M_model( gpu_index,   model_file , working_folder, steps, date_run):
    """
        Produce command, based on model_file
    """
    script_name = 'plot_surface.py'
    if gpu_index == '':  # if no value, do not specify any GPU 
        cmd_prefix = f'nohup python {script_name}'
    else:
        cmd_prefix = f'CUDA_VISIBLE_DEVICES={gpu_index} nohup python {script_name}'
    log_file, plot_folder = set_up_folder_name_1M (model_file, date_run  )
    cmd_suffix = f' > {log_file}.out & 2>&1'

    print( '----------STILL ALIVE # 02' )
    part1 = f' --name {plot_folder} '
    part2 = f' --x=-1:1:{steps} --y=-1:1:{steps} --plot --model_file {working_folder}/{model_file} '
    part3 = '--dir_type weights'
    part4 = ' --model resnet56 --dataset cifar10 --cuda  --batch_size 2048 '    # --ngpu 1 --threads 8 

    # Step 1: remove the folders with the plots
    cmd_rm_img_folder =  f'rm -r -f {working_folder}/{plot_folder}'  #  datasets/cifar10/trained_nets/test_plot_3m
    cmd_2d_run =cmd_rm_img_folder + ' && ' + cmd_prefix + part1 + part2 + part3 + part4 + cmd_suffix

    return cmd_2d_run 

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    ### Major script

    print('------------------------------------------------' )
    print('--------------  START run_1M.py -----------' )
    print('-------current directory is:', os.getcwd()  )
    print('------------------------------------------------' )

    ### setting up directories
    if args.model_file =='':
        print(f'use epoch index {args.epoch_index}; process file epoch_{args.epoch_index}_sd.pt')
        cmd_2d_run = set_up_command_1M_epoch( args.gpu_index, args.epoch_index ,  args.working_folder, args.steps,  args.date_run )
    else:
        print(f'process model file {args.model_file}')
        cmd_2d_run = set_up_command_1M_model( args.gpu_index, args.model_file,  args.working_folder, args.steps,  args.date_run )

    print('effective command: ' , cmd_2d_run )
    os.system( cmd_2d_run ) 


#=======================
# cd datasets/cifar10/trained_nets


