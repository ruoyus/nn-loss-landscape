

'''
This script is the one-shot script to run the interpolation code plot_surface.py for mulitiple model files.
Restriction: operate on the models with different epochs, with name epoch_k_sd.pt, where k =0,1,2,..,50 (or 100).

Typical command: (we want to replace it by something else)

rm -r -f datasets/cifar10/trained_nets/test_plot_2d  &&
CUDA_VISIBLE_DEVICES=2  nohup   python plot_surface.py --name test_plot_2d --model resnet56 --dataset cifar10
 --x=-1:1:21 --y=-1:1:21 --plot --model_file datasets/cifar10/trained_nets/epoch_45_sd.pt 
--cuda --ngpu 1 --threads 8 --batch_size 512 > resnet_2d_0908.out & 2>&1

  rm -r -f datasets/cifar10/trained_nets/test_plot_3m  &&
CUDA_VISIBLE_DEVICES=1  nohup python plot_surface.py --name test_plot_3m --model resnet56 --dataset cifar10 
--x=-0.1:1:19 --y=-0.1:1:19 --plot   --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt 
--model_file2 datasets/cifar10/trained_nets/epoch_10_sd.pt 
--model_file3 datasets/cifar10/trained_nets/epoch_45_sd.pt 
--cuda --ngpu 1 --threads 8 --batch_size 256 > 3m_0908.out & 2>&1
'''

import os
import time 
import argparse

parser = argparse.ArgumentParser(description='Experiments of "visualize the landscape of neural nets')
# parser.add_argument('--epoch_index', type=int, default = 10,  help='' )  # note this is integer 
parser.add_argument('--steps', type=int, default= 3, help='num of interpolated points')
# parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
parser.add_argument('--no_1M', action='store_true', help='run 1M models')  # add "--1M_ind" if want to exclue 1M_ind run
parser.add_argument('--no_3M', action='store_true', help='run 3M models')  
parser.add_argument('--no_move', action='store_true', help='move new images to desired folder, for future downloading')    
parser.add_argument('--clean_new_folder', '--clean', action='store_true', 
                   help='clean the folder that stores images for future downloading, before moving the new images to this folder')  
  # to shut down, add --no_move, to make it true 
parser.add_argument('--dates', default = '0915',  help='add date to the log files' ) 
args = parser.parse_args()

### Major script

print('======================================================================================================' )
print('--------------  START 1M_3M_MultiTimes.py -----------' )
print('======================================================================================================' )
start_time = time.time()

### setting up directories
# global 
script_name = ' plot_surface.py '
# global 
working_folder = 'datasets/cifar10/trained_nets'  # folder that contains saved models, and the folder of plots

print('------------------------------------------------' )
print('-------current directory is:', os.getcwd()  )
print('-------working folder is:', working_folder  )
print('------- python script is :', script_name  )
print('------------------------------------------------' )


# print( '----------STILL ALIVE # 01' )
# some standard 

# set up the name of the two varying folder/file: log_file, plot_folder
def set_up_folder_name ( model_file  ):
    """
        Produce log_file, plot_folder based on model_file (a name)
    """
    model_pure = model_file[:model_file.rfind('.pt')]
    plot_folder = 'plot_' + model_pure
    log_file = f'log_{args.dates}_' + model_pure + f'steps_{args.steps}'  # .out  This is the log file, recording the printing
    return log_file, plot_folder 

# add a new folder name to the folder list 
def add_folder_1M( model_file, server_folder_list_1M=[] ):
    _, plot_folder = set_up_folder_name (model_file )  
    server_folder_list_1M.append(plot_folder)    # collect all plot_folder names  
    return server_folder_list_1M

# set up the command based on the gpu_index and model_file 
def set_up_command_1M( gpu_index, model_file , dir_type = 'weights'):
    """
        Produce command, based on gpu_index, plot_folder
    """
    log_file, plot_folder = set_up_folder_name (model_file)
    print('----- the complete directory of plot_folder, which saves all plots:', f'{working_folder}/{plot_folder}' )
    print(f'----- the log file that saves the running record: ********* {log_file}.out  ******** ' )
    print(f'----- GPU {gpu_index} will be running this script' )
    print('----------------------------------------------------------' )

    cmd_prefix = f'CUDA_VISIBLE_DEVICES={gpu_index} nohup python '
    cmd_suffix = f' > {log_file}.out & 2>&1'

    # model_file = working_folder + 'epoch_45_sd.pt'
    # model_file2 = working_folder + 'epoch_1_sd.pt'
    # model_file3 = working_folder + 'epoch_1_sd.pt'
    part1 = f' --name {plot_folder} --model resnet56 --dataset cifar10 '
    part2 = f' --x=-1:1:{args.steps} --y=-1:1:{args.steps} --model_file {working_folder}/{model_file} '
    part3 = f' --cuda --plot --dir_type {dir_type} --batch_size 2048 '  # single GPU version   # --threads 8 

    # Step 2: define command name
    cmd_2d_run = cmd_prefix + script_name +  part1 + part2 + part3 + cmd_suffix
    return cmd_2d_run 

#--------------------------------------------------------------------------------------------
# set up the name of the two varying folder/file: log_file, plot_folder
def set_up_folder_name_3M ( model_file , model_file2, model_file3 ):
    """
        Produce log_file, plot_folder based on model_file (a name)
    """
    model_p1 = extra_ind (model_file)
    model_p2 = extra_ind (model_file2)
    model_p3 = extra_ind (model_file3)
    model_comb = f'ep_{model_p1}_{model_p2}_{model_p3}'
    plot_folder = 'plot_' + model_comb
    log_file = f'log_{args.dates}_' + model_comb + f'_steps_{args.steps}'  # .out  This is the log file, recording the printing
    
    # print('------------------------------------------------' )
    print('------ plot_folder name is:', plot_folder  )
    # print('------------------------------------------------' )

    return log_file, plot_folder 

# extra pure index from model file name 
def extra_ind (model_file):
    model_p1 = model_file[:model_file.rfind('_sd.pt')]
    print('------------------------------------------------' )
    print('input model_file name:', model_file )
    # print( 'model_p1: ' ,model_p1)
    model_pure = model_p1[model_p1.find('_'): ]
    model_pure = model_pure[1:]
    print( 'extracted key index: ' ,model_pure)
    return model_pure 

# add a new folder name to the folder list 
def add_folder_3M( model_file, model_file2, model_file3 , server_folder_list_3M=[] ):
    _, plot_folder = set_up_folder_name_3M (model_file , model_file2, model_file3)  
    server_folder_list_3M.append(plot_folder)    # collect all plot_folder names  
    return server_folder_list_3M

# set up the 3-model command based on the gpu_index and model_file 
def set_up_command_3M( gpu_index, model_file, model_file2, model_file3 , dir_type = 'states'):
    """
        Produce command, based on gpu_index, plot_folder
    """
    log_file, plot_folder = set_up_folder_name_3M (model_file , model_file2, model_file3)
    # print('---------------------------------------------------------' )
    print('----- the complete directory of plot_folder, which saves all plots:', f'{working_folder}/{plot_folder}' )
    print(f'----- the log file that saves the running record: ********* {log_file}.out  ******** ' )
    print(f'----- GPU {gpu_index} will be running this script' )
    print('----------------------------------------------------------' )
    cmd_prefix = f'CUDA_VISIBLE_DEVICES={gpu_index} nohup python '
    cmd_suffix = f' > {log_file}.out & 2>&1'

    # print( '----------STILL ALIVE # 02' )
    # model_file = working_folder + 'epoch_45_sd.pt'
    # model_file2 = working_folder + 'epoch_1_sd.pt'
    # model_file3 = working_folder + 'epoch_1_sd.pt'
    part1 = f' --name {plot_folder} --model resnet56 --dataset cifar10 --x=-0.5:1.5:{args.steps} --y=-0.5:1.5:{args.steps}'
    part2 = f' --model_file {working_folder}/{model_file} --model_file2 {working_folder}/{model_file2} --model_file3 {working_folder}/{model_file3}'
    part3 = f' --cuda --plot --dir_type {dir_type} --batch_size 2048 '  # single GPU version   # --threads 8 

    ## Start running 
    # Step 1: remove the folders with the plots
    # Step 2: define command name
    cmd_run =  cmd_prefix + script_name +  part1 + part2 + part3 + cmd_suffix

    return cmd_run

# print( '----------STILL ALIVE # 05---', cmd_2d_run )

#=================================================================
#===      1 model  2 dim 
#=================================================================
cmd_1M_run = ''
multi_run_dict = {
    0: [ 1, 'epoch_15_sd.pt' ], 
    1: [ 1, 'epoch_10_sd.pt' ], 
    2: [ 1, 'epoch_5_sd.pt' ],
    3: [ 1, 'epoch_1_sd.pt' ],
 #   3: [ 2, 'epoch_20_sd.pt' ],
 #   4: [ 2, 'epoch_20_sd.pt' ],
 #   5: [ 2, 'epoch_20_sd.pt' ],
}

idx = 0
server_folder_list_1M = []
for something , model_file_all in multi_run_dict.items():
    model_file = model_file_all[1]
    gpu_index = model_file_all[0]
    server_folder_list_1M = add_folder_1M(model_file, server_folder_list_1M)
    cmd_run = set_up_command_1M( gpu_index, model_file )
    if idx == 0:
        cmd_1M_run += cmd_run  
    else: 
        cmd_1M_run += ' & ' + cmd_run   # for parallel run 
    idx += 1

    ## Start running 
# Step 1: remove the folders with the plots
str2 = f' {working_folder}/'.join(server_folder_list_1M)
all_folder = f'{working_folder}/{str2}'
cmd_rm_img_folder =  f'rm -r -f {all_folder}'  #  datasets/cifar10/trained_nets/test_plot_3m ... ... (3 files)

cmd_1M_run = f'{cmd_rm_img_folder} && {cmd_1M_run}'  # combine: remove folders; run all scripts for 1M

print('======================================================================================================' )
print(f'------------  Finish setting up the command for 1M ---------' )

no_1M = args.no_1M 
# no_1M = 1     # now shut it down for sure 
if not no_1M:   # run_1M_ind:
    os.system('wait')  # wait until previous jobs are done

    print('======================================================================================================' )
    print(f'------------  Start running the main script for {idx} cases in {idx} GPUs in parallel ---------' )
    print(f'-----Note: this comman consists of the concatenation of {idx} commands, connected by &---------')
    print('----------------------------------------------------------------------' )
    print('----- command to run 1 model landscape: ', cmd_1M_run )
    print('-------------------------------------------------------------------' )
    os.system( cmd_1M_run ) 


#=========================================================================================================
#===      3 model  2 dim 
#=========================================================================================================
cmd_3M_run = ''
multi_run_dict = {
    0: [ 0, 'epoch_190_sd.pt', 'epoch_20_sd.pt', 'epoch_1_sd.pt' ], 
    1: [ 0, 'epoch_190_sd.pt', 'epoch_10_sd.pt', 'epoch_1_sd.pt' ], 
    2: [ 0, 'epoch_190_sd.pt', 'epoch_5_sd.pt', 'epoch_1_sd.pt' ]
}


idx = 0
server_folder_list_3M = []

# os.system( cmd_rm_img_folder  ) 
for something, model_file_all in multi_run_dict.items():
    model_file = model_file_all[1]
    model_file2 = model_file_all[2]
    model_file3 = model_file_all[3]
    gpu_index = model_file_all[0]  # gpu index 
    server_folder_list_3M = add_folder_3M(model_file, model_file2, model_file3,server_folder_list_3M)
    cmd_run  = set_up_command_3M( gpu_index, model_file, model_file2, model_file3 )
    if idx == 0:
        cmd_3M_run += cmd_run  
    else: 
        cmd_3M_run += ' & ' + cmd_run   # for parallel run 
    idx += 1

str2 = f' {working_folder}/'.join(server_folder_list_3M)
all_folder = f'{working_folder}/{str2}'
cmd_rm_img_folder =  f'rm -r -f {all_folder}'  #  datasets/cifar10/trained_nets/test_plot_3m

cmd_3M_run = f'{cmd_rm_img_folder} && {cmd_3M_run}'   

print('======================================================================================================' )
print(f'------------  Finish setting up the command for 3M ---------' )

# run_3M_ind = 1
if not args.no_3M: 
    os.system('wait')  # wait until previous jobs are done
    print('======================================================================================================' )
    print(f'------------  Start running the main script for {idx} cases in {idx} GPUs in parallel ---------' )
    print(f'-----Note: this comman consists of the concatenation of {idx} commands, connected by &---------')
    print('----------------------------------------------------------------------' )
    print('----- command to run 3 models fitting: ', cmd_3M_run )
    print('-------------------------------------------------------------------' )
    os.system( cmd_3M_run ) 

#============================================================================================================
#  automatically sync the server side figures to local folders
# move_images = 1
if not args.no_move: 
    os.system('wait')  # wait until previous jobs are done
    folder_to_download = f'{working_folder}/new_plots_to_download'
    os.system(f'mkdir -p {folder_to_download}')
    
    print('======================================================================================================' )
    os.system(f"echo ----start moving images to {folder_to_download} ----")
    
    if args.clean_new_folder:  # delete all existing files in the folder, before moving new images into the folder. 
        print(f'---- remove all files in the folder {folder_to_download}')
        os.system(f'rm -rf {folder_to_download}/*')  

    server_folder_list = server_folder_list_1M + server_folder_list_3M
    print(f'------------------ the folder lists: {server_folder_list} ----------------')
    for  server_saved_folder in server_folder_list:     # ['plot_3m_100_1_20', 'plot_3m_100_1_20_weight', 'plot_3m_100_50_20' ]: 
        server_image_dir = f'{working_folder}/{server_saved_folder}'
        print(f'---- folder to be moved: {server_image_dir}')
        copy_images = "scp -r " + server_image_dir + ' ' + folder_to_download  # local_dir_image   
        os.system(copy_images)
        os.system("echo ----moving images succeed----")


final_time = time.time()
print('-------Total time: final_time - start_time')

#============================================================================================================
# cd datasets/cifar10/trained_nets



