

import os
import run_1M

date_run = '0916'
steps = 3
working_folder =  'datasets/cifar10/trained_nets'

multi_run_dict = {
    0: [0,  'epoch_11_sd.pt'],
    1: [0, 'epoch_12_sd.pt'],
    2: [0,  'epoch_13_sd.pt'],
}
idx = 0
for _, model_file_all in multi_run_dict.items():
    model_file = model_file_all[1]
    gpu_index = model_file_all[0]
    cmd_2d_run = run_1M.set_up_command_1M_model( gpu_index, model_file , working_folder, steps, date_run)
    print(f'----- command to run: {cmd_2d_run} /n' )
    os.system( cmd_2d_run ) 
    

###========================================================================
###========================================================================
###  Optional: Force all commands to run in parallel
###  This seems unnecessary, since all commands will always be run in GPU in parallel

def set_up_cmd_parallel():
    cmd_all_run = ''
    for _, model_file_all in multi_run_dict.items():
        model_file = model_file_all[1]
        gpu_index = model_file_all[0]
        cmd_2d_run = run_1M.set_up_command_1M_model( gpu_index, model_file , \
             working_folder, steps, date_run)
        if idx == 0:
            cmd_all_run += cmd_2d_run  
        else: 
            cmd_all_run += ' & ' + cmd_2d_run   # for parallel run 
        idx += 1

    return cmd_all_run   

# print('------------------------------------------------' )
# print('----- command to run: ', cmd_all_run )
# os.system( cmd_all_run ) 


#=======================
# cd datasets/cifar10/trained_nets