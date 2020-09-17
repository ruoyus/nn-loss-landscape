
import os
import argparse

parser = argparse.ArgumentParser(description='Experiments of "A Walk with SGD"')
parser.add_argument('--epoch_index', type=int, default = 4,  help='' )  # note this is integer 
args = parser.parse_args()

# define two key directories 
local_parent_dir = "/Users/ruoyusun/Desktop/SunDirac_Mac/科研/Coding/CODE_landscape/"
server_parent_dir  = 'ruoyus@sun.csl.illinois.edu:landscape2020/'
folder_name = 'nn-loss-landscape'  #  the major folder name, in both local computer and the

copy_all = "scp -r " + local_parent_dir + folder_name  + server_parent_dir  
upload_folder_ind = 0
if upload_folder_ind == 1:
    os.system(copy_all)
    os.system("echo ----upload whole folder succeed----")

# os.system("python ./1.py")
# copy preparation files

server_dir = server_parent_dir + folder_name   # the directory of the server folder 
local_dir = local_parent_dir + folder_name 
copy0 = "scp models_sgd_walk.py " + server_dir
copy1 = "scp sgd.py " + server_dir 
# copy original interpolation 
copy2 = "scp plot_surface.py " + server_dir 
copy3 = "scp net_plotter.py " + server_dir 
copy4a = "scp plot_1D.py " + server_dir 
copy4b = "scp plot_2D.py " + server_dir 
copy4c = "scp plot_loss.py " + server_dir 
copy5 = "scp run_1M.py " + server_dir 
copy5b = "scp run_1M_repeat.py " + server_dir 
copy6 = "scp run_1M_3M_MultiTimes.py " + server_dir 
# Another way: copy1 = "scp plot_surface.py ruoyus@sun.csl.illinois.edu:landscape2020/2020Aug_walk_with_sgd"    

upload_ind = 1
if upload_ind == 1:
    os.system("echo ----start uploading code----")
  #  os.system(copy0)
  #  os.system(copy1)
    os.system(copy2)
    os.system(copy3)
    os.system(copy4a)
    os.system(copy4b)
    os.system(copy4c)
    os.system(copy5)
    os.system(copy5b)
    os.system(copy6)
  #  os.system(copy7)
    os.system("echo ----upload code succeed----")

download_ind = 0
if download_ind == 1:
    # changing_index = str(args.epoch_index)
    download_images = 1
    if download_images: 
        # for epoch_ind in [10, 15]:  # range(25):
        os.system("echo ----start downloading image ----")
        local_dir_image = local_dir + '/datasets/cifar10/trained_nets'  # 
        for  server_saved_dir in ['new_plots_to_download']:  #  ['plot_3m_100_1_20', 'plot_3m_100_1_20_weight', 'plot_3m_100_50_20' ]: 
       # ['plot_2d_small']:  # ['test_plot_2d_seed111', 'test_plot_3m', 'plot_3m_45_1_15', 'plot_3m_45_1_20']:
            server_image_dir = server_dir + "/datasets/cifar10/trained_nets/" + server_saved_dir
            copy_back = "scp -r " + server_image_dir + ' ' + local_dir_image   
            os.system(copy_back)
            os.system("echo ----download image succeed----")

    download_model = 0
    if download_model:
        for epoch_ind in [1,2]:  # range(25):
            local_dir_models = local_dir + '/{folder_name}/datasets/cifar10/trained_nets'
            changing_index = str( epoch_ind )        
            server_dir = "ruoyus@sun.csl.illinois.edu:/home/ruoyus/landscape2020/2020Aug_walk_with_sgd/ResNet_new_2ep/epoch_" \
                 + changing_index + '_sd.pt'
            copy_back = "scp -r " + server_dir + ' ' + local_dir_models 
            os.system(copy_back)
            os.system("echo copy succeed")

# move generated models to desired folders
move_model_ind = 0
model_save_dir = server_parent_dir + '2020Aug_walk_with_sgd/ResNet_new_50ep/'
if move_model_ind:
    os.system("echo ----start moving models----")
    # os.system(move_model_code)
    os.system("echo ----upload code succeed----")

# ls datasets/cifar10/trained_nets/test_plot_1