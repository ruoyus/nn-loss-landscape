"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import data_loader
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler

import sys
sys.path.insert(0, './datasets/cifar10/trained_nets')  # see https://github.com/pytorch/pytorch/issues/3678
import mpi4pytorch as mpi  

def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print("%s is already set up" % surf_file)
            return
        f.close()

    f = h5py.File(surf_file, 'a')
    # f['dir_file'] = dir_file # This does not work and does not seem to be used again

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel using MPI reduce.
        Input (major): 
             net, data,
             rank: index of the machine in the distributed system 
        Output: the loss values and save in surf_file
        Dependency: scheduler.get_job_indices, net_plotter.set_weights, evaluation.eval_loss
    """
    print( '----------STILL ALIVE # 01' )
    print( '-----------------------surf_file path'  , surf_file )
    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')  # check https://docs.h5py.org/en/stable/quick.html for h5py files
    print( '----------STILL ALIVE # 02' )
    print('-----------------------------------------------------------------' ) 
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    # RS: Utilize distributed computation, since inds are split into multiple machines comm; 
    # Key idea: the main job is to evaluate the losses of multiple models, which can be done in parallel 
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d' % (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    # Loop over all uncalculated loss values. inds is defined a few lines above
    # RS: I suspect that the for loop is not sequential but parallel; when enumerating inds, automatically split into multiple GPUs. 
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        # RS: e.g. if coord = 0.5 in 1d case, then we obtain net = 0.5 (model_1 + model_2). 
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Compute the loss value, and record the time
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        # RS: I'm confused: why syncing during the for loop? I thought the process is: 
        # every GPU computes its own data points (e.g. 10 on GPU-1 + 8 on GPU-0), then they combine. But this code says they sync at each loop, why?
        # For 1-GPU, this part probably does not matter (not sure whether 8 threads play a role in distributed computation)
        syc_start = time.time()
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        # print('----sync time of this part:', syc_time )  # print the time to sync
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        # print syc_time to check 
        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
            rank, count, len(inds), 100.0 * count / len(inds), str(coord), loss_key, loss,
            acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()


###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--name', default = 'test_plot', help='name of the plot run; is used as a folder for the files produced')

    # computation
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    parser.add_argument('--use_testset', default=False, help='use the test set for computing the landscape')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='',
                        help='specify the name of direction file, or the path to an existing direction file')
    parser.add_argument('--dir_type', default='states',   # 'weights',
                        help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:xmax:xnum')
    parser.add_argument('--y', default='', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False,
                        help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='',
                        help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    args = parser.parse_args()

    ### Set the random seed manually for reproducibility.
    use_cuda = torch.cuda.is_available()
    np.random.seed(args.seed)
    # random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:  # you have GPU
        if not args.cuda:    # but you do not choose to run with GPU
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:  # you have GPU and run with GPU; good 
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True

    prep_start = time.time()

    print('-----------------------------------------------------------------' )  
    print('----------   Start running  plot_surfacy.py ---------------------')
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))  # print current time 
    print('-----------------------------------------------------------------' )  
    # --------------------------------------------------------------------------
    # Environment setup
    # --------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1
    print('-----------------------------------------------------------------' ) 
    print('---args.mpi', args.mpi, '---distributed basic: rank =', rank, '---number of processors =', nproc )   
    print('-----------------------------------------------------------------' )  

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    # --------------------------------------------------------------------------
    # Check plotting resolution
    # --------------------------------------------------------------------------

    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    # --------------------------------------------------------------------------
    # Load models and extract parameters
    # --------------------------------------------------------------------------
    net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net)  # initial parameters
    s = copy.deepcopy(net.state_dict())  # deepcopy since state_dict are references
    if args.ngpu > 1:  # RS: not sure how ngpu > 1 makes a difference; often set ngpu = 1, but -n 3
        # data parallel with multiple GPUs on a single node
        if args.ngpu > torch.cuda.device_count():
            raise ValueError(
                f'Please do not enter a value higher than the available gpus ({torch.cuda.device_count()}).')
        net = nn.DataParallel(net, device_ids=range(args.ngpu))

    # --------------------------------------------------------------------------
    # Save the parameters to a json file
    # --------------------------------------------------------------------------
    model_dir = os.path.dirname(os.path.abspath(args.model_file))  # the directory the model is in
    print(f'------ working folder: {model_dir} ')
    plot_dir = f'{model_dir}/{args.name}'  # the directory where all the output of this run is saved
    print(f'------ a sub folder that stores the plots: {plot_dir} ')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)  # create the directory
    with open(f'{plot_dir}/parameters.json', 'w') as outfile:
        json.dump(args.__dict__, outfile)

    # --------------------------------------------------------------------------
    # Setup the direction file and the surface file
    # 根据输入args和当前的模型 net, 计算dir, 并存入 dir_file
    # 核心函数: net_plotter.setup_direction
    # --------------------------------------------------------------------------
    dir_file = f"{plot_dir}/direction.h5" # name the direction file
    # equivalent but unnessary:  net_plotter.name_direction_file(args, plot_dir )  

    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)     

    surf_file = f'{plot_dir}/surf_file.h5'  # typical name: datasets/cifar10/models/
    # unnessary and equivalent: surf_file = net_plotter.name_surface_file(args, plot_file)
    print('surf_file name: ', surf_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)
        print(f'just set up surface file with name {surf_file}')

    # wait until master has setup the direction file and surface file
    if args.mpi:
        mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the cosine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    # --------------------------------------------------------------------------
    # Setup data_loader
    # --------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root='datasets/' + args.dataset + '/data', train=True, download=True)

    if args.mpi:   
        mpi.barrier(comm)

    trainloader, testloader = data_loader.load_dataset(args.dataset, args.datapath,
                                                       args.batch_size, args.threads, args.raw_data,
                                                       args.data_split, args.split_idx,
                                                       args.trainloader, args.testloader)
    prep_time = time.time() - prep_start                                                   
    print('----preparation time (setting up environment, loading models, loading data):', prep_time ) 

    # --------------------------------------------------------------------------
    # Start the computation
    # --------------------------------------------------------------------------
    print('---------------------------------------------------------------------------------') 
    print('------------------------ computation starts ---------------------------------')
    dataset = 'train'
    loader = trainloader
    if args.use_testset:
        dataset = 'test'
        loader = testloader

    crunch(surf_file, net, w, s, d, loader, f'{dataset}_loss', f'{dataset}_acc', comm, rank, args)

    print('------------- computation finishes ----------------------')
    print('=================================================================================')
    # --------------------------------------------------------------------------
    # Plot figures
    # --------------------------------------------------------------------------
    if args.plot and rank == 0:
        start_time = time.time()
        if args.y and args.proj_file:  # if args contains y and args contains proj_file. Often for 2d plots, 3 models. 
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, f'{dataset}_loss', args.show)
            print('----------------------------\n ----plot 2d countour and trajectory ------\n----------------------')
        elif args.y:  # if args contains y, while NOT contains proj_file. Often for 2d plot, 1 model
            plot_2D.plot_2d_contour(surf_file, f'{dataset}_loss', args.vmin, args.vmax, args.vlevel, args.show)
            print('----------------------------\n ----plot 2d countour ------\n----------------------')
        else:   # if args does NOT contains y; must be 1d plot, 2 models. 
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
            print('----------------------------\n ----plot 1d interpolation ------\n----------------------')
        
        plot_time = time.time() - start_time
        print('Plotting time:', plot_time )    

    # save all args in a txt file in the same folder 
    print_all_args = 1
    if print_all_args:
        print('------ store all values of args into ALL_args.txt')
        dict_args = parser.parse_known_args()[0].__dict__
        for key in dict_args:
            print('value for %s is: %s'% (key, dict_args[key]))
        file = open(f'{plot_dir}/ALL_args.txt', 'w') 
        # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
        for k,v in dict_args.items():
            file.write(str(k)+'   '+str(v)+'\n')
        # 注意关闭文件
        file.close()