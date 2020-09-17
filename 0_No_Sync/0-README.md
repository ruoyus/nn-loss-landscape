# Visualizing the Loss Landscape of Neural Nets

This repository contains the PyTorch code for the paper 

> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.    Github repo:  https://github.com/tomgoldstein/loss-landscape

我们的目标：1) 复现该文章的实验. 

  2) 看Landscape在非随机方向 (两个model连线) 是否有神奇的凸性; 本文和单调path的文章是否有冲突. 

  3) 为GAN 的visualization打下基础. 

 时间: 2020年9月8日Tu, 2pm-09/16. 

## 1. Setup

**Environment**: One or more multi-GPU node(s) with the following software/libraries installed:

- [PyTorch 0.4](https://pytorch.org/)
- [openmpi 3.1.2](https://www.open-mpi.org/) 用 conda install mpi安装
- [mpi4py 2.0.0](https://mpi4py.readthedocs.io/en/stable/install.html)   同上. 
- [h5py 2.7.0](http://docs.h5py.org/en/stable/build.html#install) 在作者最早的版本,有bug; 见[此帖](https://github.com/tomgoldstein/loss-landscape/issues/4); 但后来解决了这个bug
- [numpy 1.15.1](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
- [scipy 0.19](https://www.scipy.org/install.html)

**Pre-trained models**:
The code accepts pre-trained PyTorch models for the CIFAR-10 dataset.
To load the pre-trained model correctly, the model file should contain `state_dict`, which is saved from the `state_dict()` method.
The default path for pre-trained networks is `cifar10/trained_nets`.
Some of the pre-trained models and plotted figures can be downloaded here:

- [VGG-9](https://drive.google.com/open?id=1jikD79HGbp6mN1qSGojsXOZEM5VAq3tH) (349 MB)
- [ResNet-56](https://drive.google.com/a/cs.umd.edu/file/d/12oxkvfaKcPyyHiOevVNTBzaQ1zAFlNPX/view?usp=sharing) (10 MB)
- [ResNet-56-noshort](https://drive.google.com/a/cs.umd.edu/file/d/1eUvYy3HaiCVHTzi3MHEZGgrGOPACLMkR/view?usp=sharing) (20 MB)
- [DenseNet-121](https://drive.google.com/a/cs.umd.edu/file/d/1oU0nDFv9CceYM4uW6RcOULYS-rnWxdVl/view?usp=sharing) (75 MB)

**Data preprocessing**:
The data pre-processing method used for visualization should be consistent with the one used for model training. No data augmentation (random cropping or horizontal flipping) is used in calculating the loss values.   TBD: 当前的code sgd.py, 生成的模型用到了data augmentation; 需要后期修改sgd.py. 

---

### 2 生成 1D linear interpolations (on a server with single GPU)

这一步是关键，通过测试一个简单的情况(1d curve)，把code跑通; 不追求是否生成了一个很好的plots.   这一步做完，可以保证脚本和设置基本正确. 简要步骤小结:

1. 准备两个模型(同样的网络架构，不同的参数configuration), 并放到合适的文件夹(见下文). 方法是把一次正常的训练之生成的中间网络参数存储下来. 可以考虑MLP for MNIST, 或者ResNet for CIFAR10. 本文考虑后者.

2. (推荐) 单机测试 CIFAR10, using ResNet56.  (optional), 单机测试 MNIST.

### 2.1 详细步骤

##### 第一步: 生成Models.

如果datasets/cifar10/trained_nets/ 文件夹中, **epoch_1_sd.pt**,  **epoch_2_sd.pt** 等文件存在,**可以忽略这一步, 直接进入第二步**. 

如果没有训练过的模型, 可以快速生成几个模型. 方式是: 正常的训练神经网络, run: 

```
CUDA_VISIBLE_DEVICES=0 python sgd.py --save_dir datasets/cifar10/trained_nets --epochs 2
```

  --epochs 2: 总训练步数; 默认50epochs. 为了快速测试, 可选 2 epochs. 

  --save_dir datasets/cifar10/trained_nets: 生成的模型放在该文件夹. 这个地址可以修改. Whatever directory, 下一步的--model_file需要用到这个地址里面存放的 .pt 文件. 所以--save_dir如果修改的话, 下一步也要跟着修改. 

  **生成结果**: 默认放在datasets/cifar10/trained_nets/文件夹; 如果epoch=2, 那么生成 datasets/cifar10/trained_nets/**epoch_1_sd.pt** 和datasets/cifar10/trained_nets/**epoch_2_sd.pt**. 

**Remark**: 如果要对其他数据集画图，那么在datasets/ 下面建立另外的文件夹储存datasets和相应的models. 

 远程版本 (约4 hr):

```
CUDA_VISIBLE_DEVICES=1 nohup python sgd.py --save_dir datasets/cifar10/trained_nets --epochs 200 > train_net_0909.out & 2>&1
```

 **Remark**: 命名中的_sd是为了提醒我们: 储存神经网络参数必须用 state_dict(), 且对应的key应为 'state_dict'. 如果有另外的训练并保存模型的代码，要把相应的储存模型的部分改为如下的代码: (see sgd.py for the context)

```
with open( save_dir, 'wb') as f:
    torch.save( { 'state_dict': model.state_dict() } , f)  
```

##### 第二步: interpolation and plot: 在两个模型的线性连接线上生成K个点, 计算Loss, 画图并存储. 

The 1D linear interpolation method [1] evaluates the loss values along the direction between two models (e.g.. minimizers of the same network loss function). Produced using the `plot_surface.py` method.

   如何快速的跑一个版本?  **控制grid的精度** 把--x= -1:1:51 改成更小的数 0:1:5，使用较大的batchsize 2048.  

   假设我们有1个GPU, 命令如下: (77 sec)

 ```
rm -r -f datasets/cifar10/trained_nets/test_plot_1d  &&
CUDA_VISIBLE_DEVICES=1 python plot_surface.py --name test_plot_1d --model resnet56 --dataset cifar10  --x=0:1:5 --plot --dir_type states --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --batch_size 2048
 ```

 --x=0:1:5 sets the range and resolution for the plot.  The x-coordinates in the plot will run from -1 to 1 (the models are located at 0 and 1), and the loss value will be evaluated at 5 locations along this line. 
--The two model files contain network parameters describing two configurations (e.g. two distinct minimizers, or two intermediate models along the training trajectory).  The plot will interpolate between these two models.
`--dir_type states` indicates the direction contains dimensions for all parameters as well as the statistics of the BN layers (`running_mean` and `running_var`). Note that ignoring `running_mean` and `running_var` cannot produce correct loss values when plotting two solutions togeather in the same figure. 
--name  test_plot_1  生成的图存储在 test_plot_1文件夹; 此文件夹在model_file的父文件夹里, 此例中为datasets/cifar10/trained_nets/test_plot_1).  

时间: 7 sec * 6  42 sec.

**Remark:**  **由于代码设计的缺陷, 以上代码仍然需要安装mpi相关套件!!!!** 安装步骤可见Appendix "安装MPI". 
这个版本里面, 我们并没有用到mpi. 原代码的设计初衷是: 如果你的server上没有安装mpi, 那么可以用上面这个版本; 如果装了mpi, 那么可以加上mpirun (见后文"Multi-GPU"). 然而, 不装mpi用上面的版本会报错, 因为原代码中即使 args.mpi 为false的情况仍然会用到 mpi的命令. 暂时不知如何修改. 

```
rm -r -f datasets/cifar10/trained_nets/plot_1d_1_99  &&
CUDA_VISIBLE_DEVICES=2 python plot_surface.py --name plot_1d_1_99 --model resnet56 --dataset cifar10 --x=-0.5:1.5:101 --plot --dir_type states --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_99_sd.pt  --cuda --batch_size 2048
```

### 2.2 What to expect during and after runing

  显示 Rank 0 use GPU 0 of 3 GPUs on cyberpower, 这说明确实使用了一个GPU (标号为0). 

每个Iteration是这样的:

> Evaluating rank 0 2/11 (18.2%) coord=-0.6 	train_loss= 2.331 	train_acc=25.22 	time=7.58 	sync=0.00 

时间计算: 每个点 7 sec, 一共7*11= 77 sec = 1.3 min.  (如果 3600个grid points, 那么需要 8**3600 = 24000 sec = 8 hr. )

  生成结果: 一个"图片文件夹“, **datasets/cifar10/trained_nets/test_plot_1d,** 包含了3张 interpolate两个models的图,所有的图都是pdf. 另外还有2个.h5文件, 一个是direction file (model_2 减去 model_1), 一个是 surf_file, 包含程序计算出的loss values. 

Remark: 如果图片文件夹的文件已经存在, 这个脚本不会覆盖它; 所以需要指定新的文件夹. 

  下面是生成的一个图片; 可以看到两条曲线. 
  Remark: 本文中所有图片可在[这个github仓库](https://github.com/ruoyus/online_img/tree/master/uPic)下载.

![Screen Shot 2020-09-15 at 3.51.44 PM](https://cdn.jsdelivr.net/gh/ruoyus/online_img@master/uPic/Screen%20Shot%202020-09-15%20at%203.51.44%20PM.png)

 如何判断这个图片是否合理? 

  第一,横坐标 t=0对应model 1, t=1对应model 2, 我们选的model1 是epoch_1, model2是epoch_25, 因此model1的loss应该远大于model2的loss, 图中蓝色的线 t=0对应loss 2, t=1对应loss 0.3, 是合理的;红色的线, 初始acc 30%, epoch 25的acc = 84%, 也是合理的. 

   第二, 从t=0到1, 先上升, 后下降，并没有太多的bumps, 这和 Goodfellow'14的发现基本一致. 所以也合理.  

  从这个图还能学到一些其他的信息，这里不再一一解读. 



### 2. Multi-GPU by mpirun (on 1D)

多GPU: 使用2个GPU, 用 mpirun. (区别: 用 mpirun开头, 加了 --mpi) 如果没有安装mpi相关套件, 安装步骤可见Appendix "安装MPI". 
使用mpirun的关键参数: **--n 2**. 代码如下: 

```
rm -r -f datasets/cifar10/trained_nets/plot_1d_1_190  &&
CUDA_VISIBLE_DEVICES=0,2 mpirun -n 2 python plot_surface.py --mpi --name plot_1d_1_190 --model resnet56 --dataset cifar10 --x=0:1:11  --dir_type states --plot --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --batch_size 4096
```

  注意: **要用mpirun**才可使用多GPU; **-n num** 设置执行MPI程序的进程总数 (不需要设置 -ngpu 2); **加--mpi** 表示使用mpi. 这三项是多GPU必需的改变.  

时间:  如果--batchsize 2048 (or 4096): 每个model的时间 5 sec, 总时间 30 sec (两个机器各计算6 and 5个values).

运行成功时，输出如下 (rank 1代表一个GPU, rank 0 代表另一个GPU):

```
Evaluating rank 1  0/5  (0.0%)  coord=0.6 	train_loss= 1.947 	train_acc=34.44 	time=6.76 	sync=0.00
Evaluating rank 0  0/6  (0.0%)  coord=0.0 	train_loss= 1.947 	train_acc=31.11 	time=6.92 	sync=0.00
Evaluating rank 1  1/5  (20.0%)  coord=0.7 	train_loss= 1.914 	train_acc=36.10 	time=5.20 	sync=0.00
Evaluating rank 0  1/6  (16.7%)  coord=0.1 	train_loss= 1.955 	train_acc=31.11 	time=5.06 	sync=0.00
```

更长时间的版本: 换成 x=-0.5:1.5:101, 时间 5 min (on 2 GPU). 

```
rm -r -f datasets/cifar10/trained_nets/plot_1d_1_25  &&
CUDA_VISIBLE_DEVICES=0,1,2 mpirun -n 3 python plot_surface.py --mpi  --name plot_1d_1_25 --model resnet56 --dataset cifar10 --x=-0.5:1.5:20  --dir_type states --plot --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_25_sd.pt  --cuda --batch_size 2048
```

之后的代码大部分使用mpirun; 如果单机设置 -n 1即可. 



## 3. 生成2d的plots

本文主要考虑的三种plots是 (Appendix E 有详细解释)

* M2 (1d) 两个模型 + 只画 x: 只生成 xdir= model2 - model1, 生成两点之间的linear interpolation. 不生成ydir. 
* M1 (2d): 一个模型 + 画 x, y: xdir 和 ydir 都是随机向量. 生成某个点附近的landscape. 
* M3 (3m): 三个模型 + 画 x, y: xdir = model2 - model1, ydir = model3 - model 1.  生成三个点张成的平面. 

在这一部分，我们生成2d plots, 包括 2d (1 model) 和 3m (3 models). 默认使用 mpi; 如果单GPU把 -n改成1即可. 

#### 2d (M1): 一个模型附近的2d landscape

和上面的code相比，我们增加了 --y=-1:1:4 (这样会激活2d_plot); 然后把--x改为 --x = -1:1:4.  另外一个区别: 我们只需要看某一个Model附近的landscape, 所以 mode_file2不需要. 

```
 rm -r -f datasets/cifar10/trained_nets/toy_plot_2d_ep10  &&
CUDA_VISIBLE_DEVICES=1,2 mpirun -n 2 python plot_surface.py --mpi --name toy_plot_2d_ep10 --model resnet56 --dataset cifar10 --dir_type weights --x=-1:1:3 --y=-1:1:3 --plot --model_file datasets/cifar10/trained_nets/epoch_10_sd.pt --cuda --batch_size 8192
```

 主要可修改参数:
     --x=-1:1:3 --y=-1:1:3  范围[-1,1], steps 3.  steps是运行时间的关键因素, 呈线性关系: time = steps * const. 
     --name toy_plot_2d: 储存生成的图和loss的文件夹. 如果已经存在,这个script不会覆盖; 因此我们在跑这个script之前先用一行 rm命令删除文件夹. 修改的时候,注意两个文件夹名称都要修改. 
        命名tip: 如果steps较小, 文件夹名加toy; 如果steps较大(e.g. 51), 去掉toy. 见下面"nohup版本". 
     --batchsize: 经过测试, 选2048或8192是最快的; 选256会慢10%-20%. 
     --dir_type  weights. 和M2不同; 这是因为我们要取random direction, 因此不需要考虑 bn 的stats (?) 一般不需要修改. To do: 写一个appendix详细解释到底取 weights和取 states 有什么区别.  

 时间(单机): 一个点大概6秒. 16个点的时间 6*16 = 1.5 min. 其他:6 * 16^2 = 24 min; 6*21^2 = 44min; 51^2 需要 250 min = 4 hr. 

生成结果: 4张2d/3d的图, 3个.h5文件. 

**生成示意图**如下, for epoch 45. 比较spiky; 原文档指出: 对skinny和needle-like的图,可以调整scale的factor. 搜索"skinny" for details.



<img src="Screen Shot 2020-09-16 at 12.28.06 AM-20200916003108256-20200916100940532.png" alt="Screen Shot 2020-09-16 at 12.28.06 AM" style="zoom:40%;" />         <img src="Screen Shot 2020-09-16 at 12.31.47 AM.png" alt="Screen Shot 2020-09-16 at 12.31.47 AM" style="zoom:40%;" />

不挂断nohup 版本: 

```
 rm -r -f datasets/cifar10/trained_nets/plot_2d_ep10  &&
CUDA_VISIBLE_DEVICES=0,1 nohup mpirun -n 2 python plot_surface.py --mpi --name plot_2d_ep10 --model resnet56 --dataset cifar10 --dir_type weights --x=-1:1:41 --y=-1:1:41 --plot --model_file datasets/cifar10/trained_nets/epoch_10_sd.pt --cuda --batch_size 8192 > resnet_2d_0915.out & 2>&1
```

#### 3M: 三个模型张成的平面的landscape

给定三个模型 m1, m2, m3, 画出 span(m1, m2, m3)的loss values. 这是2-model linear interpolation的直接推广. 和M2 (1d)比, 增加--y, 增加 --model_file3.  和M1 (2d)的code相比, 增加 --model_file2, --model_file3, 修改--dir_type 为 states. 

``` 
rm -r -f datasets/cifar10/trained_nets/toy_plot_3m  &&
CUDA_VISIBLE_DEVICES=1 python plot_surface.py --name toy_plot_3m --model resnet56 --dataset cifar10 --x=-1:1:4 --y=-1:1:4 --plot --model_file3 datasets/cifar10/trained_nets/epoch_5_sd.pt  --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt --cuda --ngpu 1 --threads 8 --batch_size 1024
```

注意: --dir_type weights还是 states? 这里默认是 states; 但到底哪个更合理? 为什么 M1, M3 要用states, 而 M2 要用weights? 这部分涉及到batch normalization对formulation的影响, 需要更多讨论. 

​      --x=-1:1:19 --y=-1:1:19: 用多大的精度画图, 越大时间越长.
​     --name toy_plot_3m: 储存生成的图和loss的文件夹. 如果已经存在,这个script不会覆盖; 因此我们在跑这个script之前先用一行 rm命令删除文件夹. 修改的时候,注意两者都修改. 正式命名可改为 plot_3m_ep_20_1_15
​     --batchsize: 经过测试, 选2048或8192是最快的; 选256会慢10%-20%. 
​     --model_file, --model_file2, --model_file3: 三个模型. 

 生成结果: 4张2d/3d的图, 2个.h5文件. 



## 4. 单独画图 (可在laptop上进行)

 在上一步跑完 plot_surface.py之后, 会发现生成的3d plot很奇怪, 这是因为有一些loss特别大(比如1e8), 因此我们希望做一些后期处理, 把特别大的值truncate, 得到比较合理的图. 如果想对图像做其他修改，也可以在这一步完成. 所有的loss 数据都存储在surf_file.h5文件里, size很小, 因此这一步可在laptop上完成.

假设生成的图片和surf_file.h5文件储存 plot_3m_100_50_20 这个文件夹. 标准code如下: 

```
python plot_2D.py --surf_file datasets/cifar10/trained_nets/plot_3m_190_5_1/surf_file.h5 --des_ht 2.0  --show
```

 --des_ht: 期望的3d surface的高度; 比如正常训练的 loss 在0到2之间, 那么可设置 2.0 或 0.5. 
 --show: 加这条的话, 将展示4个图. 其中第4个 3d的图可以直接拖动旋转 (不太灵敏, 要多试试). 
 --调整3d plot的另一个方法: 在plot_2D.py中, 调整 ax.view_init(elev=20,azim= -25) 的数值, 比如 -25 换成 -55 或 50.  

 结果: 重新生成4个图, pdf形式. 
 为测试plot_2D.py是否正常, 可以把母文件夹的4个pdf图全部删除(或移到别处); 然后跑上面的code, 应当生成4个完全相同的 pdf图. 

示意图: 100_50_20, i.e., models at epoch 100, 50, 20. 可以看到, (x,y) = (0,0), (1,0), (0,1)中, (0,0)最低, (0,1)最高, 符合我们的预期: epoch 100的loss最低, epoch 20的loss最高. 从 epoch 20到50是单调下降; 但从20到100不是单调下降, 和Goodfellow'14的报告(单调下降)略有矛盾; 但和之前的1d plot是一致的 (先上升再下降). 

<img src="Screen Shot 2020-09-16 at 12.23.28 AM-0233845.png" alt="Screen Shot 2020-09-16 at 12.23.28 AM" style="zoom:40%;" />

其他可使用的code: 

python plot_2d.py --surf_file datasets/cifar10/trained_nets/plot_3m_ep_1_15_45_old/surf_file.h5 --des_ht 2.0  --show

python plot_2d.py --surf_file datasets/cifar10/trained_nets/plot_3m_ep_1_20_45_old/surf_file.h5 --des_ht 2.0  --show

python plot_2d.py --surf_file datasets/cifar10/trained_nets/plot_3m_ep_190_5_1/surf_file.h5 --des_ht 2.0  --show

python plot_2d.py --surf_file datasets/cifar10/trained_nets/plot_3m_ep_190_10_1/surf_file.h5 --des_ht 2.0  --show

python plot_2d.py --surf_file datasets/cifar10/trained_nets/plot_2d_ep190/surf_file.h5 --des_ht 2.0  --show

  

## 4. 大规模实验的一些code (单机)

我们列举一些大规模实验的script和时间。一般需要用到 nohup. 

###  4.1 运行单个script的简单形式 (2d)

修改需要修改文件名, model_files和log files, 比较麻烦. 我们提供一个统一的脚本`run_1M.py`,  生成2d,1m的Landscape.

**要求**: working_folder = 'datasets/cifar10/trained_nets', model_file放在这个文件夹里, 且以 _sd.pt结尾.

调用方式一 (完整):  

```
python run_1M.py --mod epoch_10_sd.pt --steps 2 --g 0  --date_run 0916
```

 --mod 要处理的 model_file名称;
 --steps 默认x = -1:1:steps, y = -1:1:steps. 取2, 时间20 = 2^2 * 5 秒. 
 --g  使用的GPU index; 可以不指定
 --date_run 运行log文件名中可带有运行日期.

**输出**: plot_folder ="datasets/cifar10/trained_nets/plot_2d_epoch_10'', 该文件夹包含生成的图片. 
         log_file ="log_0916_2d_epoch_10.out", 运行日志.

简化版:

```
python run_1M.py --mod epoch_10_sd.pt --s 5
```

**调用方式二** (只指定epoch index):

```
python run_1M.py --ep 15 --s 3 --g 0 --d 0916
```

### 运行多个scripts

一个GPU 可同时跑多个脚本; 单个脚本5s/point, 而同时跑3个脚本大概9s/point, 速度可以接受. 

```
python run_1M_repeat.py
```





#### 4.1 2d (1M): 一个模型附近的2d landscape

``` 
rm -r -f datasets/cifar10/trained_nets/plot_2d_ep45  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_2d_ep45 --model resnet56 --dataset cifar10 --x=-1:1:41 --y=-1:1:41 --plot --model_file datasets/cifar10/trained_nets/epoch_45_sd.pt --cuda --batch_size 2048 > log_2d_0908.out & 2>&1
```

25*6sec = 2.5 min.  如果是 51^2 需要 250 min = 4 hr.  41^2=160min = 2.5 hr. 

换一个random seed 111:

```
 rm -r -f datasets/cifar10/trained_nets/plot_2d_ep45_seed111  &&
CUDA_VISIBLE_DEVICES=2  nohup   python plot_surface.py --name test_plot_2d_seed111 --model resnet56 --dataset cifar10 --x=-1:1:51 --y=-1:1:51 --plot --model_file datasets/cifar10/trained_nets/epoch_45_sd.pt --cuda --ngpu 1 --threads 8 --batch_size 512 > seed111_2d_0908.out & 2>&1
```

两组实验一起跑: 需要10个小时. 



#### 4.2 M3: 三个模型张成的平面的landscape

1, 10, 45 一个

``` 
rm -r -f datasets/cifar10/trained_nets/test_plot_3m  &&
CUDA_VISIBLE_DEVICES=1  nohup python plot_surface.py --name test_plot_3m --model resnet56 --dataset cifar10 --x=-0.1:1:19 --y=-0.1:1:19 --plot   --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_10_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_45_sd.pt --cuda --ngpu 1 --threads 8 --batch_size 256 > 3m_0908.out & 2>&1
```

100*6sec = 10 min.    400* 10 sec = 1.2 hr.   共用, 20sec, 2.5 hr. 

1, 15,   45 一个

``` 
rm -r -f datasets/cifar10/trained_nets/plot_3m_45_1_15  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_3m_45_1_15 --model resnet56 --dataset cifar10 --x=-0.1:1:41 --y=-0.1:1:41 --plot  --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_15_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_45_sd.pt --cuda --ngpu 1 --threads 8 --batch_size 256 > 3m_45_1_15_0908.out & 2>&1
```

 1, 20, 45 一个, 1600 * 6 = 160min, 2.5 hr. 实际上 10 hr. 

```
rm -r -f datasets/cifar10/trained_nets/plot_3m_45_1_20  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_3m_45_1_20 --model resnet56 --dataset cifar10 --x=-0.1:1:41 --y=-0.1:1:41 --plot   --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_15_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_45_sd.pt --cuda --ngpu 1 --threads 8 --batch_size 256 > 3m_45_1_20_0908.out & 2>&1
```

实际上: 30 sec, 时间 5倍, 一共要 10-12 hrs.



2020.09.12 

用weights来计算: 

rm -r -f datasets/cifar10/trained_nets/plot_3m_100_1_20_weight  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_3m_100_1_20_weight --dir_type weights --model resnet56 --dataset cifar10 --x=-0.1:1:41 --y=-0.1:1:41 --plot  --model_file datasets/cifar10/trained_nets/epoch_100_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_20_sd.pt --cuda --ngpu 1 --batch_size 4096 > log_3m_100_1_20_weight_0912.out & 2>&1

7 sec一个point; 40^2 = 1600 * 7 = 3 hr. 

用states 来计算: 

rm -r -f datasets/cifar10/trained_nets/plot_3m_100_1_20  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_3m_100_1_20 --dir_type states --model resnet56 --dataset cifar10 --x=-0.1:1:41 --y=-0.1:1:41 --plot  --model_file datasets/cifar10/trained_nets/epoch_100_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_20_sd.pt --cuda --ngpu 1 --batch_size 4096 > log_3m_100_1_20_0912.out & 2>&1

用states 来计算: 

rm -r -f datasets/cifar10/trained_nets/plot_3m_100_50_20  &&
CUDA_VISIBLE_DEVICES=0  nohup python plot_surface.py --name plot_3m_100_50_20 --dir_type states --model resnet56 --dataset cifar10 --x=-0.1:1:41 --y=-0.1:1:41 --plot  --model_file datasets/cifar10/trained_nets/epoch_100_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_50_sd.pt --model_file3 datasets/cifar10/trained_nets/epoch_20_sd.pt --cuda --ngpu 1 --batch_size 256 > log_3m_100_50_20_0912.out & 2>&1

2个模型一起跑: 7 sec/epoch. 

3个模型一起跑: 大概 13 sec/epoch;  13*1601 = 20.8 k;  每个模型花了 20k sec = 5.8 hr. 



2020.09.15

 python run_1M_3M_MultiTimes.py --steps 51  --no_1M --no_move

  3*3=9, 需要 90 sec.

  51*51 = 2500, 需要25k sec = 7 hr. 

2020.09.16

python run_1M_3M_MultiTimes.py --steps 51  --no_3M --no_move  

---

## Reference

[1] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. ICLR, 2015.

[2] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. ICLR, 2017.

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```



# Appendix

## Appendix 0. 原始的文档, from Goldstein's github

### Visualizing 1D loss curve

#### Creating 1D linear interpolations

The 1D linear interpolation method [1] evaluates the loss values along the direction between two minimizers of the same network loss function. This method has been used to compare the flatness of minimizers trained with different batch sizes [2].
A 1D linear interpolation plot is produced using the `plot_surface.py` method.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-0.5:1.5:401 --dir_type states \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7 --plot
```

- `--x=-0.5:1.5:401` sets the range and resolution for the plot.  The x-coordinates in the plot will run from -0.5 to 1.5 (the minimizers are located at 0 and 1), and the loss value will be evaluated at 401 locations along this line.
- `--dir_type states` indicates the direction contains dimensions for all parameters as well as the statistics of the BN layers (`running_mean` and `running_var`). Note that ignoring `running_mean` and `running_var` cannot produce correct loss values when plotting two solutions togeather in the same figure.  
- The two model files contain network parameters describing the two distinct minimizers of the loss function.  The plot will interpolate between these two minima.

![VGG-9 SGD, WD=0](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_loss_landscape_goldstein_new/doc/images/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1_model_300.t7_vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1_model_300.t7_states.h5_[-1.0,1.0,401].h5_1d_loss_acc.jpg)

  备注: 上面的code用的是 mode.t7, 如果你生成的模型是 model.pt怎么办, 如何转化?  答案: 不用转化，直接用mode.pt和.t7都可以, 因为读取方式是一样的, 都是torch.load. 

#### Producing plots along random normalized directions

A random direction with the same dimension as the model parameters is created and "filter normalized."
Then we can sample loss values along this direction.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-1:1:51 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --plot
```

 - `--dir_type weights` indicates the direction has the same dimensions as the learned parameters, including bias and parameters in the BN layers.
 - `--xnorm filter` normalizes the random direction at the filter level. Here, a "filter" refers to the parameters that produce a single feature map.  For fully connected layers, a "filter" contains the weights that contribute to a single neuron.
 - `--xignore biasbn` ignores the direction corresponding to bias and BN parameters (fill the corresponding entries in the random vector with zeros).


 ![VGG-9 SGD, WD=0](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_loss_landscape_goldstein_new/doc/images/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7_weights_xignore=biasbn_xnorm=filter.h5_[-1.0,1.0,51].h5_1d_loss_acc.jpg)



We can also customize the appearance of the 1D plots by calling `plot_1D.py` once the surface file is available.


### Visualizing 2D loss contours

To plot the loss contours, we choose two random directions and normalize them in the same way as the 1D plotting.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
```

![ResNet-56](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_loss_landscape_goldstein_new/doc/images/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5_train_loss_2dcontour.jpg)

Once a surface is generated and stored in a `.h5` file, we can produce and customize a contour plot using the script `plot_2D.py`.

```
python plot_2D.py --surf_file path_to_surf_file --surf_name train_loss
```

- `--surf_name` specifies the type of surface. The default choice is `train_loss`,
- `--vmin` and `--vmax` sets the range of values to be plotted.
- `--vlevel` sets the step of the contours.


### Visualizing 3D loss surface

`plot_2D.py` can make a basic 3D loss surface plot with `matplotlib`.
If you want a more detailed rendering that uses lighting to display details, you can render the loss surface with [ParaView](http://paraview.org).

![ResNet-56-noshort](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_loss_landscape_goldstein_new/doc/images/resnet56_noshort_small.jpg) ![ResNet-56](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_loss_landscape_goldstein_new/doc/images/resnet56_small.jpg)

To do this, you must

1. Convert the surface `.h5` file to a `.vtp` file.

```
python h52vtp.py --surf_file path_to_surf_file --surf_name train_loss --zmax  10 --log
```

   This will generate a [VTK](https://www.kitware.com/products/books/VTKUsersGuide.pdf) file containing the loss surface with max value 10 in the log scale.

2. Open the `.vtp` file with ParaView. In ParaView, open the `.vtp` file with the VTK reader. Click the eye icon in the `Pipeline Browser` to make the figure show up. You can drag the surface around, and change the colors in the `Properties` window.
3. If the surface appears extremely skinny and needle-like, you may need to adjust the "transforming" parameters in the left control panel.  Enter numbers larger than 1 in the "scale" fields to widen the plot.
4. Select `Save screenshot` in the File menu to save the image.





### Appendix A: Debugging Basic Code

原code的bug集锦: https://github.com/tomgoldstein/loss-landscape/issues/4

### A.1 第一个可跑的版本debug

最开始在本地Macbook上测试. 我使用的版本: (在本地Macbook上测试)

 ```
python plot_surface.py --name test_plot --model resnet56 --dataset cifar10 --x=-1:1:51 --y=-1:1:51 --plot --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt
 ```

Bug 1: ModuleNotFoundError: No module named 'models', 搜索发现: model.pt文档应当只储存 model.state_dict(), 而且对应的key必须是 'state_dict'.  所以需要重新生成 .pt 文件. 

```
with open( save_dir, 'wb') as f:
    torch.save( { 'state_dict': model.state_dict() } , f)  
```

Bug2： Missing key(s) in state_dict: "conv1.weight"
             Unexpected key(s) in state_dict: "pre_clf.0.weight
  这说明存储模型用的models文件和这里的models文件使用的keys不同. 需要把这里的models文件拿过去用. 
  花了3个小时找直接的解决方案, 未能找到. 因此决定替换models.py: 把sgd.py对应的models.py换成Hao Li使用的版本, 如果能训练出来，那么应该不会出现版本不兼容的情况. 

Bug3: 没有安装 mpi.
 解决方案: 把若干出现mpi的地方改为 if args.mpi:  mpi.blablabla
 但发现出现mpi的地方太多,跑起来还是有错误, 所以我们决定还是安装 mpi. 找了一些攻略, 用homebrew和pip安装了mpi.  

"Bug"4: 太慢. 
   已经可以跑了，但每个iteration是这样的:

> 0/2601 (0.0%) coord=[-1. -1.] 	train_loss= 5838.750 	train_acc=10.00 	time=299.07 

需要5分钟*2600 = 200 hr = 9 days. 所以还是在 GPU上跑起来. 发现可以加速 40倍: 300 sec --> 7 sec.

### A.2 后期的bugs

  Bug1: mpi错误. 没有使用Goldstein的版本 mpirun -n 2, 而是用了后人的版本 mpirun -ngpu 2, 结果导致没有分配到2个gpu上. 

 Bug2: 同样的bug出现在[这个帖子](https://github.com/tomgoldstein/loss-landscape/issues/4) (exactly same visualization code), 或者更一般的问题[关于h5py lock](https://github.com/tomgoldstein/loss-landscape/pull/28)

```
Traceback (most recent call last):
 File "plot_surface.py", line 368, in <module>
  crunch(surf_file, net, w, s, d, loader, f'{dataset}_loss', f'{dataset}_acc', comm, rank, args)
 File "plot_surface.py", line 92, in crunch
  f = h5py.File(surf_file, 'r+' if rank == 0 else 'r') # check https://docs.h5py.org/en/stable/quick.html for h5py files
 File "/home/ruoyus/anaconda3/lib/python3.7/site-packages/h5py/_hl/files.py", line 394, in __init__
  swmr=swmr)
 File "/home/ruoyus/anaconda3/lib/python3.7/site-packages/h5py/_hl/files.py", line 172, in make_fid
  fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
 File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
 File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
 File "h5py/h5f.pyx", line 85, in h5py.h5f.open
OSError: Unable to open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
```

 简单来说: 如果版本更新, 多机运算有错误; 可能要pip install h5py==2.7.0

 具体的方案(作者的回答): Version info printed from `dpkg -s libhdf5-dev` and `print(h5py.version.hdf5_version)` are not necessarily consistent. h5py 2.8.0 will print out `1.10.2` and h5py 2.7.0 will print out `1.8.18`. So please let me know your version info by `print(h5py.version.hdf5_version)`.  
     Installing h5py 2.7.0 should solve this problem, i.e., `pip install h5py==2.7.0`. You can check the installed h5py version by `pip list`. Note that use `pip2` if python 3 is also installed

   但后来有人贡献了一个commit, 加了一行code解决了这个bug. 所以不用管. 

**Bug3:** 

```
Computing 7 values for rank 0
Traceback (most recent call last):
 File "plot_surface.py", line 368, in <module>
  crunch(surf_file, net, w, s, d, loader, f'{dataset}_loss', f'{dataset}_acc', comm, rank, args)
 File "plot_surface.py", line 152, in crunch
  losses = mpi.reduce_max(comm, losses)
 File "/home/ruoyus/landscape2020/2018_landscape_goldstein_2D/mpi4pytorch.py", line 86, in reduce_max
  comm.Reduce(array, total, op=mpi4py.MPI.MAX, root=0)
```

 解决方案: 先run 

```
rm -r -f datasets/cifar10/trained_nets/test_plot_2d
```

**Bug4:** 09/16 所有的计算都是misleading的，可能因为data pre-processing的缘故.  

 sgd.py使用了data augmentation来生成training data set; 而在interpolation时我们计算的是无data augmentation版本的training set的loss. 
  经检验发现, 原本训练过程的training loss, epoch 1 是1.8, epoch 25 是 1.34; 在我们的1d plot里, plot_epoch_1_25里的training_loss.pkl.txt记载的数据, t = 0的loss是1.98, t=1的loss是1.48; 都比原来的高. 这很有可能是因为data augmentation使用方法不同导致的.

  解决方案一: sgd.py的training data set不用data augmentation.

  解决方案二: surface_plot中使用的training_data set改成和sgd.py一样的data augmentation. 



### Appendix B  本地Laptop 测试的code

#### Locally without GPU

Implicit (short version):

```shell script
python plot_surface.py --name test_plot_1 --model resnet56 --dataset cifar10 --x=-1:1:11 --plot --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --ngpu 1 --threads 8 --batch_size 256
```

每个点需要 320 sec (比GPU的8 sec慢40倍), 一共 5min*6 = 30 min.  不过稍微多一些grid points就太慢了. 所以本地laptop可以帮助debug, 但不建议拿来做真正的计算. 

Explicit (long version):

```shell script
python plot_surface.py --name test_plot --model resnet56 --dataset cifar10 --x=-1:1:51 --y=-1:1:51 --plot \
--model_file datasets/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn
```



### Appendix C: 查看CPU个数

\# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
 \# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

 \# 查看物理CPU个数: 1
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

\# 查看每个物理CPU中core的个数(即核数): 8
 cat /proc/cpuinfo| grep "cpu cores"| uniq

\# 查看逻辑CPU的个数: 16
 cat /proc/cpuinfo| grep "processor"| wc -l

在我们的server上，只有1个物理CPU, 有8个core, 16个逻辑CPU, 因此可以当做16个CPU. 在code里选择--threads 8

--ngpu 1, 说明只用到一个GPU.如果是16个CPU，之前推荐 --threads 8. 关于逻辑CPU和物理CPU的区别,见附件. 



### Appendix D: 对比 --states 和 --weights

如果使用 --weights: Fig D.1

```
python plot_surface.py --name test_plot_1 --model resnet56 --dataset cifar10 --x=-1:1:11 --plot --dir_type states --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --ngpu 1
```

如果使用 --states:  Fig D.2

```
python plot_surface.py --name test_plot_1 --model resnet56 --dataset cifar10 --x=-1:1:11 --plot --dir_type states --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --ngpu 1
```

![1d_demo_weights](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_landscape_goldstein_2D/datasets/cifar10/trained_nets/test_plot_1/1d_demo_weights.png) ![1d_demo_states](/Users/ruoyusun/Desktop/SunDirac_Mac/科研/我的课题/1_我写的paper/2018 DL survey/CODE for landscape/2018_landscape_goldstein_2D/datasets/cifar10/trained_nets/test_plot_1/1d_demo_states.png)

​                   Fig D.1                                       Fig D.2

哪个是正确的？因为 横坐标0对应的是model_1 (at epoch 1), 横坐标 1对应的是model_2 (at epoch 2), 因此蓝色的线从0到1 应该是递减 (training loss at epoch 2 < training loss at epoch 1), 所以右图才是正确的. 

为什么默认--dir_type = weights 而不是 states? 这个需要之后figure out.  



### Appendix E: 方向 direction file 的设置

我们解释核心函数之一 net_plotter.setup_direction 的主要原理. 

如果dir_file存在, 那么不进行任何操作. 

当dir_file不存在时，根据 model1, model2, model3 设置方向. xdir 即 xdirection, ydir 即 ydirection 
* 第一层选择: 两个模型, or 一个模型, or 三个模型. 根据 args 的 --model_file2 和 --model_file3是否存在决定.
* 第二层选择: 画1d 还是 2d, 根据 args的 --y 是否存在来决定. 如果有 --y, 那么画 2d. 

这个函数对如下的5中情况分别定义了xdir, 或 xdir and ydir. 
* 两个模型 + 只画 x (1d, 2m): 只有 xdir= model2 - model1

* (少见) 两个模型 + 画x,y: xdir = model2 - model 1, 但另一个方向ydirection 是随机

* (少见) 一个模型 + 只画 x: 只有 xdir = random. 

* 一个模型 + 画 x, y (2d, 1m): xdir 和 ydir 都是随机

* 三个模型 + 画 x, y (2d, 3m): xdir = model2 - model1, ydir = model3 - model 1. 

我们主要考虑三个情况:  1d_2m, 2d_1m, 2d_3m. 为简单起见，我们把它们叫做 1d, 2d, 3m. 
另外两个“少见”的情况不做考虑. 它们也有意义。比如1d_1m可以当做是 1d_2m的一个特殊情况, 也可以查看是否是local minimum. 

**Explicit (long version)**: 除了 --dir_type weights 或 states之外, 如下的控制变量 (默认变量) 也会影响方向；但除了默认的选择，似乎原代码没有提供别的备选项，需要后期研究. 

```shell script
--xnorm filter --xignore biasbn --ynorm filter --yignore biasbn 
```



### Appendix F: 对新的数据集复现实验

## What exactly do I need to do to make it work for new data sets?

1. If you have a new dataset: add a new folder ``datasets/{your_dataset_name}``.

2. Add you data to ``datasets/{your_dataset_name}/data``.

3. Add the model definitions to a file in ``datasets/{your_dataset_name}/models``.

4. Add your trained network to a file in ``datasets/{your_dataset_name}/trained_nets/{your_model_with_hyper_parameters}``.

5. Add a file ``data_loader.py`` in ``datasets/{your_dataset_name}`` and implement the method ``get_data_loaders()``. You can find documentation in [data_loader.py](datasets/cifar10/data_loader.py).

6. Add a file ``model_loader.py`` in ``datasets/{your_dataset_name}`` and implement the method ``load()``. Also add to the file a dictionary called ``models`` containing a mapping between the name of your model and the model function. You can find documentation in [model_loader.py](datasets/cifar10/model_loader.py).

   

### Appendix G 其他知识

​    filter_norm的意义是什么, 如何使用, 有没有别的替代品; 4个plots分别是什么含义. 

​    和原始的code相比, 我修改了 dir_file和surf_file的命名. 如果用原来的code, 这个文件夹有两个.h5文件, 一个储存了"方向", 一个储存了2d平面所有的loss. 之后进行后续操作, 比如画图, 可以再次调用plot_2D来画图. 需要先这样操作: 修改其中一个文件名为 surf_file.h5. 这个文件比另一个小很多 (e.g. 32kb v.s. 7mb), 名字更长包含 [-0.1,1.0,41]x[-0.1,1.0,41]. 它是由 plot_surface.py中储存 surf_file 这一步所得到的文件. 之后再对 surf_file.h5 操作就比较简单. 

​    如果使用plot_2D中的其他画图函数, 可以参考这个code:   python -c 'import plot_2D; plot_2D.plot_2d_contour("datasets/cifar10/trained_nets/plot_3m_100_1_20/surf_file.h5")'



### Appendix H  不同Batch size 和不同GPU 数量的速度对比

 简单来说: batch size 2048 -8192是最快的; 超过 16384速度开始下降; 2个GPU所花时间是1个GPU的一半; 3个GPU所花时间是1个GPU的1/3.

1d 实验

```
rm -r -f datasets/cifar10/trained_nets/plot_1d_1_190  &&
CUDA_VISIBLE_DEVICES=0,2 mpirun -n 2 python plot_surface.py --mpi  --name plot_1d_1_190 --model resnet56 --dataset cifar10 --x=0:1:11  --dir_type states --plot --model_file datasets/cifar10/trained_nets/epoch_1_sd.pt --model_file2 datasets/cifar10/trained_nets/epoch_2_sd.pt  --cuda --batch_size 4096
```

| \# of GPUs | batchsize | 单点时间 (sec) | 总时间 (sec) | Thread | 显存     | GPU_uti |
| ---------- | :-------- | -------------- | ------------ | ------ | -------- | ------- |
| 1          | 128       | 7.8            | 78           | 8      |          |         |
| 1          | 512       | 6.7            | 67           | 2      | 0.98G    | 25--52% |
| 1          | 2048      | 5.1            | 51           | 8      |          |         |
| 1          | 8192      | 5.04           | **50.0**     | 8      | 3.4G     |         |
| 1          | 16384     | 6              | 62           | 8      | **6.1G** | 100%    |
| 2          | 128       |                |              |        |          |         |
| 2          | 512       |                |              |        |          |         |
| 2          | 2048      |                |              |        |          |         |
| 2          | 8192      |                |              |        |          |         |
|            |           |                |              |        |          |         |

 注: threads = 2 or 8 (default) 不影响时间 (至少对单机 batchsize 8192是如此).

2d实验:

| Size | \# of GPUs | batchsize | 时间 (sec) | sync时间 | 总时 | Thread | 显存     | GPU_uti  |
| ---- | ---------- | :-------- | ---------- | -------- | ---- | ------ | -------- | -------- |
| 16   | 1          | 128       |            |          |      | 8      |          |          |
|      | 1          | 1024      | 83.7       |          |      | 8      | 2.5->1.2 | 58%      |
|      | **1**      | **2048**  | **78**     |          |      | **8**  | **2.3G** | **77%**  |
|      | 1          | 8192      | 78         |          |      | 8      | 3.4 G    | 92%      |
|      | 2          | 2048      | 49         | 0        |      |        | 1.4G     | 100%     |
|      | **2**      | **4096**  | **44**     |          |      | **2**  | **2.1G** | **100%** |
|      | 2          | 8192      | 48         | 5.5      |      | 8      |          |          |
|      | 3          | 128       | 97         | 0.1      |      | 8      | 0.8G     | 30%      |
|      | 3          | 512       | 77         | 0        |      | 8      | 0.8G     | 50%      |
|      | 3          | 2048      | 45         | 0.44     |      | 8      |          |          |
|      | **3**      | **4096**  | **37**     | 0.14     |      | 8      | 2.1G     | 100%     |
|      | **3**      | **8192**  | **37**     | 1.8--3.5 |      | 8      | 3.4G     | 100%     |
|      | 3          | 16384     | 43         |          |      | 8      | 6.1G     | 100%     |
| 361  | 3          | 8192      | 741        | 84       | 14 m |        |          |          |
| 2601 | 1          |           |            |          | 3.5h |        |          |          |

4^2 = 16 --> 37 sec; 那么 15^2 = 225 --> 500 sec = 9 min. 实际 5*75=6 min.  19^2 = 361, 时间 361/3 * 6 = 12 min; 实际时间 820 sec = 14 min.  

## Appendix I 安装mpi

最简单的方法: 

```
conda install mpi4py 
```

安装过程中将见到如下输出: 

The following packages will be downloaded:

  package          |      build

  ---------------------------|-----------------

  _libgcc_mutex-0.1     |       main      3 KB

  conda-4.8.4        |      py37_0     3.0 MB

  conda-package-handling-1.6.1|  py37h7b6447c_0     886 KB

  mpi-1.0          |      mpich     13 KB

  mpi4py-3.0.3        |  py37hf046da1_1     644 KB

  mpich-3.3.2        |    hc856adb_0     6.4 MB

  \------------------------------------------------------------

​                      Total:    11.0 MB

另外一种方法: 官方网站 [mpi4py 2.0.0](https://mpi4py.readthedocs.io/en/stable/install.html) 推荐使用pip install mpi4py, 但也指出需要先安装mpi的基本架构之后再安装mpi4py (即mpi for python). 其中一种选择是 mpich, 这个选择下的安装步骤为:

```
 $ brew install mpich 
 $ pip install mpi4py
```

相当于安装上面列表里的最后两个步骤。在Laptop上安装成功;但安装mpi4py却在Ubuntu上失败了，原因未知. 所以选择 conda install mpi4py.  

### Appendix J: 移动code的command

scp ruoyus@sun.csl.illinois.edu:/home/ruoyus/landscape2020/2018_landscape_goldstein_2D/run_2d_multiGPU.py    /Users/ruoyusun/Desktop/SunDirac_Mac/科研/Coding/CODE_landscape/2018_landscape_goldstein_2D



### Appendix K: 大的修改

增加的code: sgd.py, utils.py, models_sgd_walk.py, plot_loss.py, run_1M.py, run_1M_repeat.py, 
修改的文件夹: 把cifar10放到 datasets的子文件夹

09/08: 开始跑代码,解决若干文件存储的bugs. 安装mpi; 
09/09: 加入sgd.py, 修改model存储部分, 使其和当前的代码兼容.  
09/10: 写run_1M.py, run_1M_3M_MultiTimes.py, 可以一键跑多个脚本. 开始使用多机的code.
09/14: 修改2d_Plot.py 的后期处理; 允许调整. 
09/15: 放弃name_direction_file和name_surf_file这两个函数; 改用简单的命名方式 direction.h5, surf_file.h5.
      原来的复杂命名是为了后面查看文件夹时，知道参数设置; 因此增加了一个 ALL_args.txt文件, 记录所有的自变量的取值. 
09/16: 重写了 run_1M.py 和 run_1M_repeat.py. 几个改变: i) 引入了自变量, 可以控制输入; ii)同时允许 model_file和 epoch_index两种输入文件名的方法;  iii) 之前的写法用了 && 来使得几个脚本同时运行; 但for循环自动并行运行几个脚本，因此&&可以去掉. 
   TBD: 

* 修改sgd.py, check生成的loss list, 以和后期画图的对比;  check是否data preprocessing导致了结果不同;
*  修改run_3M.py; 
* 在2D_plot里修改3d plot的画图方式(加上vmax=10的限制), 去掉 des_ht. 
* 生成一些solid的图; 



