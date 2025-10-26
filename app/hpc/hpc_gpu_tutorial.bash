# setup read-only git on vanda hpc machine:
Host github.com-finsight-quantlab
        Hostname github.com
        User git
        IdentityFile ~/.ssh/id_ed25519_samgit #sam's private key to use with just the finsight-quantlab project
        IdentitiesOnly yes

#test the mapping
ssh -T git@github.com-finsight-quantlab

qstat -q

#    server: stdct-m
#
#    Queue            Memory CPU Time Walltime Node   Run   Que   Lm  State
#    ---------------- ------ -------- -------- ---- ----- ----- ----  -----
#    workq              --      --       --     --      0     0   --   D S
#    interactive_cpu    --      --    12:00:00    2     1     0   --   E R
#    interactive_gpu    --      --    12:00:00    2     3     3   --   E R
#    batch_cpu          --      --    240:00:0   10    85    11   --   E R
#    batch_gpu          --      --    168:00:0    2    32     2   --   E R
#    auto               --      --       --     --      0     0   --   E R
#    cpu_serial         20gb    --    168:00:0   10     0     0   --   E R
#    cpu_parallel       --      --    168:00:0   10    15     4   --   E R
#    gpu                --      --    48:00:00  --      2     1   --   E R
#    large_mem          --      --    120:00:0    2     0     0   --   E R
#    auto_free          --      --       --     --      0     0   --   E R
#    gpu_amd            --      --    48:00:00  --      0     0   --   E R
#                                                   ----- -----
#                                                     138    21
#
#    Queue            Max Memory Max CPUs Max GPUs Max Walltime Node   Run   Que   Lm  State
#    ---------------- ------     -------- -------- -------- ---- ----- ----- ----  -----
#    interactive_cpu    --         --         -      12:00:00    2     2     0   --   E R
#    interactive_gpu    --         --         2     12:00:00    2     7     3   --   E R
#    batch_cpu          --         720        0     240:00:0   10    70    10   --   E R
#    batch_gpu          --         720        2      168:00:0    2    56     2   --   E R
#    auto               --         --                  --     --      0     0   --   E R -- route_destinations = batch_cpu,batch_gpu
#    cpu_serial         20gb       1          0     168:00:0   10     0     0   --   E R
#    cpu_parallel       --         640        0     168:00:0   10    16    10   --   E R
#    gpu                --         36         2     48:00:00  --      5     1   --   E R
#    auto_free          --         --                  --     --      0     0   --   E R -- route_destinations = cpu_serial,cpu_parallel,gpu
#    gpu_amd                       144              48:00:00

scp "frlivanda:/home/svu/$USER/pbs/sam_rag_train.*" ~/Projects/FinSight_BackEnd/app/hpc/logs
scp -r "frlivanda:/scratch/$USER/*.stdct-mgmt-02" ~/Projects/FinSight_BackEnd/app/hpc/logs



ssh-keygen -t ed25519 -C "user@hpc.url.edu" -f ~/.ssh/id_ed25519_atlas9login
ssh-copy-id -i ~/.ssh/id_ed25519_atlas9login.pub user@hpc.url.edu
exit
ssh -i /Users/SamarthSoni/.ssh/id_ed25519_atlas9login 'user@hpc.url.edu'
ssh -J user@hpc.url.edu e1554287@cnode-33-43-31 -L 16108:localhost:37521
ll
vim ~/.ssh/config
# add config "atlas" here
ssh-keygen -t ed25519 -C "user@vandahpc.url.edu" -f ~/.ssh/id_ed25519_vandalilogin
cat ~/.ssh/id_ed25519_vandalilogin.pub
#ssh-ed25519 xxx user@vandahpc.url.edu
vim ~/.ssh/config
# add config "frlivanda" here
ssh frlivanda
amgr login
qstat
qstat -Qf
# to check queues details
qstat -Q
# tabular queue details
#Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
#---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
#workq                0     0  no  no     0     0     0     0     0     0 Exe*
#interactive_cpu      0     0 yes yes     0     0     0     0     0     0 Exe*
#interactive_gpu      0     7 yes yes     0     4     3     0     0     0 Exe*
#batch_cpu            0    56 yes yes    10    46     0     0     0     0 Exe*
#batch_gpu            0    47 yes yes     0    45     2     0     0     0 Exe*
#auto                 0     0 yes yes     0     0     0     0     0     0 Rou*
#cpu_serial           0     0 yes yes     0     0     0     0     0     0 Exe*
#cpu_parallel         0    19 yes yes     6    13     0     0     0     0 Exe*
#gpu                  0     8 yes yes     1     6     1     0     0     0 Exe*
#large_mem            0     0 yes yes     0     0     0     0     0     0 Exe*
#auto_free            0     0 yes yes     0     0     0     0     0     0 Rou*
#gpu_amd              0     0 yes yes     0     0     0     0     0     0 Exe*
qsub -q interactive_cpu -I -l select=1:ncpus=2:mem=4gb -l walltime=00:05:00 -A finsight
# start interactive cpu sessions
#    qsub: waiting for job 304553.stdct-mgmt-02 to start
#    qsub: job 304553.stdct-mgmt-02 ready
#    
#    hostname
#    [user@NODEID ~]$ hostname
#    NODEID
#    [user@NODEID ~]$ lscpu | egrep '^CPU|Thread|Core|Socket'
#    CPU op-mode(s):                     32-bit, 64-bit
#    CPU(s):                             72
#    CPU family:                         6
#    Thread(s) per core:                 1
#    Core(s) per socket:                 36
#    Socket(s):                          2
#    [user@NODEID ~]$ python3 - <<'PY'
#    > import multiprocessing, tensorflow as tf
#    print("CPUs visible:", multiprocessing.cpu_count())
#    tf.config.threading.set_intra_op_parallelism_threads(2)
#    tf.config.threading.set_inter_op_parallelism_threads(1)
#    a=tf.random.uniform([3000,3000]); b=tf.random.uniform([3000,3000]); _=tf.linalg.matmul(a,b).numpy()
#    print("TF CPU matmul OK")
#    PY
#    [user@NODEID ~]$ exit
#    qsub: job 304553.stdct-mgmt-02 completed

qsub -q interactive_gpu -I -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=00:10:00
# start interactive gpu sessions
#    qsub: waiting for job 304556.stdct-mgmt-02 to start
#    qsub: job 304556.stdct-mgmt-02 ready
#
#    [user@GPUNODEID ~]$ hostname
#    GPUNODEID
#    [user@GPUNODEID ~]$ nvidia-smi
#    Thu Oct 23 11:32:11 2025
#    +-----------------------------------------------------------------------------------------+
#    | NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
#    |-----------------------------------------+------------------------+----------------------+
#    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
#    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
#    |                                         |                        |               MIG M. |
#    |=========================================+========================+======================|
#    |   0  NVIDIA A40                     Off |   00000000:0A:00.0 Off |                    0 |
#    |  0%   30C    P8             22W /  300W |       0MiB /  46068MiB |      0%      Default |
#    |                                         |                        |                  N/A |
#    +-----------------------------------------+------------------------+----------------------+
#
#    +-----------------------------------------------------------------------------------------+
#    | Processes:                                                                              |
#    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
#    |        ID   ID                                                               Usage      |
#    |=========================================================================================|
#    |  No running processes found                                                             |
#    +-----------------------------------------------------------------------------------------+
#    [user@GPUNODEID ~]$ =>> PBS: job killed: walltime 668 exceeded limit 600
#
#    qsub: job 304556.stdct-mgmt-02 completed

mkdir pbs
mkdir projects
ll ~/.ssh
vim ~/.ssh/authorized_keys
# add my ssh public key here
mv projects/ projects-sam
cat ~/.ssh/id_ed25519_samgit.pub
ssh-ed25519 xxx samgit@lisvanda
cd projects-sam/
git clone git@github.com:samarthsoni17/FinSight-QuantLab.git
ssh -T git@github.com
vim ~/.ssh/config
# add config for github.com-finsight-quantlab
ssh -T git@github.com-finsight-quantlab
ssh: connect to host github.com port 22: Connection timed out
ssh -vT git@github.com-finsight-quantlab
# still didnt work
chmod 600 ~/.ssh/id_ed25519_samgit
chmod 700 ~/.ssh
vim ~/.ssh/config
ssh -vT git@github.com-finsight-quantlab
# still didnt work, i realised hpc doesnt allow outgoing connections to ports 22/443
git clone https://github.com/samarthsoni17/FinSight-QuantLab.git
#    Cloning into 'FinSight-QuantLab'...
#    Username for 'https://github.com': samarthsoni17
#    Password for 'https://samarthsoni17@github.com':
#    remote: Enumerating objects: 103, done.
#    remote: Counting objects: 100% (103/103), done.
#    remote: Compressing objects: 100% (92/92), done.
#    remote: Total 103 (delta 11), reused 91 (delta 8), pack-reused 0 (from 0)
#    Receiving objects: 100% (103/103), 15.00 MiB | 16.46 MiB/s, done.
#    Resolving deltas: 100% (11/11), done.
#    Updating files: 100% (82/82), done.
cd FinSight-QuantLab/
git status
#    On branch develop
#    Your branch is up to date with 'origin/develop'.
#
#    nothing to commit, working tree clean
pwd
#/home/svu/user/projects-sam/FinSight-QuantLab
ll /
ll /atlas
ls /atlas/hpctmp/
ll /atlas/hpctmp/$USER
#total 0
hpc s
#
#     -----------------------------------------------------------------
#       Reported at: Thu Oct 23 03:55:49 PM +08 2025
#     -----------------------------------------------------------------
#       Disk space for HPC Vanda home directory
#     -----------------------------------------------------------------
#              HPC Home Dir      Usage      Quota      Status
#        /home/svu/user    104.64k     40.00G       0.0%
#     =================================================================
#
#     -----------------------------------------------------------------
#       Disk space for Vanda Scratch /scratch/user
#     -----------------------------------------------------------------
#           HPC /scratch Dir      Usage      Quota      Status
#         /scratch/user       0.00    500.00G       0.0%
#     =================================================================

module purge
#  Unloading shared
#    WARNING: Did not unuse /cm/shared/modulefiles
module avail 2>&1 | grep -i tensorflow # to check available tensorflow modules
#    dm-haiku/0.0.9-foss-2022a                    libspatialindex/1.9.3-GCCcore-11.3.0                           tensorflow-probability/0.16.0-foss-2021b
#    dm-haiku/0.0.13-foss-2023a                   libspatialite/5.0.1-GCC-11.2.0                                 tensorflow-probability/0.19.0-foss-2022a
#    dm-tree/0.1.6-GCCcore-10.3.0                 LIBSVM-Python/3.30-foss-2022a                                  TensorFlow/2.5.3-foss-2021a
#    dm-tree/0.1.6-GCCcore-11.2.0                 LIBSVM/3.25-GCCcore-11.2.0                                     TensorFlow/2.6.0-foss-2021a
#    dm-tree/0.1.8-GCCcore-11.3.0                 LIBSVM/3.30-GCCcore-11.3.0                                     TensorFlow/2.7.1-foss-2021b
#    dm-tree/0.1.8-GCCcore-12.3.0                 libtasn1/4.18.0-GCCcore-11.2.0                                 TensorFlow/2.8.4-foss-2021b
#    DMLC-Core/0.5-GCC-10.3.0                     libtasn1/4.19.0-GCCcore-11.3.0                                 TensorFlow/2.9.1-foss-2022a
#    DMLC-Core/0.5-GCCcore-11.3.0                 libtasn1/4.19.0-GCCcore-12.3.0                                 TensorFlow/2.9.1-foss-2022a-CUDA-11.7.0
#    dominate/2.8.0-GCCcore-11.3.0                LibTIFF/4.0.10-GCCcore-8.3.0                                   TensorFlow/2.11.0-foss-2022a
#    double-conversion/3.1.4-GCCcore-8.3.0        LibTIFF/4.1.0-GCCcore-10.2.0                                   TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#    double-conversion/3.1.5-GCCcore-9.3.0        LibTIFF/4.2.0-GCCcore-10.3.0                                   TensorFlow/2.13.0-foss-2023a
#    Horovod/0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.9.1   Szip/2.1.1-GCCcore-12.3.0
#    Horovod/0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.11.0  Szip/2.1.1-GCCcore-13.2.0

qstat -q
qstat -Q #more queue info
#    Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
#    ---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
#    workq                0     0  no  no     0     0     0     0     0     0 Exe*
#    interactive_cpu      0     1 yes yes     0     1     0     0     0     0 Exe*
#    interactive_gpu      0    10 yes yes     0     7     3     0     0     0 Exe*
#    batch_cpu            0    81 yes yes    10    70     0     0     0     1 Exe*
#    batch_gpu            0    58 yes yes     0    56     2     0     0     0 Exe*
#    auto                 0     0 yes yes     0     0     0     0     0     0 Rou*
#    cpu_serial           0     0 yes yes     0     0     0     0     0     0 Exe*
#    cpu_parallel         0    26 yes yes    10    16     0     0     0     0 Exe*
#    gpu                  0     6 yes yes     0     5     1     0     0     0 Exe*
#    large_mem            0     0 yes yes     0     0     0     0     0     0 Exe*
#    auto_free            0     0 yes yes     0     0     0     0     0     0 Rou*
#    gpu_amd              0     0 yes yes     0     0     0     0     0     0 Exe*
qstat -Qf #full length queue info
git config --global credential.helper store
git config credential.helper store
vim train_tf_smoke.py
vim $HOME/pbs/tf_smoke.pbs
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0 && module list
#    Loading TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#      Loading requirement: GCCcore/11.3.0 zlib/1.2.12-GCCcore-11.3.0 binutils/2.38-GCCcore-11.3.0 GCC/11.3.0 ncurses/6.3-GCCcore-11.3.0
#        numactl/2.0.14-GCCcore-11.3.0 XZ/5.2.5-GCCcore-11.3.0 libxml2/2.9.13-GCCcore-11.3.0 libpciaccess/0.16-GCCcore-11.3.0 hwloc/2.7.1-GCCcore-11.3.0
#        OpenSSL/1.1 libevent/2.1.12-GCCcore-11.3.0 libfabric/1.15.1-GCCcore-11.3.0 PMIx/4.1.2-GCCcore-11.3.0 UCX/1.12.1-GCCcore-11.3.0 UCC/1.0.0-GCCcore-11.3.0
#        OpenMPI/4.1.4-GCC-11.3.0 OpenBLAS/0.3.20-GCC-11.3.0 FlexiBLAS/3.2.0-GCC-11.3.0 FFTW/3.3.10-GCC-11.3.0 gompi/2022a FFTW.MPI/3.3.10-gompi-2022a
#        ScaLAPACK/2.2.0-gompi-2022a-fb foss/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0 GDRCopy/2.3-GCCcore-11.3.0 UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
#        NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 bzip2/1.0.8-GCCcore-11.3.0 libreadline/8.1.2-GCCcore-11.3.0 Tcl/8.6.12-GCCcore-11.3.0
#        SQLite/3.38.3-GCCcore-11.3.0 GMP/6.2.1-GCCcore-11.3.0 libffi/3.4.2-GCCcore-11.3.0 Python/3.10.4-GCCcore-11.3.0 pybind11/2.9.2-GCCcore-11.3.0
#        SciPy-bundle/2022.05-foss-2022a Szip/2.1.1-GCCcore-11.3.0 HDF5/1.12.2-gompi-2022a h5py/3.7.0-foss-2022a cURL/7.83.0-GCCcore-11.3.0
#        dill/0.3.6-GCCcore-11.3.0 double-conversion/3.2.0-GCCcore-11.3.0 flatbuffers/2.0.7-GCCcore-11.3.0 giflib/5.2.1-GCCcore-11.3.0 ICU/71.1-GCCcore-11.3.0
#        JsonCpp/1.9.5-GCCcore-11.3.0 NASM/2.15.05-GCCcore-11.3.0 libjpeg-turbo/2.1.3-GCCcore-11.3.0 LMDB/0.9.29-GCCcore-11.3.0 nsync/1.25.0-GCCcore-11.3.0
#        protobuf/3.19.4-GCCcore-11.3.0 protobuf-python/3.19.4-GCCcore-11.3.0 libpng/1.6.37-GCCcore-11.3.0 snappy/1.1.9-GCCcore-11.3.0 networkx/2.8.4-foss-2022a
hpc project
which python #check if module loaded
/app1/ebapps/arches/flat-avx2/software/Python/3.10.4-GCCcore-11.3.0/bin/python
#    python3 -c 'import tensorflow as tf,sys;print(tf.__version__)'
#    2025-10-23 18:04:29.117000: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F
#    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#    2.11.0
module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
python3 -c 'import tensorflow as tf,sys,numpy as np,scipy as scpy;print(tf.__version__ + np.__version__ + scpy.__version__)'
#    2025-10-23 18:24:47.584568: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F
#    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#    2.11.01.22.31.8.1
#create virtual env
VENV=$HOME/.venvs/finsight-tf211-cuda117
python -m venv "$VENV"
source "$VENV/bin/activate"
(finsight-tf211-cuda117) $ python -m pip install --upgrade pip wheel
#    Requirement already satisfied: pip in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (22.0.4)
python3 - <<'PY'
import sys, tensorflow as tf, pandas as pd, plotly, sklearn,numpy as np,scipy as scpy
print("TF", tf.__version__, "Python", sys.version.split()[0])
print(np.__version__, " ; ", scpy.__version__, " ; ", pd.__version__)
PY
#    2025-10-23 18:41:58.844788: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F
#    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#    TF 2.11.0 Python 3.10.4
#    1.22.3  ;  1.8.1  ;  1.4.2
vim $HOME/pbs/finsight_tf_smoke.pbs #check other file in this repo
qsub -v MODE=sanity_gpu $HOME/pbs/finsight_tf_smoke.pbs
305019.stdct-mgmt-02
qcat -j $JOBID -t OU
qstat -ans

#    stdct-mgmt-02:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    305019.stdct-m* user batch_g* sam_finsi* 813094   1  36  240gb 01:00 R 00:00
#       GN-A40-078/1*36
#       Job run at Thu Oct 23 at 18:51 on (GN-A40-078[1]:ncpus=36:mem=25165824...

tail -n 200 -f /scratch/$USER/$JOBID/train.finsight.$JOBID.log
ll /scratch/$USER
#    total 32
#    drwx------ 2 user svuusers 0 Oct 23 18:51 305019.stdct-mgmt-02
echo $JOBID
#    305019.stdct-mgmt-02

#to kill job
qdel $JOBID
vim $HOME/pbs/finsight_tf_smoke.pbs
mv ~/projects-sam/FinSight-QuantLab/src/sam_finsight_train.o305019 ./
cat sam_finsight_train.o305019
#    + TF_MODULE=TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#    + VENV=/home/svu/user/.venvs/finsight-tf211-cuda117
#...

vim $HOME/pbs/finsight_tf_smoke.pbs
JOBID=$(qsub -v MODE=sanity_gpu $HOME/pbs/finsight_tf_smoke.pbs)
qstat -ans
#
#    stdct-mgmt-02:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    305037.stdct-m* user batch_g* sam_finsi*    --    1  36  240gb 01:00 R   --
#       GN-A40-078/1*36
#       Job run at Thu Oct 23 at 19:26 on (GN-A40-078[1]:ncpus=36:mem=25165824...
qdel 305038.stdct-mgmt-02
qstat
#    Job id            Name             User              Time Use S Queue
#    ----------------  ---------------- ----------------  -------- - -----
#    305037.stdct-mgm* sam_finsight_tr* user                 0 R batch_gpu
#    305038.stdct-mgm* sam_finsight_tr* user          00:00:08 R batch_gpu

qstat -ans1
#    stdct-mgmt-02:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    305037.stdct-m* user batch_g* sam_finsi*    --    1  36  240gb 01:00 R   --  GN-A40-078/1*36
#       Job was sent for execution at Thu Oct 23 at 19:28 on (GN-A40-078[1]:nc...
#    305038.stdct-m* user batch_g* sam_finsi* 12642*   1  36  240gb 01:00 E 00:00 GN-A40-099/0*36
#       Job run at Thu Oct 23 at 19:26 on (GN-A40-099[0]:ncpus=36:mem=25165824...
qdel $JOBID
#    qdel: Job has finished 305038.std ct-mgmt-02
qstat
#    Job id            Name             User              Time Use S Queue
#    ----------------  ---------------- ----------------  -------- - -----
#    305037.stdct-mgm* sam_finsight_tr* user                 0 H batch_gpu
tail -f /scratch/$USER/305038.stdct-mgmt-02/pbs.live.305038.stdct-mgmt-02.out

vim $HOME/projects-sam/FinSight-QuantLab/src/train_tf_smoke.py
source $HOME/.venvs/finsight-tf211-cuda117/bin/activate
(finsight-tf211-cuda117) [user@vanda ~]$ module purge
#    Unloading shared
#      WARNING: Did not unuse /cm/shared/modulefiles
(finsight-tf211-cuda117) [user@vanda ~]$ module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#    Loading TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#      Loading requirement: GCCcore/11.3.0 zlib/1.2.12-GCCcore-11.3.0 binutils/2.38-GCCcore-11.3.0 GCC/11.3.0 ncurses/6.3-GCCcore-11.3.0 numactl/2.0.14-GCCcore-11.3.0 XZ/5.2.5-GCCcore-11.3.0
#        libxml2/2.9.13-GCCcore-11.3.0 libpciaccess/0.16-GCCcore-11.3.0 hwloc/2.7.1-GCCcore-11.3.0 OpenSSL/1.1 libevent/2.1.12-GCCcore-11.3.0 libfabric/1.15.1-GCCcore-11.3.0 PMIx/4.1.2-GCCcore-11.3.0
#        UCX/1.12.1-GCCcore-11.3.0 UCC/1.0.0-GCCcore-11.3.0 OpenMPI/4.1.4-GCC-11.3.0 OpenBLAS/0.3.20-GCC-11.3.0 FlexiBLAS/3.2.0-GCC-11.3.0 FFTW/3.3.10-GCC-11.3.0 gompi/2022a FFTW.MPI/3.3.10-gompi-2022a
#        ScaLAPACK/2.2.0-gompi-2022a-fb foss/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0 GDRCopy/2.3-GCCcore-11.3.0 UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
#        bzip2/1.0.8-GCCcore-11.3.0 libreadline/8.1.2-GCCcore-11.3.0 Tcl/8.6.12-GCCcore-11.3.0 SQLite/3.38.3-GCCcore-11.3.0 GMP/6.2.1-GCCcore-11.3.0 libffi/3.4.2-GCCcore-11.3.0 Python/3.10.4-GCCcore-11.3.0
#        pybind11/2.9.2-GCCcore-11.3.0 SciPy-bundle/2022.05-foss-2022a Szip/2.1.1-GCCcore-11.3.0 HDF5/1.12.2-gompi-2022a h5py/3.7.0-foss-2022a cURL/7.83.0-GCCcore-11.3.0 dill/0.3.6-GCCcore-11.3.0
#        double-conversion/3.2.0-GCCcore-11.3.0 flatbuffers/2.0.7-GCCcore-11.3.0 giflib/5.2.1-GCCcore-11.3.0 ICU/71.1-GCCcore-11.3.0 JsonCpp/1.9.5-GCCcore-11.3.0 NASM/2.15.05-GCCcore-11.3.0
#        libjpeg-turbo/2.1.3-GCCcore-11.3.0 LMDB/0.9.29-GCCcore-11.3.0 nsync/1.25.0-GCCcore-11.3.0 protobuf/3.19.4-GCCcore-11.3.0 protobuf-python/3.19.4-GCCcore-11.3.0 libpng/1.6.37-GCCcore-11.3.0
#        snappy/1.1.9-GCCcore-11.3.0 networkx/2.8.4-foss-2022a
(finsight-tf211-cuda117) [user@vanda ~]$ source $HOME/.venvs/finsight-tf211-cuda117/bin/activate
(finsight-tf211-cuda117) [user@vanda ~]$ pip install "vllm>=0.5" transformers accelerate
#    ...
#    Successfully installed aiofiles-25.1.0 babel-2.17.0 backoff-2.2.1 beautifulsoup4-4.14.2 courlan-1.3.2 cryptography-46.0.3 dataclasses-json-0.6.7 dateparser-1.2.2 emoji-2.15.0 faiss-cpu-1.12.0 filetype-1.2.0 html5lib-1.1 htmldate-1.9.3 justext-3.0.2 langdetect-1.0.9 lxml-5.4.0 lxml_html_clean-0.4.3 marshmallow-3.26.1 mypy-extensions-1.1.0 nltk-3.9.2 numpy-1.24.4 olefile-0.47 pypdf-6.1.3 python-iso639-2025.2.18 python-magic-0.4.27 python-oxmsg-0.0.2 rapidfuzz-3.14.1 requests-toolbelt-1.0.0 scipy-1.15.3 sentence-transformers-5.1.2 soupsieve-2.8 tld-0.13.1 trafilatura-2.0.0 typing-inspect-0.9.0 tzlocal-5.3.1 unstructured-0.18.15 unstructured-client-0.42.3
cd projects-sam/FinSight-QuantLab/
git pull
git branch
git checkout -b feature/rag-sam
#    Switched to a new branch 'feature/rag-sam'
mkdir ragllm
cd ragllm/
vim rag_server.py
vim rag_requirements.txt
cd ~/pbs/
cp finsight_tf_smoke.pbs finsight_rag.pbs
ll
#    total 160
#    -rw-r--r-- 1 user svuusers   4380 Oct 24 11:18 finsight_rag.pbs
#    -rw-r--r-- 1 user svuusers   4380 Oct 23 21:08 finsight_tf_smoke.pbs
#    -rw------- 1 user svuusers 223375 Oct 23 18:57 sam_finsight_train.o305019

# START THE JOB
(finsight-tf211-cuda117) [user@vanda pbs]$ JOBID=$(qsub $HOME/pbs/finsight_rag.pbs)
(finsight-tf211-cuda117) [user@vanda pbs]$ ll
total 184
-rw-r--r-- 1 user svuusers   5566 Oct 24 12:01 finsight_rag.pbs
-rw-r--r-- 1 user svuusers   4380 Oct 23 21:08 finsight_tf_smoke.pbs
-rw------- 1 user svuusers 223375 Oct 23 18:57 sam_finsight_train.o305019
-rw------- 1 user svuusers   2424 Oct 24 12:12 sam_rag_train.o305894
-rw------- 1 user svuusers 222442 Oct 24 12:32 sam_rag_train.o305920

(finsight-tf211-cuda117) [user@vanda pbs]$ cat sam_rag_train.o305894
# ...
(finsight-tf211-cuda117) [user@vanda pbs]$ tail -f /scratch/$USER/$JOBID/pbs.live.$JOBID.out

(finsight-tf211-cuda117) [user@vanda pbs]$ less sam_rag_train.o305920


#add ssh key so can monitor and open ports on compute node from login node
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tunneltocomputenode -N ""
cat ~/.ssh/id_ed25519_tunneltocomputenode.pub >> ~/.ssh/authorized_keys
cat  ~/.ssh/authorized_keys
#    ecdsa-sha2-nistp256 xxx= user@stdct-login-01
#    ssh-ed25519 xxx user@vanda.nus.edu.sg
#    ssh-ed25519 xxx user@stdct-login-01

# now login to hugging face
[user@vanda pbs]$ module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
pip install -U huggingface_hub
#    Requirement already satisfied: huggingface_hub in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (0.36.0)
#    Requirement already satisfied: filelock in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (3.20.0)
#    Requirement already satisfied: fsspec>=2023.5.0 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (2025.9.0)
#    Requirement already satisfied: packaging>=20.9 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (25.0)
#    Requirement already satisfied: pyyaml>=5.1 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (6.0.3)
#    Requirement already satisfied: requests in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (2.32.5)
#    Requirement already satisfied: tqdm>=4.42.1 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (4.67.1)
#    Requirement already satisfied: typing-extensions>=3.7.4.3 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (4.15.0)
#    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from huggingface_hub) (1.1.10)
#    Requirement already satisfied: charset_normalizer<4,>=2 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from requests->huggingface_hub) (3.4.4)
#    Requirement already satisfied: idna<4,>=2.5 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from requests->huggingface_hub) (3.11)
#    Requirement already satisfied: urllib3<3,>=1.21.1 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from requests->huggingface_hub) (2.5.0)
#    Requirement already satisfied: certifi>=2017.4.17 in /nfs/home/svu/user/.venvs/finsight-tf211-cuda117/lib/python3.10/site-packages (from requests->huggingface_hub) (2025.10.5)
hf auth login
          _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
          _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
          _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
          _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
          _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

          To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
      Enter your token (input will not be visible):
      Add token as git credential? (Y/n)
      Token is valid (permission: fineGrained).
      The token `xyztokenname` has been saved to /home/svu/user/.cache/huggingface/stored_tokens
      Your token has been saved in your configured git credential helpers (store).
      Your token has been saved to /home/svu/user/.cache/huggingface/token
      Login successful.
      The current active token is: `xyztokenname`
# RESTART THE JOB
JOBID=$(qsub $HOME/pbs/finsight_rag.pbs)
qstat -ans

#    stdct-mgmt-02:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    310866.stdct-m* user batch_g* sam_rag_t* 37796*   1  36  240gb 12:00 R 00:00
#       GN-A40-076/0*36
#       Job run at Fri Oct 24 at 19:46 on (GN-A40-076[0]:ncpus=36:mem=25165824...

ll /scratch/$USER/$JOBID/
#    total 112
#    -rw------- 1 user svuusers 218708 Oct 24 19:46 pbs.live.310866.stdct-mgmt-02.out
tail -f /scratch/$USER/$JOBID/pbs.live.$JOBID.out
vim finsight_rag.pbs
JOBID=$(qsub $HOME/pbs/finsight_rag.pbs)
(finsight-tf211-cuda117) [user@vanda pbs]$ qstat -ans

#    stdct-mgmt-02:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    310911.stdct-m* user batch_g* sam_rag_t* 17735*   1  36  240gb 12:00 R 00:00
#       GN-A40-099/1*36
#       Job run at Fri Oct 24 at 19:54 on (GN-A40-099[1]:ncpus=36:mem=25165824...
tail -f /scratch/$USER/$JOBID/pbs.live.$JOBID.out

#    + HF_HOME=/scratch/user/hf
#    + mkdir -p /scratch/user/hf/hub
#    + export HUGGING_FACE_HUB_TOKEN=hf_xxxx
#    + HUGGING_FACE_HUB_TOKEN=hf_xxxx
#    + export VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
#    + VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
#    + export VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
#    + VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
#    + echo 'vLLM: 127.0.0.1:8000  |  RAG API: 127.0.0.1:8080'
#    vLLM: 127.0.0.1:8000  |  RAG API: 127.0.0.1:8080
#    + echo 'VLLM_URL: http://127.0.0.1:8000/v1/chat/completions | VLLM_MODEL: meta-llama/Meta-Llama-3.1-8B-Instruct'
#    VLLM_URL: http://127.0.0.1:8000/v1/chat/completions | VLLM_MODEL: meta-llama/Meta-Llama-3.1-8B-Instruct
#    + python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000 --download-dir /scratch/user/hf/hub --gpu-memory-utilization 0.92 --max-model-len 8192
#    + SSH_OPTS='-i /home/svu/user/.ssh/id_ed25519_tunneltocomputenode -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new'
#    + uvicorn rag_server:app --app-dir ./ --host 127.0.0.1 --port 8080
#    + wait
#    + ssh -i /home/svu/user/.ssh/id_ed25519_tunneltocomputenode -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -N -R 28080:127.0.0.1:8080 user@vanda.nus.edu.sg

#FINALLY, THE SERVER HAS STARTER!
# IT IS RUNNING, NOW OPEN THE TUNNEL FROM LOGIN NODE


#some more commands
less +G sam_rag_train.o305920 #to start frmo bottom
ss -ltnp | egrep ':8000|:8080' #check if ports are listening
pbsnodes GN-A40-078 #check node status
qstat -ans1 305960 #check specific job
ls ~/.cache/huggingface/
#    stored_tokens  token
ls ~/.cache/huggingface/token
#    /home/svu/e1538626/.cache/huggingface/token
cat ~/.cache/huggingface/token
#    hf_xxxx
cat /scratch/user/310911.stdct-mgmt-02/vllm.log #more logs




















