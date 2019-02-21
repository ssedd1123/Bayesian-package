module load GCC/8.2.0-2.31.1
module load OpenMPI/4.0.0
module unload python
export PATH="/mnt/home/tsangchu/anaconda2/bin:$PATH"
source activate Bay3

pip show mpi4py 1>/dev/null #pip for Python 2
if [ $? == 0 ]; then
   echo "Installed" #Replace with your actions
else
    MPICC_DIR=$(which mpicc)
    git clone https://github.com/mpi4py/mpi4py.git ./mpi4py.git
    cd mpi4py.git
    python setup.py build --mpicc=${MPICC_DIR}
    python setup.py install
    cd ../
    rm -rf mpi4py.git
fi
