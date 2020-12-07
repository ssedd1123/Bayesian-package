if [ "$#" -lt 1 ]; then
  echo "To use, enter" $0 "<number of cores> <option result file>"
  exit 0
fi

CORES=$1
CORES=$(( CORES+1 ))
pwd | cd
export OMP_NUM_THREADS="1"
mpiexec -n ${CORES} python -m GUI.GUIController.GUIController ${@: 2}
