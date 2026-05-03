#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# shellcheck disable=SC1091
source /home/nlpir/miniconda3/etc/profile.d/conda.sh
conda activate base

conda create -n MemGUI python=3.11.0 -y
echo "Created conda environment MemGUI with Python 3.11.0."

conda activate MemGUI
pip install -r requirements.txt
echo "Installed packages in MemGUI from requirements.txt."


### android_world ###
name="android_world"
python_version="3.11.8"
conda create -n $name python=$python_version -y
echo "Created conda environment ${name} with Python ${python_version}."
conda activate $name
cd ./framework/android_env
python setup.py install
echo "Successfully run setup.py for AndroidEnv"
cd ../../
pip install -r ./requirements/${name}.txt
pip install protobuf==5.29.0
cd ./framework/models/AndroidWorld
python setup.py install
echo "Successfully run setup.py for AndroidWorld"
cd ../../../
echo "Installed packages in ${name}."
conda deactivate


# echo "Script completed successfully."
