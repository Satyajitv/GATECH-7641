Code:
=====
github url: https://github.com/Satyajitv/GATECH-7641/tree/main/assignment4

Steps to setup the conda env:
=============================
1. conda env create -f requirements.yml
2. conda activate test-env
3. login into the env and run below two command
    conda install -c conda-forge gym
    pip install pymdptoolbox

Run analysis:
==============
git clone https://github.com/Satyajitv/GATECH-7641.git
cd ./assignment4/
Run above "Steps to setup the conda env"
python frozen_lake_MDP.py # for frozen lake analysis
python forest_management_analysis.py #for forest management analysis
