# Intrinsic Motivation using an Inverse Dynamics Model
This repository builds on a PyTorch implementation of the Soft Actor Critic found [here](https://github.com/denisyarats/pytorch_sac) by using the prediction error of an inverse dynamics model as a measure of intrinsic reward.

## Requirements
Install [MuJoCo](http://www.mujoco.org/).

Install these libraries
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install these dependencies
```sh
conda env create -f conda_env.yml
conda activate pytorch_sac
```

## Instructions
To train ICM on Walker Walk run
```
python train.py agent=icm task=walker_walk
```

## Monitoring
Logs are stored in the `exp_local` folder. T
The console output is also available in a form:
```
| train | F: 6000 | S: 6000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```

# Intrinsic Rewards
Intrinsic rewards can be scaled using the ``` icm_scale ``` hyperparameter in ```agents/icm.yaml```.
