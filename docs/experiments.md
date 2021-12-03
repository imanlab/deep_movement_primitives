# Experiments

This is a comprehensive list and description of all experiments.

- [Experiments](#experiments)
  - [How to run](#how-to-run)
    - [Examples](#examples)
  - [Autoencoder](#autoencoder)
  - [RTP-RGB Experiments](#rtp-rgb-experiments)
    - [Experiment 01 - RTP-RGB FC](#experiment-01---rtp-rgb-fc)
    - [Experiment 02 - RTP-RGB CNN](#experiment-02---rtp-rgb-cnn)
  - [RTP-RGBD Experiments](#rtp-rgbd-experiments)
    - [Experiment 03 - RTP-RGBD FC](#experiment-03---rtp-rgbd-fc)
    - [Experiment 04 - RTP-RGBD CNN](#experiment-04---rtp-rgbd-cnn)
    - [Experiment 05 - RTP-RGBD CNN-RES](#experiment-05---rtp-rgbd-cnn-res)
    - [Experiment 06 - RTP-RGBD POINTNET](#experiment-06---rtp-rgbd-pointnet)
    - [Experiment 07 - RTP-RGBD DMP](#experiment-07---rtp-rgbd-dmp)
  - [WPP-RGB Experiments](#wpp-rgb-experiments)
    - [Experiment 08 - WPP-RGB CNN](#experiment-08---wpp-rgb-cnn)
    - [Experiment 09 - WPP-RGB CNN-RES](#experiment-09---wpp-rgb-cnn-res)
    - [Experiment 10 - WPP-RGB DMP](#experiment-10---wpp-rgb-dmp)
    - [Experiment 11-20 - WPP-RGB CNN-ABLATION](#experiment-11-20---wpp-rgb-cnn-ablation)

## How to run

Experiments can simply be executed by running the corresponding Python module from the root folder of the repository.

See the help text for more details:

```
python -m experiments.e04_rtp-rgbd_cnn.experiment --help
```

```
positional arguments:
  command               Command. Must be one of:
                          run:   execute both training and evaluation
                          train: execute the training
                          eval:  execute the evaluation

optional arguments:
  -h, --help            show this help message and exit
  -n [NAME], --name [NAME]
                        Name of the model to run/train/save.
                        Defaults to the current date and time for 'run' and 'train', while it's required by 'eval'.
  -m [MULTI], --multi [MULTI]
                        Number of times this experiment will run, providing average and standard deviation values for each metric in the end.
                        If not specified the experiment will run only one time and no aggreagate metrics will be computed.
                        If specified without a value, it will default to 10 times.
                        This argument only affects the 'run' command.
```

The output of each experiment will be stored in that experiment's folder.

Each experiment can be fully configured by tweaking the parameters and settings in the `config.py` file present in that experiment's directory.

Note that all RTP and WPP experiments require access to a pretrained saved autoencoder model. We already provide pretrained autoencoder models for the three datasets `rtp-rgb`, `rtp-rgbd` and `wpp-rgb` in `/experiments/e00_autoencoder/models`. 

### Examples

These are some examples of commands to run experiments in various modes.

Train and evaluate experiment 01 with name "my_exp":
```
python -m experiments.e01_rtp-rgb_fc.experiment run -n my_exp
```

Train experiment 03 with name "some_name", but do not evaluate it:
```
python -m experiments.e03_rtp-rgbd_fc.experiment train -n some_name
```

Evaluate the previously trained experiment 03 with name "some_name":
```
python -m experiments.e03_rtp-rgbd_fc.experiment eval -n some_name
```

Train and evaluate experiment 08 with name "multiple_run" 10 times.  
Metrics from each run will be saved individually, as well as mean and standard deviation of each metric over the 10 runs.
```
python -m experiments.e08_wpp-rgb_cnn.experiment run -n multiple_run -m 10
```

## Autoencoder

A CNN autoencoder trained on the RGB images taken from the home position.

Pretrained autoencoder models for the three datasets `rtp-rgb`, `rtp-rgbd` and `wpp-rgb` are avialable in `/experiments/e00_autoencoder/models`. All RTP and WPP experiments are already cnfigured to use those models. 

If you want to train your own autoencoder change the contents of config.py to your use case. If you then want to use the model you trained in any experiment, change the `AUTOENCODER_MODEL_PATH` variable in that experiment configuration file. 

## RTP-RGB Experiments

These are the experiments performed on the mockup dataset RTP-RGB.

### Experiment 01 - RTP-RGB FC

FC model predicting full ProMP weights of RTP using the RTP-RGB dataset.

Task: RTP

Dataset: RTP-RGB (mock)

Input: RGB images from home position

Output: full ProMP weights of RTP trajectory

Model: RGB image -> Encoder -> bottleneck image -> Flatten -> FC -> ProMP weights

Optimizer: Adam

Training loss: RMSE on trajectory in joint space.

Metrics: 
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each region has its own test set, evaluated separately.

### Experiment 02 - RTP-RGB CNN

CNN model predicting full ProMP weights of RTP using the RTP-RGB dataset.

Task: RTP

Dataset: RTP-RGB (mock)

Input: RGB images from home position

Output: full ProMP weights of RTP trajectory

Model: RGB image -> Encoder -> bottleneck image -> CNN -> ProMP weights

Optimizer: Adam

Training loss: RMSE on trajectory in joint space

Metrics:
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each region has its own test set, evaluated separately.

## RTP-RGBD Experiments

These are the experiments performed on the RTP-RGBD dataset.

### Experiment 03 - RTP-RGBD FC

FC model predicting full ProMP weights of RTP using the RTP-RGBD dataset.

Task: RTP

Dataset: RTP-RGBD

Input: RGB images from home position

Output: full ProMP weights of RTP trajectory

Model: RGB image -> Encoder -> bottleneck image -> Flatten -> FC -> ProMP weights

Optimizer: Adam

Training loss: RMSE on trajectory in joint space.

Metrics: 
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each region has its own test set, evaluated separately.

### Experiment 04 - RTP-RGBD CNN

CNN model predicting full ProMP weights of RTP using the RTP-RGBD dataset.

Task: RTP

Dataset: RTP-RGBD

Input: RGB images from home position

Output: full ProMP weights of RTP trajectory

Model: RGB image -> Encoder -> bottleneck image -> CNN -> ProMP weights

Optimizer: Adam

Training loss: RMSE on trajectory in joint space

Metrics:
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each region has its own test set, evaluated separately.

### Experiment 05 - RTP-RGBD CNN-RES

CNN model predicting residual ProMP weights of RTP using the RTP-RGBD dataset, averages computed for each region.

Task: RTP

Dataset: RTP-RGBD

Input: RGB images from home position + average ProMP weights

Output: full ProMP weights of RTP trajectory

Model: (RGB image -> Encoder -> bottleneck image -> CNN -> residual ProMP weights) + average ProMP weights -> full ProMP weights

Optimizer: Adam

Training loss: RMSE on trajectory in joint space

Metrics:
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each region has its own test set, evaluated separately.

### Experiment 06 - RTP-RGBD POINTNET

TODO

### Experiment 07 - RTP-RGBD DMP

TODO

## WPP-RGB Experiments

These are the experiments performed on the WPP-RGB dataset.

### Experiment 08 - WPP-RGB CNN

CNN model predicting full ProMP weights of WPP using the WPP-RGB dataset, using joint space trajectories.

Task: WPP

Dataset: WPP-RGB

Input: RGB images from home position with a circle on the target position + pixel coordinates of the target point

Output: full ProMP weights of WPP trajectory in joint space

Model: (RGB image -> Encoder -> bottleneck image -> CNN -> feature vector) + target coordinates -> dense network -> full ProMP weights

Optimizer: Adam

Training loss: RMSE on joint trajectories.

Metrics:
- RMSE on trajectory in joint space.
- RMSE on the cartesian trajectory of the EE.
- RMSE on the orientation angle of the EE.
- Euclidean distance on final trajectory point in cartesian space.

Each configuration has its own test set, evaluated separately.

### Experiment 09 - WPP-RGB CNN-RES

TODO

### Experiment 10 - WPP-RGB DMP

TODO

### Experiment 11-20 - WPP-RGB CNN-ABLATION

These experiments are identical to experiment 08 (WPP CNN), but each one uses a different train-test data split, to understand the generalization over different palpation paths.

**Experiment 11**

Training paths: 1, 2, 3, 6, 7
Testing paths: 4, 5

**Experiment 12**

Training paths: 1, 2, 5, 6, 7
Testing paths: 3, 4

**Experiment 13**

Training paths: 1, 2, 5
Testing paths: 3, 4
Excluded paths: 6, 7

**Experiment 14**

Training paths: 1, 4, 5
Testing paths: 2, 3
Excluded paths: 6, 7

**Experiment 15**

Training paths: 1, 2, 3, 6, 7
Mixed training and testing paths: 4, 5

**Experiment 16**

Training paths: 1, 2, 5, 6, 7
Mixed training and testing paths: 3, 4

**Experiment 17**

Training paths: 1, 2, 5
Mixed training and testing paths: 3, 4
Excluded paths: 6, 7

**Experiment 18**

Training paths: 1, 4, 5
Mixed training and testing paths: 2, 3
Excluded paths: 6, 7

**Experiment 19**

Mixed training and testing paths: 1, 2, 3, 4, 5, 6, 7

**Experiment 20**

Mixed training and testing paths: 1, 2, 3, 4, 5
Excluded paths: 6, 7