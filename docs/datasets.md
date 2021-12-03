# Datasets

These are the descriptions of the datasets collected.

Contents:
- [Datasets](#datasets)
  - [RTP-RGB (mock dataset)](#rtp-rgb-mock-dataset)
  - [RTP-RGBD](#rtp-rgbd)
  - [WPP-RGB](#wpp-rgb)

## RTP-RGB (mock dataset)

The first dataset to be collected only included RGB images from the home position (no depth data) and was used for a mock study.  
The study showed a possible correlation between performance and density of the distribution of the nipple position on the xy plane. This was taken into account in the design of the RTP-RGBD dataset.

Structure of the dataset. Each sample is identified by a unique progressive 3 digit number (examples: 001, 263, 431...).
```
rtp-rgb
├── color_img
|   ├── 001.png
|   ├── ...
│   └── ID.png
│           PNG images taken at the home position with random breast phantom poses. Shape (480, 640).
└── trajectories
    ├── 001.json
    ├── ...
    └── ID.json
            JSON files containing the robot state at each step of the trajectory.
            Most importantly, the joint coordinates and the end effector task space coordinates. Shape (N_samples, 7).
```

## RTP-RGBD

This dataset was used for the actual study of the RTP task. Its structure is similar to RTP-RGB, but includes depth data alongside the RGB images.

The plane was divided in 4 regions (A, B, C, D) and the samples were collected so that the final point of the trajectories follow a uniform distribution with different density in each region.

Structure of the dataset. Each sample is identified by a unique combination of a letter (identifying the region) and a progressive 3 digit number (examples: A_001, C_372, B_372...).
```
rtp-rgbd
├── bad_samples
|       Imperfect samples, excluded from the actual dataset (e.g. corrupted image, wrong trajectory)
├── color_img
|   ├── A_001.png
|   ├── ...
│   └── ID.png
│           PNG images taken at the home position with random breast phantom poses. Shape (480, 640).
├── depth
|   ├── A_001.npy
|   ├── ...
│   └── ID.npy
│           Numpy files containing the depth data of the image taken at the home position. Shape (480, 640).
├── pointclouds
|   ├── A_001.npy
|   ├── ...
│   └── ID.npy
│           Numpy files containing the XYZ RGB pointclouds taken at the home position. Shape (N_points, 6).
├── trajectories
|   ├── A_001.json
|   ├── ...
|   └── ID.json
|           JSON files containing the robot state at each step of the trajectory.
|           Most importantly, the 7 joint coordinates. Shape (N_samples, 7).
└── trajectories_task
    ├── A_001.json
    ├── ...
    └── ID.json
            JSON files containing the end effector pose at each step of the trajectory.
            Pose is a vector of length 7: (x, y, z) position vector concatenated (a, b, c, d) orientation quaternion. Shape (N_samples, 7).
```

## WPP-RGB

This dataset was used for the study of the WPP task. The general structure is similar to RTP-RGBD, but the data follows a slightly different layout.

The breast phantom was placed in 4 different configurations (1, 2, 3, 4). On each configuration 7 Wedged Palpation Paths were defined (1 through 7), as it can be seen from the image. For each path of each configuration 30 samples were recorded, for a total of 30 * 7 * 4 = 840 samples. Each sample contains the image from the home position, with the target point annotated with a circle, as well as all the data captured during the demonstration (joint states and tactile sensor readings).

Structure of the dataset. Each sample is identified by a unique combination of a first digit (identifying the configuration), a second digit (identifying the palpation path) and 3 final progressive digits (examples: 1_1_010, 3_7_372, 2_1_010...).

```
wpp-rgb
├── img_base
|   ├── 1.jpg
|   ├── 2.jpg
|   ├── 3.jpg
│   └── 4.jpg
│           JPG images taken at the home position in the 4 breast configurations, without annotations. Shape (480, 640).
├── img_base_resized
|   ├── 1.png
|   ├── 2.png
|   ├── 3.png
│   └── 4.png
│           PNG images taken at the home position in the 4 breast configurations, without annotations. Reshaped to (256, 256).
├── img_resized
|   ├── 1_1_000.png
|   ├── ...
│   └── ID.png
│           PNG images taken from the home positions, annotated with a circle on the target point. Shape (256, 256).
└── trajectories
    ├── 1_1_000.json
    ├── ...
    └── ID.json
            JSON files containing the robot state at each step of the trajectory, plus some general data.
            Most importantly:
             - the 7 joint coordinates. Shape (N_samples, 7).
             - the target point in pixel coordinates on the resized image. Shape (2, ).
```
