# HydraFusion
Code for our paper titled _**"HydraFusion: Context-Aware Selective Sensor Fusion for Robust and Efficient Autonomous Vehicle Perception"**_

This repository contains the algorithmic implementation of our HydraFusion model. 
Our model is intended to be used with the RADIATE dataset available here: https://pro.hw.ac.uk/radiate/

**hydranet.py** -- contains the class HydraFusion, which defines our top-level model specification.

**stem.py** -- defines the stem modules in HydraFusion

**branch.py** -- defines the branches implemented in our model.

**gate.py** -- contains the gating module implementations.

**fusion.py** -- contains the definition of the fusion block along with the algorithms to fuse the bounding boxes output by each active branch.


The stems and branches are built using a split architecture implementation of Faster R-CNN with a ResNet-18 backbone.
HydraFusion can be used with any image-based multi-modal dataset. In our evaluations we used two cameras, one radar sensor, and one lidar sensor as inputs to the model.
