# Reinforcement Learning for Bipedal walking robot.
<p align="justify">Previously, this repository contains the <b>simulation architecture</b> based in <b>Gazebo</b> environment for implementing reinforcement learning algorithm, <b>DDPG</b> for generating bipedal walking patterns for the robot. </p>
<p align="justify"> But here, I am trying to implement <b><a href="https://arxiv.org/pdf/1707.06347.pdf">PPO algorithm</a></b> with the help of <b><a href="https://www.tensorflow.org/agents">Tensorflow Agents</a></b>. </p>

## Planar Bipedal walking robot in Gazebo environment using Proximal Policy Optimization (PPO).
<p align="justify"> Still working on... </p>


## What you need before starting (Dependencies & Packages):
- <b><a href="http://releases.ubuntu.com/18.04/">Ubuntu 18.04</a></b>
- <b><a href="http://wiki.ros.org/kinetic">ROS Melodic</a></b>
- <b><a href="http://gazebosim.org/">Gazebo 7</a></b>
- <b><a href="https://www.tensorflow.org/">TensorFlow: 2 </a></b>
- <b><a href="https://www.tensorflow.org/">Tensor-probability: 0.8.0</a></b>
- <b><a href="https://www.tensorflow.org/">TF-Agents: 0.3.0</a></b>
- <b><a href="https://gym.openai.com/docs/">gym: 0.9.3</a></b>
- <b>Python 3.6.9</b>

## File setup:
- ***walker_gazebo*** contains the robot model(both **.stl** files & **.urdf** file) and also the gazebo **launch** file.

- ***walker_controller*** contains the reinforcement learning implementation of **DDPG algorithm** for control of the bipedal walking robot.

## Learning to walk, initial baby steps (DDPG)
<p align= "center">
  <img src="walker_controller/src/training_1.gif/" height="250" width="400" hspace="5">
  <img src="walker_controller/src/training_2.gif/" height="250" width="400">
</p>

## Stable bipedal walking (DDPG)
<p align= "center">
  <img src="walker_controller/src/trained.gif/" height="300" width="550">
</p>

<strong>[<a href="https://goo.gl/1hwqJy*">Project video</a>]</strong>

**Note:** A stable bipedal walking was acheived after training the model using a <strong>Nvidia GeForce GTX 1050 Ti GPU</strong> enabled system for over 41 hours. The visualization for the horizontal boom(attached to the waist) is turned off.

## Sources:
<ol>
  <li>Lillicrap, Timothy P., et al.<b><a href="https://arxiv.org/abs/1509.02971"> Continuous control with deep reinforcement learning.</a></b> arXiv preprint arXiv:1509.02971 (2015).</li>
<li>Silver, David, et al.<b><a href="http://proceedings.mlr.press/v32/silver14.pdf"> Deterministic Policy Gradient Algorithms.</a></b> ICML (2014).</li>
</ol>

## Project Collaborator(s):
**<a href="https://github.com/ioarun">Arun Kumar</a>** (arunkumar12@iisc.ac.in) & **<a href="http://www.aero.iisc.ernet.in/people/s-n-omkar/">Dr. S N Omkar</a>** (omkar@iisc.ac.in)

## Future work
<p align= "justify">Implement state of the art RL algorithms(<b>TRPO & PPO</b>) for the same. Hopefully lead to faster training and less convergence time.</p>
