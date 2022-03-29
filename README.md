# cppRL
Coverage path planning with Reinforcement Learning. Using ROS, controlling a TurtleBot3 with a UV lamp to cover an area with a minimum UV energy to kill coronavirus.
![Alt text](img/uv_lamp_sim.png?raw=true "sim")

## Setup and Instalation 
ROS is required for the simulation. Requires UVC lamp from ... 

I used a docker container for Tensorflow and Baselines. Using the Baselines zoo CUDA image, I installed the required ROS packages with rospypi, so that the ROS installation is not needed in the container.
I launched my container with `docker run --runtime=nvidia --env="DISPLAY=${localEnv:DISPLAY}" --volume="/tmp/.X11-unix:/tmp/.X11-unix" --net=host -v /home/havard/cppRL:/usr/share/cppRL -it stablebaselines/rl-baselines-zoo`

In addition, the UVC lamp workspace needs to be sourced in the docker container `source devel/setup.sh`.  

