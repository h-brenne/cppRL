# cppRL
Coverage path planning with Reinforcement Learning. Using ROS, controlling a TurtleBot3 with a UV lamp to cover an area with a minimum UV energy to kill coronavirus. Gym environment implementation were based on the implementation from [gym-turtlebot3](https://github.com/ITTcs/gym-turtlebot3)
![Alt text](img/uvc_lamp_sim.png?raw=true "sim")

## Setup and Instalation 
ROS is required for the simulation. Requires ROS-UV-Cleaning-Robot, can be pulled as a submodule with `git submodule update --init --recursive` 

I used a docker container for Tensorflow and Baselines. Using the Baselines zoo CUDA image, I installed the required ROS packages with rospypi, so that the ROS installation is not needed in the container.
I launched my container with `docker run --runtime=nvidia --env="DISPLAY=${localEnv:DISPLAY}" --volume="/tmp/.X11-unix:/tmp/.X11-unix" --net=host -v /home/$USER/cppRL:/usr/share/cppRL -it stablebaselines/rl-baselines-zoo` This exposes the network adapter for ROS communication, adds a shared folder where the project is, as well as a display.

In addition, the UVC lamp workspace needs to be sourced in the docker container `source ROS-UV-Cleaning-Robot/devel/setup.sh`. 

To launch, a gazebo simulator is needed, a pre-made map must be published with map_server, and the uvc_lamp node needs to be launched. If angle control(instead of angle rate control) is used, the angle_controller in ROS-UV-Cleaning-Robot needs to be run.

Two maps have been tested with the RL agent, small-room and small-house.

### From turtleRL_ws, after building and sourcing:

Launch Gazebo with a map: `roslaunch turtlebot3_gazebo turtlebot3_small_room.launch`

Launch map_server with the same map: `rosrun map_server map_server maps/small_room_40cm.yaml`


### From ROS-UV-Cleaning-Robot, after building and sourcing:

Launch uv_lamp: `roslaunch uvc_lamp uvc_grid_map.launch`

Launch angle_controller: `rosrun angle_contoller angle_controller.py`

