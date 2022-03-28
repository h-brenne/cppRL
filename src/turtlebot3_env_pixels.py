import gym
import rospy
import numpy as np
import math
import time
import os
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from uvc_lamp.srv import Reset_uv
from matplotlib import pyplot as plt

from gym import spaces
from gym.utils import seeding
from tf.transformations import euler_from_quaternion

class TurtleBot3EnvPixels(gym.Env):

    def __init__(self, 
            goal_list=None,
            max_env_size=None,
            continuous=False,
            use_angle_vel=False,
            use_pixels_space=True,
            only_reward_end = True,
            reduce_coverage_state = False,
            lidar_size=24, #24 lidar 
            action_size=4, 
            min_range = 0.015,
            max_range = 3.5,
            min_ang_vel = -1.5,
            max_ang_vel = 1.5,
            min_cmd_ang = -math.pi,
            max_cmd_ang = math.pi,
            const_linear_vel = 0.15,
            collision_distance = 0.165,
            reward_goal=0,
            reward_covered=1,
            reward_collision=-100,
            robot_pos = 0,
            covered=0,
            covered_indicies = None,
            grids_to_cover=0,
            percentage_coverage_goal=0.6,
            covered_previous=0,
            covered_twice = 0,
            covered_twice_previous=0,
            map_width = 0,
            map_heigth = 0,
            pos_map = None,
            maps = None
        ):
        
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goal = False
        self.use_angle_vel = use_angle_vel
        self.use_pixels_space = use_pixels_space
        self.reduce_coverage_state = reduce_coverage_state
        self.only_reward_end = only_reward_end
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.pub_cmd_ang = rospy.Publisher('cmd_ang', Float32, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        
       
        #Coverage path planning observation
        self.robot_pos = robot_pos
        self.covered = covered
        self.covered_indicies = covered_indicies
        self.grids_to_cover = grids_to_cover
        self.percentage_coverage_goal = percentage_coverage_goal
        self.covered_previous = covered_previous
        self.covered_twice = covered_twice
        self.covered_twice_previous = covered_twice_previous
        self.reward_covered=reward_covered
        self.coverage_percentage = 0
        self.overlap_percentage = 0
        
        self.sub_pos_map = rospy.Subscriber("maps", Float32MultiArray, self.getMaps)
        self.pos_map = pos_map
        self.maps = maps
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)


        self.lidar_size = lidar_size
        self.const_linear_vel = const_linear_vel
        self.min_range = min_range
        self.max_range = max_range
        self.min_ang_vel = min_ang_vel
        self.max_ang_vel = max_ang_vel
        self.min_cmd_ang = min_cmd_ang
        self.max_cmd_ang = max_cmd_ang
        self.collision_distance = collision_distance
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.continuous = continuous
        self.max_env_size = max_env_size
        self.map_width = map_width
        self.map_heigth = map_heigth


        if self.continuous:
            low, high, shape_value = self.get_action_space_values()
            self.action_space = spaces.Box(low=low, high=high, shape=(shape_value,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(action_size)
            if self.use_angle_vel:
                ang_step = max_ang_vel/((action_size - 1)/2)
            else:
                ang_step = 2/action_size
            self.actions = [-1+action * ang_step for action in range(action_size)]

        while(self.map_heigth==0):
            time.sleep(0.5)
        if use_pixels_space:
            low, high = self.get_pixel_observation_space_values()
            self.observation_space = spaces.Box(low, high, dtype=np.uint8)
        else:
            low, high = self.get_observation_space_values()
            self.observation_space = spaces.Box(low, high, dtype=np.uint8)

        self.num_timesteps = 0
        self.lidar_distances = None
        self.ang_vel = 0
        self.cmd_ang = 0

        self.start_time = time.time()
        self.last_step_time = self.start_time

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action_space_values(self):
        if self.use_angle_vel:
            low = self.min_ang_vel
            high = self.max_ang_vel
        else:
            low = -1
            high = 1
        shape_value = 1

        return low, high, shape_value

    def get_pixel_observation_space_values(self):
        low = np.full((self.map_width, self.map_heigth, 3), 0)
        high = np.full((self.map_width, self.map_heigth, 3), 255)
        """if self.reduce_coverage_state:
            high = np.append(high, np.full((self.map_width, self.map_heigth, 1), 255), axis=2)
        else:
            high = np.append(high, np.full((self.map_width, self.map_heigth, 1), 255), axis=2)"""
        return low, high

    def get_observation_space_values(self):
        low = np.append(np.full(self.lidar_size, self.min_range), 0)
        low = np.append(low, np.full((self.map_width, self.map_heigth, 1), 0).flatten())
        high = np.append(np.full(self.lidar_size, self.max_range), self.map_heigth*self.map_width)
        high = np.append(high, np.full((self.map_width, self.map_heigth, 1), 1).flatten())
        #low = np.full(self.lidar_size, self.min_range)
        #high = np.full(self.lidar_size, self.max_range)
        return low, high

    def _getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)


        heading = yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading

    def getMaps(self, maps):
        #Data format:
        #multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]

        self.map_width = maps.layout.dim[0].size
        self.map_heigth = maps.layout.dim[1].size

        #Put data in np array. Should be possible to do more efficient with numpy
        self.maps =np.ndarray((maps.layout.dim[0].size,maps.layout.dim[1].size,maps.layout.dim[2].size))
        for i in range(maps.layout.dim[0].size):
            for j in range(maps.layout.dim[1].size):
                for k in range(maps.layout.dim[2].size):
                    self.maps[i,j,k] = maps.data[maps.layout.dim[1].stride*i + maps.layout.dim[2].stride*j + k]
        #Save space when saving in buffer
        self.maps = self.maps.astype(np.uint8)

        #Set covered and overlapped cells
        covered = self.maps[:,:,1]

        #Scale to 255
        self.maps[:,:,1] = covered*8

        occupied_grids = (self.maps[:,:,0]>1).sum()
        self.grids_to_cover = covered.size - occupied_grids
        self.covered = (covered > 5).sum()
        self.covered_twice = (covered > 30).sum()
        
        #Reduce to binary
        if self.reduce_coverage_state:
            self.maps[:,:,1] = (covered > 10)*255
        
        if not self.use_pixels_space:
                robot_index = np.where(self.maps == 255)
                self.robot_pos = int(robot_index[0]+robot_index[1] * self.map_width)
                covered_indicies = np.where(covered > 10)
                self.covered_indicies = covered_indicies[0]+covered_indicies[1]*self.map_width
                
        #Debug
        #print(self.robot_pos)
        #np.savetxt("maps0.csv", self.maps[:,:,0])
        #np.savetxt("maps1.csv", self.maps[:,:,1])
        #plt.imsave('maps0.png', self.maps[:,:,0], vmin=0,vmax=255)
        #plt.imsave('maps1.png', self.maps[:,:,1], vmin=0,vmax=255)
        #plt.imsave('maps2.png', self.maps[:,:,2], vmin=0,vmax=255)
    def get_time_info(self):
        time_info = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        time_info += '-' + str(self.num_timesteps)
        return time_info

    def episode_finished(self):
        pass


    def getState(self, scan):
        scan_range = []
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(self.max_range)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(self.min_range)
            else:
                scan_range.append(scan.ranges[i])

        self.lidar_distances = scan_range

        if min(self.lidar_distances) < self.collision_distance:
            print('Collision!! Covered grids: ', self.covered)
            done = True
        if self.covered>self.grids_to_cover*self.percentage_coverage_goal:
            print('Goal!! Covered grids: ', self.covered)
            self.get_goal = True
            done = True
        if self.use_pixels_space:
            return self.maps, done
        else:
            obs = self.lidar_distances + [self.robot_pos]
            obs = np.append(obs, self.maps[:,:,1].flatten())
            return obs, done

    def setReward(self, done):
                
        if self.get_goal:
            reward = self.reward_covered*(self.covered - 0.4*self.covered_twice)
            reward = self.reward_goal
            self.get_goal = False
            self.pub_cmd_vel.publish(Twist())
        elif done:
            reward = self.reward_collision + self.reward_covered*(self.covered - 0.4*self.covered_twice)
            self.pub_cmd_vel.publish(Twist())
        else:
            if not self.only_reward_end:
                reward = self.reward_covered*((self.covered-self.covered_previous) - 0.4*(self.covered_twice-self.covered_twice_previous))
                self.covered_previous = self.covered
                self.covered_twice_previous = self.covered_twice
            else:
                reward = 0
        #print("Reward:", reward)
        return reward

    def set_ang_vel(self, action):
        if self.use_angle_vel:
            if self.continuous:
                self.ang_vel = action
            else:
                self.ang_vel = self.actions[action]
        else:
            if self.continuous:
                self.cmd_ang = math.pi*action
            else:
                self.cmd_ang = math.pi*self.actions[action]

    def step(self, action):

        self.set_ang_vel(action)
        if self.use_angle_vel:
            vel_cmd = Twist()
            vel_cmd.linear.x = self.const_linear_vel
            vel_cmd.angular.z = self.ang_vel
            self.pub_cmd_vel.publish(vel_cmd)
        else:
            self.pub_cmd_ang.publish(self.cmd_ang)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(done)
        self.num_timesteps += 1

        return np.asarray(state), reward, done, {}


    def reset(self):
        self.coverage_percentage = 100*self.covered/self.grids_to_cover
        self.overlap_percentage = 100*self.covered_twice/self.grids_to_cover
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("gazebo/reset_simulation service call failed")
 
        uv_reset = False
        while(not uv_reset):
            try:
                rospy.wait_for_service('reset_uv', 10.0)
                reset_uv = rospy.ServiceProxy('reset_uv', Empty)
                reset_uv()
                self.covered_previous=0
                self.covered_twice_previous=0
                uv_reset = True
                break
            except (rospy.exceptions.ROSException, rospy.service.ServiceException):
                print("reset_uv service call failed, unpausing and retrying")
                try:
                    rospy.wait_for_service('gazebo/unpause_physics', 2.0)
                    print("unpausing")
                    self.unpause_proxy()
                    rospy.wait_for_service('reset_uv', 2.0)
                    reset_uv = rospy.ServiceProxy('reset_uv', Empty)
                    reset_uv()
                    rospy.wait_for_service('gazebo/pause_physics', 2.0)
                    self.pause_proxy()
                except (rospy.exceptions.ROSException, rospy.service.ServiceException):
                    print("reset_uv or gazebo physics service call failing, retrying")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        state, _ = self.getState(data)

        return np.asarray(state)


    def render(self, mode=None):
        pass


    def close(self):
        self.reset()
