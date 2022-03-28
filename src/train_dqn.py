import gym
import rospy
import tensorflow as tf

from turtlebot3_env_pixels import TurtleBot3EnvPixels

from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from stable_baselines.common.callbacks import BaseCallback

from gym.envs.registration import register

from customcnn import custom_cnn

register(
    id='TurtleBot3_Pixels-v0',
    entry_point=TurtleBot3EnvPixels
)


policy_kwargs = dict(cnn_extractor=custom_cnn)
env_name = 'TurtleBot3_Pixels-v0'

rospy.init_node(env_name.replace('-', '_'))

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

model = DQN(CnnPolicy, env, prioritized_replay=True, policy_kwargs=policy_kwargs,
buffer_size=100000, learning_rate=0.0003, target_network_update_freq=8000, learning_starts=1000, 
exploration_fraction=0.2, exploration_final_eps=0, verbose=1, tensorboard_log="./deepq_custom_tutlebot_tensorboard/")
#model.load(env_name+"_custom_dqn")
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)
        self.coverage_percentage = 0
        self.overlap_percentage = 0

    def _on_rollout_end(self) -> bool:
        if(self.training_env.get_attr("coverage_percentage")[0] != self.coverage_percentage 
        or self.training_env.get_attr("overlap_percentage")[0] != self.overlap_percentage):
            self.coverage_percentage = self.training_env.get_attr("coverage_percentage")[0]
            self.overlap_percentage = self.training_env.get_attr("overlap_percentage")[0]

            summary = tf.Summary(value=[tf.Summary.Value(tag='coverage_percentage', simple_value=self.coverage_percentage)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            summary = tf.Summary(value=[tf.Summary.Value(tag='overlap_percentage', simple_value=self.overlap_percentage)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


model.learn(total_timesteps=int(3e5), log_interval=100, callback=TensorboardCallback())
#model.save(env_name+"_custom_dqn")
