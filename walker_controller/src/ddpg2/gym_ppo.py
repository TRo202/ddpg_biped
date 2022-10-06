# setting up the tf environment for testing the biped with PPO
# this file will later be converted to tensor-compatible env via TFPyEnvironment wrapper

import numpy as np
import csv
import tensorflow as tf
import rospy
import time

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetModelConfiguration

from std_msgs.msg import Float64
from std_srvs.srv import Empty


reward_file = "reward_file.csv"
trajectory_file = "trajectory_file.csv"


BATCH_SIZE = 64
EPISODES = 500000
EVAL_INTERVAL = 1000
NUM_EVAL_EPISODES = 50
TEST = 10


pubHipR = rospy.Publisher(
    '/waist_thighR_position_controller/command', Float64, queue_size=10)
pubHipL = rospy.Publisher(
    '/waist_thighL_position_controller/command', Float64, queue_size=10)
pubKneeR = rospy.Publisher(
    '/thighR_shankR_position_controller/command', Float64, queue_size=10)
pubKneeL = rospy.Publisher(
    '/thighL_shankL_position_controller/command', Float64, queue_size=10)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy(
    '/gazebo/set_model_configuration', SetModelConfiguration)

unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

rospy.init_node('walker_control_script')
rate = rospy.Rate(50)


class RobotState(object):
    def __init__(self):
        self.waist_z = 0.0
        self.waist_y = 0.0
        self.outer_ring_inner_ring_theta = 0.0
        self.hipr_theta = 0.0
        self.hipr_theta_dot = 0.0
        self.hipl_theta = 0.0
        self.hipl_theta_dot = 0.0
        self.kneer_theta = 0.0
        self.kneer_theta_dot = 0.0
        self.kneel_theta = 0.0
        self.kneel_theta_dot = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.footr_contact = 0
        self.footl_contact = 0
        self.robot_state = [self.vel_y, self.vel_z, self.hipr_theta, self.hipr_theta_dot, self.hipl_theta, self.hipl_theta_dot,
                            self.kneer_theta, self.kneer_theta_dot, self.kneel_theta, self.kneel_theta_dot, self.footr_contact, self.footl_contact]

        self.latest_reward = 0.0
        self.best_reward = -100000000000000.0
        self.episode = 0
        self.last_outer_ring_inner_ring_theta = 0.0
        self.last_time = 0.0

        self.fall = 0
        self.done = False
        self.count_of_1 = 0
        self.avg_reward = 0.0


robot_state = RobotState()


class BipedPPoEnv(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        #self._state = robot_state.robot_state
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float64, minimum=-1.5, maximum=1.5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 1, 12), dtype=np.float64, name='observation')
        # self._episode_ended = False

    # def batched(self):
    #     return True    # batched => True

    def batch_size(self):
        return BATCH_SIZE

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        # ['waist_thighR', 'waist_thighL', 'thighR_shankR', 'thighL_shankL', 'outer_ring_inner_ring', 'inner_ring_boom', 'boom_waist']
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_simulation()
        except rospy.ServiceException as e:
            print("reset_world failed!")

        rospy.wait_for_service('gazebo/set_model_configuration')

        try:
            reset_joints("walker", "robot_description",
                         ['boom_waist', 'outer_ring_inner_ring', 'thighL_shankL', 'thighR_shankR', 'waist_thighL',
                          'waist_thighR'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            robot_state.last_outer_ring_inner_ring_theta = 0.0
        except rospy.ServiceException as e:
            print("reset_joints failed!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pause()
        except rospy.ServiceException as e:
            print("rospause failed!'")

        set_robot_state()
        observation = get_observation()
        ts.restart(observation)

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')

        try:
            unpause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        take_action = action.numpy()  # "action" input => Tensor matching the `action_spec`

        pubHipR.publish(action[0])
        pubKneeR.publish(action[1])
        pubHipL.publish(action[2])
        pubKneeL.publish(action[3])

        observation = get_observation()
        reward = -0.1  # when it used to run, used to be -0.1
        current_time = time.time()
        if (
                robot_state.outer_ring_inner_ring_theta - robot_state.last_outer_ring_inner_ring_theta) <= -0.09:  # -0.001forward motion

            delta_time = current_time - robot_state.last_time

            reward += -(robot_state.outer_ring_inner_ring_theta -
                        robot_state.last_outer_ring_inner_ring_theta) * 10
            rate.sleep()
            return ts.transition(observation, reward, discount=0.999)

        if robot_state.waist_z < -0.10:
            reward += -100
            robot_state.done = True
            robot_state.fall = 1
            self.reset()
            rate.sleep()
            return ts.termination(observation, reward, discount=0.999)

        if robot_state.outer_ring_inner_ring_theta < -9.0:
            reward += 100
            robot_state.done = True
            robot_state.fall = 1
            print("REACHED TO THE END!")
            self.reset()
            rate.sleep()
            return ts.termination(observation, reward, discount=0.999)

        robot_state.last_time = current_time
        robot_state.last_outer_ring_inner_ring_theta = robot_state.outer_ring_inner_ring_theta
        # rate.sleep()
        # return reward, robot_state.done


def get_observation():
    observation = np.zeros([1, 1, 12], dtype=np.float64)
    # states_data = np.array([robot_state.robot_state])
    # states_data = states_data.T
    # observation = states_data
    observation[0, 0:] = robot_state.robot_state

    return observation


def callbackJointStates(data):
    # ['boom_waist', 'outer_ring_inner_ring', 'thighL_shankL', 'thighR_shankR', 'waist_thighL', 'waist_thighR']
    # if vel == 0 ['waist_thighR', 'waist_thighL', 'thighR_shankR', 'thighL_shankL', 'outer_ring_inner_ring', 'boom_waist']
    robot_state.data = data

    if len(data.velocity) != 0:

        robot_state.vel_z = data.velocity[0]
        robot_state.vel_y = data.velocity[1]
        robot_state.kneel_theta_dot = data.velocity[2]
        robot_state.kneer_theta_dot = data.velocity[3]
        robot_state.hipl_theta_dot = data.velocity[4]
        robot_state.hipr_theta_dot = data.velocity[5]

        robot_state.waist_z = data.position[0]
        robot_state.waist_y = data.position[1]
        robot_state.outer_ring_inner_ring_theta = data.position[1]
        robot_state.kneel_theta = data.position[2]
        robot_state.kneer_theta = data.position[3]
        robot_state.hipl_theta = data.position[4]
        robot_state.hipr_theta = data.position[5]
    else:
        robot_state.vel_z = 0
        robot_state.vel_y = 0
        robot_state.kneel_theta_dot = 0
        robot_state.kneer_theta_dot = 0
        robot_state.hipl_theta_dot = 0
        robot_state.hipr_theta_dot = 0

        robot_state.waist_z = 0
        robot_state.waist_y = 0
        robot_state.outer_ring_inner_ring_theta = 0
        robot_state.kneel_theta = 0
        robot_state.kneer_theta = 0
        robot_state.hipl_theta = 0
        robot_state.hipr_theta = 0

    set_robot_state()
    # rate.sleep()


def callbackContactShankR(data):
    if not data.states:
        robot_state.footr_contact = 0
    else:
        robot_state.footr_contact = 1


def callbackContactShankL(data):
    if not data.states:
        robot_state.footl_contact = 0
    else:
        robot_state.footl_contact = 1


def set_robot_state():
    robot_state.robot_state = [robot_state.vel_y, robot_state.vel_z, robot_state.hipr_theta, robot_state.hipr_theta_dot,
                               robot_state.hipl_theta, robot_state.hipl_theta_dot, robot_state.kneer_theta,
                               robot_state.kneer_theta_dot, robot_state.kneel_theta, robot_state.kneel_theta_dot,
                               robot_state.footr_contact, robot_state.footl_contact]


env = BipedPPoEnv()


"""--------------This is to validate whether the environment is working by testing 5 episodes. ------------------"""
"""
utils.validate_py_environment(environment, episodes=5)
"""
"""--------------------------------------------------------------------------------------------------------------"""


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        state = policy.get_initial_state(environment.batch_size)
        episode_return = 0.0

        while not time_step.is_last():
            policy_step = policy.action(time_step, state)
            state = policy_step.state
            time_step = environment.step(policy_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def create_networks(tf_env):
    actor_net = ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        input_fc_layer_params=(128,),
        lstm_size=(128, 128),
        dtype=tf.float64,
        output_fc_layer_params=(128,),
        activation_fn=None)
    value_net = ValueRnnNetwork(
        tf_env.observation_spec(),
        input_fc_layer_params=(128,),
        lstm_size=(128, 128),
        output_fc_layer_params=(128,),
        activation_fn=None)

    return actor_net, value_net


def train_agent(EPISODES, save_each=100, print_each=1000):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    # writing rewards in the csv file
    # file = open(reward_file, 'wt')
    # writer = csv.writer(file)
    # writer.writerow(['avg_reward'])

    for episode in range(EPISODES):
        step = agent.train_step_counter.numpy()
        current_metrics = []

        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)

        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if step % print_each == 0:
            print("\nIteration: {}, loss:{:.2f}".format(
                step, train_loss.loss.numpy()))

            for i in range(len(train_metrics)):
                print('{}: {}'.format(
                    train_metrics[i].name, train_metrics[i].result().numpy()))

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(
                eval_tf_env, agent.policy, NUM_EVAL_EPISODES)
            print(f'Step = {step}, Average Return = {avg_return}')
            returns.append((step, avg_return))

        if step % save_each == 0:
            policy_save_handler.save("policies/policy_ppo")


if __name__ == '__main__':
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(env)

    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    actor_net, value_net = create_networks(tf_env)

    agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer=optimizer,  # need to define
        actor_net=actor_net,  # need to define
        value_net=value_net,  # need to define
        num_epochs=10,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.2,
        use_gae=True,
        use_td_lambda_return=True
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000
    )
    replay_buffer_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.AverageReturnMetric(batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size)
    ]

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer]+train_metrics,
        num_episodes=10
    )

    # Creates and returns a dataset that returns entries from the buffer.
    dataset = replay_buffer.as_dataset(
        sample_batch_size=BATCH_SIZE, num_steps=2, num_parallel_calls=3).prefetch(3)

    # Wrapper for tf.function with TF Agents-specific customizations.
    agent.train = common.function(agent.train)

    all_train_loss = []
    all_metrics = []
    returns = []

    # checkpoint_dir = "checkpoints/checkpoint_ppo"
    # train_checkpointer = common.Checkpointer(
    #     ckpt_dir=checkpoint_dir,
    #     max_to_keep=1,
    #     agent=agent,
    #     policy=agent.policy,
    #     replay_buffer=replay_buffer,
    #     global_step=train_step
    # )
    # train_checkpointer.initialize_or_restore()
    # train_step = tf.compat.v1.train.get_global_step()
    policy_save_handler = policy_saver.PolicySaver(agent.policy)

    # training here
    train_agent(EPISODES)

    rospy.Subscriber("/joint_states", JointState, callbackJointStates)
    rospy.Subscriber("/footR_contact_sensor_state",
                     ContactsState, callbackContactShankR)
    rospy.Subscriber("/footL_contact_sensor_state",
                     ContactsState, callbackContactShankL)

    # save at end in every case

    policy_save_handler.save("policies/policy_ppo")
