import cv2
import gym
import numpy as np
import os
import sys
import retro
from gym.wrappers import Monitor
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Convert observation to grayscale
SELECT_BUTTON_INDEX = 2
START_BUTTON_INDEX = 3

class PeriodicSaver(Callback):
    def __init__(self, save_freq, file_name="dqn_weights.h5f"):
        super(PeriodicSaver, self).__init__()
        self.save_freq = save_freq  # e.g., save the weights every save_freq episodes
        self.file_name = file_name
        self.episode = 0

    def on_episode_end(self, episode, logs={}):
        self.episode += 1
        if self.episode % self.save_freq == 0:
            self.model.save_weights(self.file_name, overwrite=True)


class rgb_to_gray(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, _oldc) = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(oldh, oldw, 1),
                                                dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:, :, None]

class RetroActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        # Convert discrete action into button array
        button_array = np.zeros(self.env.action_space.n)
        button_array[action] = 1
        return button_array
    
class ButtonOverlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ButtonOverlayWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        overlayed_observation = self._overlay_button_presses(observation, action)
        return overlayed_observation, reward, done, info

    def _overlay_button_presses(self, observation, action):
        button_names = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']  # Adjust based on your game
        active_button = button_names[action]
        
        # Use cv2 or another imaging library to overlay the text on the observation
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(observation, active_button, (10, 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return observation
    
class ReplayScreenHandler(gym.Wrapper):
    def __init__(self, env):
        self.template = cv2.imread(os.path.join(SCRIPT_DIR, 'replay_template.png'), cv2.IMREAD_GRAYSCALE)
        super().__init__(env)

    def step(self, action):
        # Avoid START and SELECT buttons
        if action in [SELECT_BUTTON_INDEX, START_BUTTON_INDEX]:
            action = 0

        observation, reward, done, info = self.env.step(action)
        is_replay = is_replay_screen(observation, self.template)
        if is_replay:
            not_replay_times = 0
            while not_replay_times < 10:
                is_replay = is_replay_screen(observation, self.template)
                if not is_replay:
                    not_replay_times += 1
                else:
                    observation, reward, done, info = self.env.step(START_BUTTON_INDEX)
                    not_replay_times = 0

        return observation, reward, done, info

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
    ), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2, 
                   batch_size=32, train_interval=4)
    return dqn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def is_replay_screen(observation, template):
    # Apply binary thresholding
    _, thresh = cv2.threshold(observation, 200, 255, cv2.THRESH_BINARY) # Adjust the threshold value accordingly

    # Use cv2.matchTemplate to find the template in the thresholded image
    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Define a threshold for match quality (experiment to find a good value)
    threshold = 0.8

    if max_val > threshold:
        return True
    else:
        return False

def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    env = retro.make("InternationalSuperstarSoccerDeluxe-Snes",
                     inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.ALL)
    env = rgb_to_gray(env)
    env = RetroActionWrapper(env)
    env = ReplayScreenHandler(env)  # Include ReplayScreenHandler during training
    env = ButtonOverlayWrapper(env)  # Include ButtonOverlayWrapper during training
    env = Monitor(env, './video_training', force=True, video_callable=lambda episode_id: True)
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    # dqn.load_weights('dqn_weights.h5f')
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    saver_callback = PeriodicSaver(save_freq=1)  # This will save weights every episode
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1, callbacks=[saver_callback])

    # Closing the environment for testing
    env.close()
    env = retro.make("InternationalSuperstarSoccerDeluxe-Snes", 
                     inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.ALL)
    env = rgb_to_gray(env)
    env = RetroActionWrapper(env)
    env = ReplayScreenHandler(env)  # Apply ReplayScreenHandler during testing
    env = Monitor(env, './video_testing', force=True, video_callable=lambda episode_id: True)  # Separate folder for testing

    dqn.test(env, nb_episodes=5, visualize=False)


if __name__ == "__main__":
    main()
