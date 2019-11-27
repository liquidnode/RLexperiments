#
#this file contains modified code from https://github.com/minerllabs/baselines
#

"""
MIT License

Copyright (c) 2019 MineRL Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from collections import OrderedDict
import copy
from logging import getLogger
import time

import gym
import numpy as np
import cv2
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder

import torch
from functools import reduce

cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)


class ResetTrimInfoWrapper(gym.Wrapper):
    """Take first return value.

    minerl's `env.reset()` returns tuple of `(obs, info)`
    but existing agent implementations expect `reset()` returns `obs` only.
    """
    def reset(self, **kwargs):
        #obs, info = self.env.reset(**kwargs)
        #return obs
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            return ret[0]
        else:
            return ret


class ContinuingTimeLimitMonitor(Monitor):
    """`Monitor` with ChainerRL's `ContinuingTimeLimit` support.

    Because of the original implementation's design,
    explicit `close()` is needed to save the last episode.
    Do not forget to call `close()` at the last line of your script.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        if self.env_semantics_autoreset:
            raise gym.error.Error(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env comes from deprecated OpenAI Universe.")
        ret = super()._start(directory=directory,
                             video_callable=video_callable, force=force,
                             resume=resume, write_upon_reset=write_upon_reset,
                             uid=uid, mode=mode)
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id
        self.stats_recorder = _ContinuingTimeLimitStatsRecorder(
            directory,
            '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
            autoreset=False, env_id=env_id)
        return ret


class _ContinuingTimeLimitStatsRecorder(StatsRecorder):
    """`StatsRecorder` with ChainerRL's `ContinuingTimeLimit` support.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
    """

    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super().__init__(directory, file_prefix,
                         autoreset=autoreset, env_id=env_id)
        self._save_completed = True

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            logger.debug('Tried to reset env which is not done. '
                         'StatsRecorder completes the last episode.')
            self.save_complete()

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_step(self, observation, reward, done, info):
        self._save_completed = False
        return super().after_step(observation, reward, done, info)

    def save_complete(self):
        if not self._save_completed:
            super().save_complete()
            self._save_completed = True

    def close(self):
        self.save_complete()
        super().close()


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class PoVWithCompassAngleWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+compass (or K+compass for gray-scaled image) channels.
    """
    def __init__(self, env):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later

        pov_space = self.env.observation_space.spaces['pov']
        compass_angle_space = self.env.observation_space.spaces['compassAngle']

        low = self.observation({'pov': pov_space.low, 'compassAngle': compass_angle_space.low})
        high = self.observation({'pov': pov_space.high, 'compassAngle': compass_angle_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        pov = observation['pov']
        compass_scaled = observation['compassAngle'] / self._compass_angle_scale
        compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
        return np.concatenate([pov, compass_channel], axis=-1)


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination

        low = self.observation(self.observation_space.low)
        high = self.observation(self.observation_space.high)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, frame):
        return np.moveaxis(frame, self.source, self.destination)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # sanity checks
        ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        if original_space != ideal_image_space:
            raise ValueError('Image space should be {}, but given {}.'.format(ideal_image_space, original_space))
        if original_space.dtype != np.uint8:
            raise ValueError('Image should `np.uint8` typed, but given {}.'.format(original_space.dtype))

        height, width = original_space.shape[0], original_space.shape[1]
        new_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        if self._key is None:
            self.observation_space = new_space
        else:
            self.observation_space.spaces[self._key] = new_space

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class SerialDiscreteActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.

    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.

    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != \
                len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack' , 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        logger.info('always ignored keys: {}'.format(self.exclude_keys))

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.noop)
                if key in self.reverse_keys:
                    op[key] = 0
                else:
                    op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action

class CombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = ICombineActionWrapper(env)
        self.action_space = self.wrapper.action_space

    def action(self, action):
        return self.wrapper.action(action)

class ICombineActionWrapper():#gym.ActionWrapper):
    """Combine MineRL env's "exclusive" actions.

    "exclusive" actions will be combined as:
        - "forward", "back" -> noop/forward/back (Discrete(3))
        - "left", "right" -> noop/left/right (Discrete(3))
        - "sneak", "sprint" -> noop/sneak/sprint (Discrete(3))
        - "attack", "place", "equip", "craft", "nearbyCraft", "nearbySmelt"
            -> noop/attack/place/equip/craft/nearbyCraft/nearbySmelt (Discrete(n))
    The combined action's names will be concatenation of originals, i.e.,
    "forward_back", "left_right", "snaek_sprint", "attack_place_equip_craft_nearbyCraft_nearbySmelt".
    """
    def __init__(self, env):
        #super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            new_actions = [noop]

            for key in valid_action_keys:
                space = self.wrapping_action_space.spaces[key]
                for i in range(1, space.n):
                    op = copy.deepcopy(noop)
                    op[key] = i
                    new_actions.append(op)
            return new_key, new_actions

        self._maps = {}
        self.combine_tuples = (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'))
        self.combine_list = []
        for keys in self.combine_tuples:
            for k in keys:
                self.combine_list.append(k)
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))
        for k, v in self._maps.items():
            logger.info('{} -> {}'.format(k, v))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        logger.debug('action {} -> original action {}'.format(action, original_space_action))
        return original_space_action

    def reverse_action(self, original_space_action):
        action = copy.deepcopy(self.noop)

        for keys in self.combine_tuples:
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            comb_action = {k: original_space_action[k] for k in valid_action_keys}
            #check if comb_action is in self._maps[new_key]
            def and_red(x1, x2):
                return x1 and x2
            ex_in_maps = [reduce(and_red, [comb_action[s] == m[s] for s in m]) for m in self._maps[new_key]]
            def or_red(x1, x2):
                return x1 or x2
            ex_in_maps = reduce(or_red, ex_in_maps)
            #prevent invalid actions
            if not ex_in_maps:
                #invalid action detected
                #if sprint and sneak is active => sneak
                #if forwards and backwards is active => none
                #if left and right is active => none
                if new_key == 'sneak_sprint':
                    op = {a: 0 for a in valid_action_keys}
                    op['sneak'] = 1
                    original_space_action.update(op)
                elif new_key == 'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                    if original_space_action['attack'] == 0:
                        print('')
                        print('')
                        print('WARNING can not resolve invalid action!!')
                        print({a: original_space_action[a] for a in valid_action_keys})
                        print('')
                        print('')
                    op = {a: 0 for a in valid_action_keys}
                    #add first non zero action
                    #exception: if attack is active ignore it
                    original_space_action['attack'] = 0
                    for k in valid_action_keys:
                        if original_space_action[k] != 0:
                            op[k] = original_space_action[k]
                            break
                    original_space_action.update(op)
                else:
                    noop = {a: 0 for a in valid_action_keys}
                    original_space_action.update(noop)
                comb_action = {k: original_space_action[k] for k in valid_action_keys}

            #convert to combine action space
            a_index = 0
            for i in range(len(self._maps[new_key])):
                if reduce(and_red, [comb_action[s] == self._maps[new_key][i][s] for s in self._maps[new_key][i]]):
                    a_index = i
                    break
            action[new_key] = a_index

        for k in original_space_action:
            if not k in self.combine_list:
                action[k] = original_space_action[k]

        return action


class SerialDiscreteCombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action

class Orientation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = IOrientation(env)
        self.observation_space = self.wrapper.observation_space
        self.action_space = self.wrapper.action_space
        
    def step(self, action):
        return self.wrapper.step(action)

    def reset(self):
        return self.wrapper.reset()

class IOrientation():
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.observation_space.spaces['orientation'] = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self.orientation = 0.0

    def add_to_obs(self, obs):
        obs['orientation'] = np.array([self.orientation / 90.0])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._step(obs, action)
        return obs, reward, done, info

    def _step(self, obs, action):
        self.orientation += action['camera'][0]
        self.orientation = max(self.orientation, -90.0)
        self.orientation = min(self.orientation, 90.0)
        obs = self.add_to_obs(obs)
        return obs

    def reset(self):
        obs = self.env.reset()
        obs = self._reset(obs)
        return obs

    def _reset(self, obs):
        self.orientation = 0.0
        obs = self.add_to_obs(obs)
        return obs

class LastAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = ILastAction(env)
        self.observation_space = self.wrapper.observation_space
        self.action_space = self.wrapper.action_space
        
    def step(self, action):
        return self.wrapper.step(action)

    def reset(self):
        return self.wrapper.reset()

class ILastAction():
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.last_action = {}
        for k in self.action_space.spaces:
            if isinstance(self.observation_space, dict):
                self.observation_space['last_'+k] = self.action_space.spaces[k]
            else:
                self.observation_space.spaces['last_'+k] = self.action_space.spaces[k]
            if isinstance(self.action_space.spaces[k], gym.spaces.Box):
                self.last_action[k] = np.zeros(self.action_space.spaces[k].shape, dtype=np.float32)
            else:
                self.last_action[k] = 0

    def add_to_obs(self, obs):
        for k in self.action_space.spaces:
            obs['last_'+k] = self.last_action[k]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._step(obs, action)
        return obs, reward, done, info

    def _step(self, obs, action):
        for k in self.action_space.spaces:
            self.last_action[k] = action[k]
        obs = self.add_to_obs(obs)
        return obs

    def reset(self):
        obs = self.env.reset()
        obs = self._reset(obs)
        return obs

    def _reset(self, obs):
        for k in self.action_space.spaces:
            if isinstance(self.action_space.spaces[k], gym.spaces.Box):
                self.last_action[k] = np.zeros(self.action_space.spaces[k].shape, dtype=np.float32)
            else:
                self.last_action[k] = 0
        obs = self.add_to_obs(obs)
        return obs



class HistoryActionReward(gym.Wrapper):
    def __init__(self, env, num_history, args):
        super().__init__(env)
        self.wrapper = IHistoryActionReward(env, num_history, args)
        self.observation_space = self.wrapper.observation_space
        self.action_space = self.wrapper.action_space
        
    def step(self, action):
        return self.wrapper.step(action)

    def reset(self):
        return self.wrapper.reset()

class IHistoryActionReward():
    def __init__(self, env, num_history, args):
        self.env = env
        try:
            self.num_history = args.history_a
        except:
            self.num_history = 0
        self.num_history_action = num_history
        self.action_space = self.env.action_space
        action_vec_len = 0
        for k in self.action_space.spaces:
            if k == 'camera':
                action_vec_len += 2
            else:
                action_vec_len += self.action_space.spaces[k].n
        self.action_vec_len = action_vec_len
        self.observation_space = copy.deepcopy(self.env.observation_space)
        if self.num_history > 0:
            self.observation_space.spaces['history_action'] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_history*action_vec_len,), dtype=np.float32)
            self.history_action_buffer = np.zeros((self.num_history, action_vec_len), dtype=np.float32)
        self.observation_space.spaces['history_reward'] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_history_action,), dtype=np.float32)
        self.history_reward_buffer = np.zeros((self.num_history_action), dtype=np.float32)

    def action_to_vect(self, c_action):
        curr = []
        for k in c_action:
            if k == 'camera':
                curr.append(c_action[k])
            else:
                curr.append(np.zeros((self.action_space.spaces[k].n), dtype=np.float32))
                curr[-1][c_action[k]] = 1.0
        return np.concatenate(curr, axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._step(obs, reward, action)
        return obs, reward, done, info

    def _step(self, obs, reward, action):
        if self.num_history > 0:
            self.history_action_buffer[0] = self.action_to_vect(action)
        self.history_reward_buffer[0] = reward
        
        obs['history_reward'] = np.copy(self.history_reward_buffer)
        if self.num_history > 0:
            obs['history_action'] = np.copy(self.history_action_buffer).reshape([self.num_history*self.action_vec_len])
        
        if self.num_history > 0:
            self.history_action_buffer = np.roll(self.history_action_buffer, 1, axis=0)
        self.history_reward_buffer = np.roll(self.history_reward_buffer, 1, axis=0)

        return obs

    def reset(self):
        obs = self.env.reset()
        obs = self._reset(obs)
        return obs

    def _reset(self, obs):
        if self.num_history > 0:
            self.history_action_buffer = np.zeros((self.num_history, self.action_vec_len), dtype=np.float32)
        self.history_reward_buffer = np.zeros((self.num_history_action), dtype=np.float32)
        if self.num_history > 0:
            obs['history_action'] = np.copy(self.history_action_buffer).reshape([self.num_history*self.action_vec_len])
        obs['history_reward'] = np.copy(self.history_reward_buffer)
        return obs

class MultiStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = IMultiStepWrapper(env)
        self.observation_space = self.wrapper.observation_space
        self.action_space = self.wrapper.action_space
        
    def step(self, action):
        return self.wrapper.step(action)

    def reset(self):
        return self.wrapper.reset()

class IMultiStepWrapper():
    def __init__(self, env):
        self.env = env
        self.wrapping_action_space = self.env.action_space

        self.rnoop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.step_reward = -0.0
        self.max_intra_step_count = 12
        self.current_intra_step = 0
        self.delta_min = 0.05
        self.delta = self.delta_min
        self.delta_chosen = False
        self.done = True
        self.reverse_reset_called = False

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera_change', 0),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])
        
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera_change':
                for i in range(1,6): #noop, inc/dec X/Y, inc bs_rect
                    op = copy.deepcopy(self.noop)
                    op[key] = i
                    self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        
        #add to obs space
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.current_action = copy.deepcopy(self.rnoop)
        for k in self.current_action:
            if k == 'camera':
                self.observation_space.spaces[k] = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            else:
                self.observation_space.spaces[k] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.wrapping_action_space.spaces[k].n,), dtype=np.float32)
        self.observation_space.spaces['bs_stuff'] = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.original_action_dim = self.get_original_dim()

        self.already_changed = {}
        for k in self.noop:
            if k != 'camera_change':
                self.already_changed[k] = False

    def get_original_dim(self):
        dim = 0
        for k in self.current_action:
            if k == 'camera':
                dim += 2
            else:
                dim += self.wrapping_action_space.spaces[k].n
        return dim

    def _vector_current(self, c_action):
        curr = []
        for k in c_action:
            if k == 'camera':
                curr.append(c_action[k])
            else:
                curr.append(np.zeros((self.wrapping_action_space.spaces[k].n), dtype=np.float32))
                curr[-1][c_action[k]] = 1.0
        return np.concatenate(curr, axis=0)

    def _add_current_to_obs(self, obs, c_action):
        assert isinstance(obs, dict) or isinstance(obs, OrderedDict)
        obs = copy.deepcopy(obs)
        for k in c_action:
            if k == 'camera':
                obs[k] = c_action[k]
            else:
                obs[k] = np.zeros((self.wrapping_action_space.spaces[k].n), dtype=np.float32)
                obs[k][c_action[k]] = 1.0
        obs['bs_stuff'] = np.array([self.delta, 1.0 if self.delta_chosen else 0.0], dtype=np.float32)
        return obs

    def reset(self):
        self.current_intra_step = 0
        self.delta = self.delta_min
        self.delta_chosen = False
        self.current_action = copy.deepcopy(self.rnoop)
        self.current_obs = self.env.reset()
        self.reverse_reset_called = False
        self.already_changed = {}
        for k in self.noop:
            if k != 'camera_change':
                self.already_changed[k] = False
        return self._add_current_to_obs(self.current_obs, self.current_action)

    def reset_intrastep(self):
        #reset intra step stuff
        self.current_intra_step = 0
        self.delta_chosen = False
        self.delta = self.delta_min
        self.current_action['left_right'] = 0
        self.current_action['attack_place_equip_craft_nearbyCraft_nearbySmelt'] = 0
        self.current_action['camera'] = np.zeros((2, ), dtype=np.float32)
        for k in self.noop:
            if k != 'camera_change':
                self.already_changed[k] = False

    def step(self, action):
        self.current_intra_step += 1
        if action == 0 or self.current_intra_step > (self.max_intra_step_count + 1):
            #noop => commit current action
            if np.abs(self.current_action['camera'][0]) > 1e-5 or np.abs(self.current_action['camera'][1]) > 1e-5:
                self.current_action['camera'] += (self.delta * 2.0 * np.random.uniform(-1.0, 1.0, (2))).astype(np.float32)
            self.current_action['camera'] = np.clip(self.current_action['camera'], -1.0, 1.0)
            real_action = self._vector_current(copy.deepcopy(self.current_action))
            self.current_action['camera'] *= 180.0
            self.current_obs, reward, done, info = self.env.step(self.current_action)

            self.reset_intrastep()

            self.done = done
            return self._add_current_to_obs(self.current_obs, self.current_action), reward, done, {'virtual_step': False, 'action': real_action, 'info': info}
        else:
            #not noop => change current action
            if not self.action_space.contains(action):
                raise ValueError('action {} is invalid for {}'.format(action, self.action_space))
            saction = self._actions[action]

            for k in saction:
                if saction[k] != 0:
                    if k != 'camera_change':
                        if not self.already_changed[k]:
                            if self.current_action[k] == saction[k]:
                                self.current_action[k] = 0 #toggle
                            else:
                                self.current_action[k] = saction[k]
                            self.already_changed[k] = True
                    else:
                        if saction[k] == 5:
                            if not self.delta_chosen:
                                self.delta *= 2.0
                                if self.delta > 0.5:
                                    self.delta_chosen = True
                                    self.delta = 0.5
                        else:
                            #start binary action choice
                            self.delta_chosen = True
                            ca = saction[k] - 1

                            if (ca & 1) == 0:
                                #dec X
                                self.current_action['camera'][0] -= self.delta
                            else:
                                #inc X
                                self.current_action['camera'][0] += self.delta

                            if ((ca >> 1) & 1) == 0:
                                #dec Y
                                self.current_action['camera'][1] -= self.delta
                            else:
                                #inc Y
                                self.current_action['camera'][1] += self.delta

                            self.delta *= 0.5


            return self._add_current_to_obs(self.current_obs, self.current_action), (self.step_reward if self.current_intra_step > 8 else 0.0), False, {'virtual_step': True, 'real_action': None, 'info': None}

    def reverse_reset(self, obs):
        self.current_intra_step = 0
        self.delta = self.delta_min
        self.delta_chosen = False
        self.current_action = copy.deepcopy(self.rnoop)
        self.current_obs = obs
        self.reverse_reset_called = True
        return self._add_current_to_obs(self.current_obs, self.current_action)

    def reverse_step(self, next_obs, original_action, treward, tdone):
        if not self.done:
            raise ValueError('Make sure last episode is finished before accessing reverse_step.')
        if not self.reverse_reset_called:
            raise ValueError('reverse_reset must be called before reverse_step.')


        obs_l = []
        action_l = []
        reward_l = []
        done_l = []
        info_l = []

        #first change everything except camera
        for k in self.current_action:
            if k != 'camera':
                self.current_intra_step += 1
                if self.current_action[k] != original_action[k]:
                    naction = copy.deepcopy(self.noop)
                    if original_action[k] == 0:
                        naction[k] = self.current_action[k]
                    else:
                        naction[k] = original_action[k]
                    action_index = self._actions.index(naction)
                    action_l.append(action_index)

                    self.current_action[k] = original_action[k]

                    robs = copy.deepcopy(self.current_obs)
                    robs = self._add_current_to_obs(robs, copy.deepcopy(self.current_action))
                    obs_l.append(robs)

                    reward_l.append(self.step_reward if self.current_intra_step > 8 else 0.0)
                    done_l.append(False)
                    info_l.append({'virtual_step': True, 'action': None, 'info': None})
                    

        #check if camera needs to be chosen
        if np.abs(original_action['camera'][0]) > 1e-5 or np.abs(original_action['camera'][1]) > 1e-5:
            #adjust delta so original_action['camera'] is reachable
            original_action['camera'] /= 180.0
            original_action['camera'] = np.clip(original_action['camera'], -1.0, 1.0)
            while np.abs(original_action['camera'][0]) > 2.0 * self.delta or \
                np.abs(original_action['camera'][1]) > 2.0 * self.delta:
                self.current_intra_step += 1
                naction = copy.deepcopy(self.noop)
                naction['camera_change'] = 5
                action_index = self._actions.index(naction)
                action_l.append(action_index)
                
                self.delta *= 2.0
                if self.delta >= 0.5:
                    self.delta_chosen = True
                    self.delta = 0.5

                robs = copy.deepcopy(self.current_obs)
                robs = self._add_current_to_obs(robs, copy.deepcopy(self.current_action))
                obs_l.append(robs)

                reward_l.append(self.step_reward if self.current_intra_step > 3 else 0.0)
                done_l.append(False)
                info_l.append({'virtual_step': True, 'action': None, 'info': None})

                if self.delta_chosen:
                    break

            self.delta_chosen = True

            #change camera
            n_camera_steps = 6
            for i in range(n_camera_steps):
                self.current_intra_step += 1

                caction = 0
                if original_action['camera'][0] < self.current_action['camera'][0]:
                    self.current_action['camera'][0] -= self.delta
                else:
                    self.current_action['camera'][0] += self.delta
                    caction = 1

                if original_action['camera'][1] < self.current_action['camera'][1]:
                    self.current_action['camera'][1] -= self.delta
                else:
                    self.current_action['camera'][1] += self.delta
                    caction += 2
                self.delta *= 0.5

                caction += 1
                naction = copy.deepcopy(self.noop)
                naction['camera_change'] = caction
                action_index = self._actions.index(naction)
                action_l.append(action_index)

                robs = copy.deepcopy(self.current_obs)
                robs = self._add_current_to_obs(robs, copy.deepcopy(self.current_action))
                obs_l.append(robs)

                reward_l.append(self.step_reward if self.current_intra_step > 3 else 0.0)
                done_l.append(False)
                info_l.append({'virtual_step': True, 'action': None, 'info': None})


        #finally add commit action
        if np.abs(original_action['camera'][0]-self.current_action['camera'][0]) >= self.delta * 2.0 + 1e-6 or \
            np.abs(original_action['camera'][1]-self.current_action['camera'][1]) >= self.delta * 2.0 + 1e-6:
            raise ValueError('current action and original action is not in range')
        self.current_intra_step += 1
        real_action = self._vector_current(copy.deepcopy(self.current_action))
        self.reset_intrastep()
        action_l.append(0)

        if not next_obs is None:
            self.current_obs = next_obs
            robs = copy.deepcopy(next_obs)
            robs = self._add_current_to_obs(robs, copy.deepcopy(self.current_action))
            obs_l.append(robs)

        reward_l.append(treward)
        done_l.append(tdone)
        info_l.append({'virtual_step': False, 'action': real_action, 'info': None})

        return obs_l, action_l, reward_l, done_l, info_l


class VisnetWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # sanity checks
        ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        if original_space != ideal_image_space:
            raise ValueError('Image space should be {}, but given {}.'.format(ideal_image_space, original_space))
        if original_space.dtype != np.uint8:
            raise ValueError('Image should `np.uint8` typed, but given {}.'.format(original_space.dtype))
        
        self.embed_dim = 512
        self.visnet = visual_net.VisualNet([3, height, width], self.embed_dim)
        self.visnet.load_state_dict(torch.load("best_bis_jz0_006/visnet_model.mdl", map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.visnet = self.visnet.to(device)
        self.visnet = self.visnet.train(False)
        
        self.observation_space = copy.deepcopy(self.observation_space)
        new_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.embed_dim,), dtype=np.float32)
        if self._key is None:
            self.observation_space = new_space
        else:
            self.observation_space.spaces[self._key] = new_space

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = np.expand_dims(frame, 0)
        frame = env_utils.minerl_to_net_image(frame)
        with torch.no_grad():
            frame = visual_net.variable(frame)
            frame = self.visnet(frame).data.cpu().numpy()
        frame = frame[0,:]
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class VisnetWrapper2(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key
        self.visnet = None

    def add_visnet(self, visnet):
        self.visnet = visnet
        self.embed_dim = 256

    def observation(self, obs):
        if self.visnet is None:
            return obs
        else:
            if self._key is None:
                frame = obs
            else:
                frame = obs[self._key]
            frame = frame[None, :]
            frame = self.visnet.liststate.premodel_forward(visual_net.variable(frame.copy()))
            frame = frame[0,:]
            if self._key is None:
                obs = frame
            else:
                obs[self._key] = frame
            return obs

class DictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_space = env.observation_space
        self.observation_space = {}
        self.t = None
        if isinstance(self.original_space, gym.spaces.Box):
            if len(self.original_space.shape) > 1:
                self.t = 'pov'
            else:
                self.t = 'state'
            self.observation_space[self.t] = self.original_space
        else:
            raise ValueError('Invalid original_space')

    def observation(self, observation):
        return {str(self.t): observation}

class MineRLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = IMineRLObsWrapper(env)
        self.observation_space = self.wrapper.observation_space
        
    def observation(self, observation):
        return self.wrapper.observation(observation)


class IMineRLObsWrapper():#gym.ObservationWrapper):
    def __init__(self, env):
        #super().__init__(env)
        self.env = env
        self.original_space = env.observation_space
        print(self.original_space)
        self.action_space = env.action_space

        self.is_list_obs = False
        self.mergeboxes_shape = [0]
        for name in self.original_space.spaces:
            if isinstance(self.original_space.spaces[name], gym.spaces.Box):
                if len(self.original_space.spaces[name].shape) > 1:
                    #raise ValueError('All boxes must be 1D')
                    self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.original_space.spaces[name].shape, dtype=np.float32)
                    self.is_list_obs = True
                else:
                    if len(self.original_space.spaces[name].shape) == 0:
                        if name != "compassAngle":
                            self.mergeboxes_shape[0] += 1
                        else:
                            self.mergeboxes_shape[0] += 2
                    else:
                        if name != "compassAngle":
                            self.mergeboxes_shape[0] += self.original_space.spaces[name].shape[0]
                        else:
                            self.mergeboxes_shape[0] += 2
            if isinstance(self.original_space.spaces[name], gym.spaces.Dict):
                for nname in self.original_space.spaces[name].spaces:
                    if isinstance(self.original_space.spaces[name].spaces[nname], gym.spaces.Box):
                        if len(self.original_space.spaces[name].spaces[nname].shape) == 1 and \
                            self.original_space.spaces[name].spaces[nname].shape[0] == 1:
                            self.mergeboxes_shape[0] += 4
                        if len(self.original_space.spaces[name].spaces[nname].shape) == 0:
                            self.mergeboxes_shape[0] += 4
                   
        if self.is_list_obs:
            self.observation_space = {'pov': self.observation_space, 'state': gym.spaces.Box(low=-1.0, high=1.0, shape=self.mergeboxes_shape, dtype=np.float32)}
        else:
            self.observation_space = {'state': gym.spaces.Box(low=-1.0, high=1.0, shape=self.mergeboxes_shape, dtype=np.float32)}


    def observation(self, observation):
        boxes = np.zeros(self.mergeboxes_shape)

        counter = 0
        if "compassAngle" in observation:
            boxes[counter] = np.cos((((observation["compassAngle"] + 180.0) * 2.0 / 360.0) - 1.0) * np.pi)
            boxes[counter+1] = np.sin((((observation["compassAngle"] + 180.0) * 2.0 / 360.0) - 1.0) * np.pi)
            counter += 2

        if not self.is_list_obs:
            boxes[counter:(counter+self.original_space.spaces["pov"].shape[0])] = observation["pov"]
            counter += self.original_space.spaces["pov"].shape[0]

        if "inventory" in observation:
            for i in observation["inventory"]:
                nbox = observation["inventory"][i]
                nvec = [-1.0]
                if nbox > 1:
                    nvec[0] = 1.0
                sig_offsets = np.array([4.0, 8.0, 32.0])
                sig_scale = np.array([2.0, 4.0, 16.0])
                sigs = np.tanh((nbox - sig_offsets) / sig_scale)
                for s in sigs:
                    nvec.append(s)
                boxes[counter:(counter+len(nvec))] = np.array(nvec)
                counter += len(nvec)
        elif "dirt" in observation:
            for i in observation["dirt"]:
                nbox = observation["dirt"][i]
                nvec = [-1.0]
                if nbox > 1:
                    nvec[0] = 1.0
                sig_offsets = np.array([4.0, 8.0, 32.0])
                sig_scale = np.array([2.0, 4.0, 16.0])
                sigs = np.tanh((nbox - sig_offsets) / sig_scale)
                for s in sigs:
                    nvec.append(s)
                boxes[counter:(counter+len(nvec))] = np.array(nvec)
                counter += len(nvec)

        for ob in observation:
            if ob != "dirt" and ob != "inventory" and ob != "compassAngle" and ob != "pov":
                boxes[counter:(counter+observation[ob].shape[0])] = observation[ob]
                counter += observation[ob].shape[0]

        #if counter != self.mergeboxes_shape[0]:
        #    raise ValueError('Mergeboxes and boxes dim dont line up')

        if not self.is_list_obs:
            return {'state': boxes}
        else:
            return {'pov': observation["pov"], 'state': boxes}


class MineRLObsWrapper2(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = IMineRLObsWrapper2(env)
        self.observation_space = self.wrapper.observation_space
        
    def observation(self, observation):
        return self.wrapper.observation(observation)


class IMineRLObsWrapper2():#gym.ObservationWrapper):
    def __init__(self, env):
        #super().__init__(env)
        self.env = env
        self.original_space = env.observation_space
        print(self.original_space)
        self.action_space = env.action_space

        self.is_list_obs = False
        self.mergeboxes_shape = [0]
        self.other_shape = {}
        self.box_space = None
        self.observation_space = {}
        for name in self.original_space.spaces:
            if isinstance(self.original_space.spaces[name], gym.spaces.Discrete):
                self.observation_space[name] = gym.spaces.Discrete(self.original_space.spaces[name].n)
            if isinstance(self.original_space.spaces[name], gym.spaces.Box):
                if len(self.original_space.spaces[name].shape) > 1:
                    #raise ValueError('All boxes must be 1D')
                    self.box_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.original_space.spaces[name].shape, dtype=np.float32)
                    self.is_list_obs = True
                else:
                    if len(self.original_space.spaces[name].shape) == 0:
                        if 'history' in name or name == 'orientation':
                            self.other_shape[name] = [1]
                            continue
                        if name != "compassAngle":
                            self.mergeboxes_shape[0] += 1
                        else:
                            self.mergeboxes_shape[0] += 2
                    else:
                        if 'last_' in name:
                            self.other_shape[name] = self.original_space.spaces[name].shape
                            continue
                        if 'history' in name or name == 'orientation':
                            self.other_shape[name] = [self.original_space.spaces[name].shape[0]]
                            continue
                        if name != "compassAngle":
                            self.mergeboxes_shape[0] += self.original_space.spaces[name].shape[0]
                        else:
                            self.mergeboxes_shape[0] += 2
            if isinstance(self.original_space.spaces[name], gym.spaces.Dict):
                for nname in self.original_space.spaces[name].spaces:
                    if isinstance(self.original_space.spaces[name].spaces[nname], gym.spaces.Box):
                        if len(self.original_space.spaces[name].spaces[nname].shape) == 1 and \
                            self.original_space.spaces[name].spaces[nname].shape[0] == 1:
                            self.mergeboxes_shape[0] += 4
                        if len(self.original_space.spaces[name].spaces[nname].shape) == 0:
                            self.mergeboxes_shape[0] += 4
                    if isinstance(self.original_space.spaces[name].spaces[nname], gym.spaces.Dict):
                        for nnname in self.original_space.spaces[name].spaces[nname].spaces:
                            if isinstance(self.original_space.spaces[name].spaces[nname].spaces[nnname], gym.spaces.Box):
                                if len(self.original_space.spaces[name].spaces[nname].spaces[nnname].shape) == 1 and \
                                    self.original_space.spaces[name].spaces[nname].spaces[nnname].shape[0] == 1:
                                    self.mergeboxes_shape[0] += 4
                                if len(self.original_space.spaces[name].spaces[nname].spaces[nnname].shape) == 0:
                                    self.mergeboxes_shape[0] += 4

                   
        if self.is_list_obs:
            self.observation_space['pov'] = self.box_space
        if self.mergeboxes_shape[0] != 0:
            self.observation_space['state'] = gym.spaces.Box(low=-1.0, high=1.0, shape=self.mergeboxes_shape, dtype=np.float32)
        for o in self.other_shape:
            self.observation_space[o] = gym.spaces.Box(low=-1.0, high=1.0, shape=self.other_shape[o], dtype=np.float32)


    def observation(self, observation):
        boxes = np.zeros(self.mergeboxes_shape)

        counter = 0
        if "compassAngle" in observation:
            boxes[counter] = np.cos((((observation["compassAngle"] + 180.0) * 2.0 / 360.0) - 1.0) * np.pi)
            boxes[counter+1] = np.sin((((observation["compassAngle"] + 180.0) * 2.0 / 360.0) - 1.0) * np.pi)
            counter += 2

        if not self.is_list_obs:
            boxes[counter:(counter+self.original_space.spaces["pov"].shape[0])] = observation["pov"]
            counter += self.original_space.spaces["pov"].shape[0]

        if "inventory" in observation:
            for i in observation["inventory"]:
                nbox = observation["inventory"][i]
                nvec = [-1.0]
                if nbox > 1:
                    nvec[0] = 1.0
                sig_offsets = np.array([4.0, 8.0, 32.0])
                sig_scale = np.array([2.0, 4.0, 16.0])
                sigs = np.tanh((nbox - sig_offsets) / sig_scale)
                for s in sigs:
                    nvec.append(s)
                boxes[counter:(counter+len(nvec))] = np.array(nvec)
                counter += len(nvec)
        elif "dirt" in observation:
            for i in observation["dirt"]:
                nbox = observation["dirt"][i]
                nvec = [-1.0]
                if nbox > 1:
                    nvec[0] = 1.0
                sig_offsets = np.array([4.0, 8.0, 32.0])
                sig_scale = np.array([2.0, 4.0, 16.0])
                sigs = np.tanh((nbox - sig_offsets) / sig_scale)
                for s in sigs:
                    nvec.append(s)
                boxes[counter:(counter+len(nvec))] = np.array(nvec)
                counter += len(nvec)

        ret = {}
        for ob in observation:
            if 'history' in ob or 'last_' in ob or ob == 'orientation':
                ret[ob] = observation[ob]
                continue
            if ob != "dirt" and ob != "inventory" and ob != "compassAngle" and ob != "pov":
                boxes[counter:(counter+observation[ob].shape[0])] = observation[ob]
                counter += observation[ob].shape[0]

        #if counter != self.mergeboxes_shape[0]:
        #    raise ValueError('Mergeboxes and boxes dim dont line up')
        
        if self.mergeboxes_shape[0] != 0:
            ret['state'] = boxes
        if self.is_list_obs:
            ret['pov'] = observation["pov"]
        return ret

class MineRLObsWrapper3(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wrapper = IMineRLObsWrapper3(env)
        self.observation_space = self.wrapper.observation_space
        
    def observation(self, observation):
        return self.wrapper.observation(observation)


class IMineRLObsWrapper3():
    def __init__(self, env):
        self.env = env
        self.original_space = env.observation_space
        print(self.original_space)
        self.action_space = env.action_space

        self.state_dim = self.get_dim(self.original_space)

        self.observation_space = {}
        self.observation_space['pov'] = gym.spaces.Box(low=-1.0, high=1.0, shape=self.original_space.spaces['pov'].shape, dtype=np.float32)
        self.observation_space['state'] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,))

    def get_dim(self, A, name = 'root', parent_name = 'root'):
        if isinstance(A, gym.spaces.Dict):
            ret = 0
            for a in A.spaces:
                if a != 'pov' and not 'amage' in a:
                    ret += self.get_dim(A.spaces[a], a, name)
            if name == 'inventory':
                print('Inventory len', ret)
            return ret
        if isinstance(A, gym.spaces.Box):
            if name == 'compassAngle':
                print('Got compassAngle')
                return 2
            elif parent_name == 'inventory':
                return 4
            else:
                if len(A.shape) <= 1:
                    return A.shape[0]
                else:
                    assert False #not implemented
        if isinstance(A, gym.spaces.Discrete):
            print('Got discrete', parent_name, name, A.n)
            return A.n
        print(A)
        assert False #not implemented

    def observation(self, observation):
        ret = {}
        ret['pov'] = observation['pov']
        ret['state'] = self.merge_state(observation, self.original_space)
        assert ret['state'].shape[0] == self.state_dim
        return ret

    def merge_state(self, obs, space, name = 'root', parent_name = 'root'):
        if isinstance(space, gym.spaces.Dict):
            ret = []
            for a in space.spaces:
                if a != 'pov' and not 'amage' in a:
                    ret.append(self.merge_state(obs[a], space.spaces[a], a, name))
            return np.concatenate(ret, axis=0)
        if isinstance(space, gym.spaces.Box):
            if name == 'compassAngle':
                return np.array([np.cos((((obs + 180.0) * 2.0 / 360.0) - 1.0) * np.pi),
                                 np.sin((((obs + 180.0) * 2.0 / 360.0) - 1.0) * np.pi)])
            elif parent_name == 'inventory':
                nvec = np.array([-1.0])
                if obs > 1:
                    nvec[0] = 1.0
                sig_offsets = np.array([4.0, 8.0, 32.0])
                sig_scale = np.array([2.0, 4.0, 16.0])
                sigs = np.tanh((obs - sig_offsets) / sig_scale)
                return np.concatenate([nvec, sigs], axis=0)
            else:
                return np.array(obs)
        if isinstance(space, gym.spaces.Discrete):
            if isinstance(obs, str):
                typ = ['none','air','wooden_axe','wooden_pickaxe','stone_axe','stone_pickaxe','iron_axe','iron_pickaxe','other']
                if obs in typ:
                    ret = np.zeros(space.n)
                    ret[typ.index(obs)] = 1.0
                    return ret
                else:
                    ret = np.zeros(space.n)
                    ret[-1] = 1.0
                    return ret
            else:
                ret = np.zeros(space.n)
                ret[obs] = 1.0
                return ret
        print(space)
        assert False #not implemented



class MultiDataWrapper():
    def __init__(self, datas):
        self.action_space = copy.deepcopy(datas[0].action_space)
        self.observation_space = copy.deepcopy(datas[0].observation_space)

        for i in range(len(datas)):
            for a in datas[i].action_space.spaces:
                if not a in self.action_space.spaces:
                    self.action_space.spaces[a] = copy.deepcopy(datas[i].action_space.spaces[a])
                else:
                    self.action_space.spaces[a] = self.merge_space(self.action_space.spaces[a],
                                                                    datas[i].action_space.spaces[a])
            for o in datas[i].observation_space.spaces:
                if not o in self.observation_space.spaces:
                    self.observation_space.spaces[o] = copy.deepcopy(datas[i].observation_space.spaces[o])
                else:
                    self.observation_space.spaces[o] = self.merge_space(self.observation_space.spaces[o],
                                                                 datas[i].observation_space.spaces[o])

        self.observation_space.spaces['env_type'] = gym.spaces.Box(0.0, 1.0, shape=(len(datas),))

    def _new_obs(self, obs, env_type):
        c_obs = {}
        for a in self.observation_space.spaces:
            if a != 'env_type':
                if a in obs:
                    c_obs[a] = self.make_input(self.observation_space.spaces[a], obs[a])
                else:
                    c_obs[a] = self.make_template(self.observation_space.spaces[a])
            else:
                c_obs[a] = np.zeros(self.observation_space.spaces['env_type'].shape[0])
                c_obs[a][env_type] = 1.0
        return c_obs
    
    def _new_act(self, action):
        c_act = {}
        for a in self.action_space.spaces:
            if a in action:
                c_act[a] = self.make_input(self.action_space.spaces[a], action[a])
            else:
                c_act[a] = self.make_template(self.action_space.spaces[a])
        return c_act

    def make_input(self, A, inp):
        if isinstance(A, gym.spaces.Dict):
            ret = {}
            for a in A.spaces:
                if a in inp:
                    ret[a] = self.make_input(A.spaces[a], inp[a])
                else:
                    ret[a] = self.make_template(A.spaces[a])
        elif isinstance(A, gym.spaces.Discrete):
            ret = inp
        elif isinstance(A, gym.spaces.Box):
            if len(A.shape) == 0:
                ret = inp
            else:
                ret = np.zeros(A.shape)
                if np.all(A.shape == inp.shape):
                    ret = inp
                else:
                    print('mismatch detected',inp.shape,A.shape)
                    if len(A.shape) == 1:
                        ret[:inp.shape[0]] = inp
                    elif len(A.shape) == 2:
                        ret[:inp.shape[0],:inp.shape[1]] = inp
                    elif len(A.shape) == 3:
                        ret[:inp.shape[0],:inp.shape[1],:inp.shape[2]] = inp
                    else:
                        assert False #not implemeted
        return ret

    def make_template(self, A):
        if isinstance(A, gym.spaces.Dict):
            template = {}
            for a in A.spaces:
                template[a] = self.make_template(A.spaces[a])
        elif isinstance(A, gym.spaces.Discrete):
            template = 0
        elif isinstance(A, gym.spaces.Box):
            if len(A.shape) == 0:
                template = 0.0
            else:
                template = np.zeros(A.shape)
        return template

    def merge_space(self, A, B):
        assert type(A) == type(B)
        if isinstance(B, gym.spaces.Dict):
            for a in B.spaces:
                if not a in A.spaces:
                    A.spaces[a] = copy.deepcopy(B.spaces[a])
                else:
                    A.spaces[a] = self.merge_space(A.spaces[a], B.spaces[a])
        elif isinstance(B, gym.spaces.Discrete):
            if A.n < B.n:
                A = copy.deepcopy(B)
        elif isinstance(B, gym.spaces.Box):
            if not np.all(A.shape == B.shape) and np.all(A.low == B.low) and np.all(A.high == B.high):
                if len(A.shape) != len(B.shape):
                    assert False #not implemented
                else:
                    if len(A.shape) != 0:
                        A = gym.spaces.Box(np.minimum(A.low, B.low), np.maximum(A.high, B.high), shape=np.maximum(A.shape, B.shape))
                    else:
                        A = gym.spaces.Box(np.minimum(A.low, B.low), np.maximum(A.high, B.high))
        else:
            assert False #not implemented
        return A
    
class SaveOrigObsAct(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        self.orig_obs = self.env.reset()
        return copy.deepcopy(self.orig_obs)

    def step(self, action):
        self.orig_action = copy.deepcopy(action)
        obs, reward, done, info = self.env.step(action)
        self.orig_obs = copy.deepcopy(obs)
        return obs, reward, done, info


class ConstantMultiDataWrapper(gym.Wrapper):
    def __init__(self, env, datas, env_type):
        super().__init__(env)
        self.env = env
        self.env_type = env_type
        print('ENVIROMENT INDEX',env_type)
        self.datas = datas
        self.action_space = copy.deepcopy(datas[0].action_space)
        self.observation_space = copy.deepcopy(datas[0].observation_space)

        for i in range(len(datas)):
            for a in datas[i].action_space.spaces:
                if not a in self.action_space.spaces:
                    self.action_space.spaces[a] = copy.deepcopy(datas[i].action_space.spaces[a])
                else:
                    self.action_space.spaces[a] = self.merge_space(self.action_space.spaces[a],
                                                                    datas[i].action_space.spaces[a])
            for o in datas[i].observation_space.spaces:
                if not o in self.observation_space.spaces:
                    self.observation_space.spaces[o] = copy.deepcopy(datas[i].observation_space.spaces[o])
                else:
                    self.observation_space.spaces[o] = self.merge_space(self.observation_space.spaces[o],
                                                                 datas[i].observation_space.spaces[o])

        self.observation_space.spaces['env_type'] = gym.spaces.Box(0.0, 1.0, shape=(len(datas),))

    def reset(self):
        return self._new_obs(self.env.reset(), self.env_type)

    def step(self, action):
        raction = self.extract_action(action)
        obs, reward, done, info = self.env.step(raction)
        robs = self._new_obs(obs, self.env_type)
        return robs, reward, done, info

    def extract_action(self, action):
        retaction = {}
        for a in self.datas[self.env_type].action_space.spaces:
            retaction[a] = action[a]
        return retaction

    def _new_obs(self, obs, env_type):
        c_obs = {}
        for a in self.observation_space.spaces:
            if a != 'env_type':
                if a in obs:
                    c_obs[a] = self.make_input(self.observation_space.spaces[a], obs[a])
                else:
                    c_obs[a] = self.make_template(self.observation_space.spaces[a])
            else:
                c_obs[a] = np.zeros(self.observation_space.spaces['env_type'].shape[0])
                c_obs[a][env_type] = 1.0
        return c_obs
    
    def _new_act(self, action):
        c_act = {}
        for a in self.action_space.spaces:
            if a in action:
                c_act[a] = self.make_input(self.action_space.spaces[a], action[a])
            else:
                c_act[a] = self.make_template(self.action_space.spaces[a])
        return c_act

    def make_input(self, A, inp):
        if isinstance(A, gym.spaces.Dict):
            ret = {}
            for a in A.spaces:
                if a in inp:
                    ret[a] = self.make_input(A.spaces[a], inp[a])
                else:
                    ret[a] = self.make_template(A.spaces[a])
        elif isinstance(A, gym.spaces.Discrete):
            ret = inp
        elif isinstance(A, gym.spaces.Box):
            if len(A.shape) == 0:
                ret = inp
            else:
                ret = np.zeros(A.shape)
                if np.all(A.shape == inp.shape):
                    ret = inp
                else:
                    print('mismatch detected',inp.shape,A.shape)
                    if len(A.shape) == 1:
                        ret[:inp.shape[0]] = inp
                    elif len(A.shape) == 2:
                        ret[:inp.shape[0],:inp.shape[1]] = inp
                    elif len(A.shape) == 3:
                        ret[:inp.shape[0],:inp.shape[1],:inp.shape[2]] = inp
                    else:
                        assert False #not implemeted
        return ret

    def make_template(self, A):
        if isinstance(A, gym.spaces.Dict):
            template = {}
            for a in A.spaces:
                template[a] = self.make_template(A.spaces[a])
        elif isinstance(A, gym.spaces.Discrete):
            template = 0
        elif isinstance(A, gym.spaces.Box):
            if len(A.shape) == 0:
                template = 0.0
            else:
                template = np.zeros(A.shape)
        return template

    def merge_space(self, A, B):
        assert type(A) == type(B)
        if isinstance(B, gym.spaces.Dict):
            for a in B.spaces:
                if not a in A.spaces:
                    A.spaces[a] = copy.deepcopy(B.spaces[a])
                else:
                    A.spaces[a] = self.merge_space(A.spaces[a], B.spaces[a])
        elif isinstance(B, gym.spaces.Discrete):
            if A.n < B.n:
                A = copy.deepcopy(B)
        elif isinstance(B, gym.spaces.Box):
            if not np.all(A.shape == B.shape) and np.all(A.low == B.low) and np.all(A.high == B.high):
                if len(A.shape) != len(B.shape):
                    assert False #not implemented
                else:
                    if len(A.shape) != 0:
                        A = gym.spaces.Box(np.minimum(A.low, B.low), np.maximum(A.high, B.high), shape=np.maximum(A.shape, B.shape))
                    else:
                        A = gym.spaces.Box(np.minimum(A.low, B.low), np.maximum(A.high, B.high))
        else:
            assert False #not implemented
        return A


if __name__ == '__main__':
    #tests for minerl wrappers
    import minerl
    if False:
        envname = 'MineRLNavigateDense-v0'
        env = gym.make(envname)

        env = ResetTrimInfoWrapper(env)
        visnetenv = VisnetWrapper(env, "pov")
        combinenv = CombineActionWrapper(visnetenv)
        multistepenv = MultiStepWrapper(combinenv)
        env = MineRLObsWrapper(multistepenv)

        #test CombineActionWrapper.reverse_action
        print('test CombineActionWrapper.reverse_action')
        print('')
        print('')
        print('')
        op = copy.deepcopy(combinenv.noop)
        op['forward_back'] = 2
        op['jump'] = 1
        op['attack_place_equip_craft_nearbyCraft_nearbySmelt'] = 1
    
        print(op)
        orig_op = combinenv.action(op)
        print('')
        print(orig_op)
        revert_op = combinenv.reverse_action(orig_op)
        print('')
        print(revert_op)
        #corrupt orig_op
        orig_op['forward'] = 1
        orig_op['place'] = 1
        orig_op['sneak'] = 1
        orig_op['sprint'] = 1
        revert_op = combinenv.reverse_action(orig_op)
        print('')
        print(revert_op)



        #test MultiStepWrapper.reverse_action
        print('')
        print('')
        print('')
        print('test MultiStepWrapper.reverse_action')
        print('')
        print('')
        print('')
        op['camera'] = np.array([-0.0033, -0.14083]) * 180.0
        dummy_obs = OrderedDict()
        first_ops = multistepenv.reverse_reset(dummy_obs)
        obs_l, action_l, reward_l, done_l, dbg_current = multistepenv.reverse_step(dummy_obs, op, 1.0, False)

        for a in action_l:
            print('')
            print(a)

    
        print('')
        print('')
        print(dbg_current)

        env.close()
        print('done')
    else:
        all_envs = ['MineRLTreechop-v0',
                    'MineRLNavigateDense-v0',
                    'MineRLNavigate-v0',
                    'MineRLNavigateExtremeDense-v0',
                    'MineRLNavigateExtreme-v0',
                    'MineRLObtainIronPickaxe-v0',
                    'MineRLObtainIronPickaxeDense-v0',
                    'MineRLObtainDiamond-v0',
                    'MineRLObtainDiamondDense-v0']
        datas = [minerl.data.make(tenv) for tenv in all_envs]
        multi_data = MultiDataWrapper(datas)
        print(multi_data.observation_space)
        print(multi_data.action_space)