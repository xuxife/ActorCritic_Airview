#!python3
"""
Huawei Airview Simulator
Gym Environment Wrapper

author: 116010252@link.cuhk.edu.cn
"""

import numpy as np
from collections import Counter, OrderedDict
import logging

import gym
from gym import spaces

" Configuration "
BANDWIDTH = 1e7
RBG_NUM = 17
TTI = 0.001
UE_ARRIVAL_RATE = 0.5  # uniform(0, 1) < TTI / RATE
EPISODE_LENGTH = 2  # the maximum simulation time of one episode
PACKET_SIZE = (int(1e3), int(1e6))
CQI_REPORT_INTERVAL = 0.02
PRIOR_THRESHOLD = 1e4
MIN_CQI = 1
MAX_CQI = 29
MIN_MCS = 1
MAX_MCS = 29
DEFAULT_MCS = 4


class UE:
    """
    UE is an abstract network user/customer for AirView routing estimator environment.
    Attributes:
        state_var: Pre-defined attributes needed in state
            {
                'scalar': List[scalar attributes],
                'vector': List[vector attributes]
            }
        attr_range: Numeric range for attributes

        avg_*: Average *
        sched_*: Scheduled *
        buffer: The remaining buffer size (bytes)
        rsrp: Reference Signal Receiving Power
        snr: Signal-to-Noise Ratio
        thp: Throughput
        cqi: Channel Quality Indicator
        se: Spectrum Efficiency
        prior: Priority of UE in the current RBG assignment
    """
    state_var = {
        'scalar': ['buffer', 'rsrp', 'avg_thp', 'avg_cqi', 'sched_rbg_num'],
        # 'scalar': ['cqi', 'mcs', 'se', 'prior', 'sched_rbg'],
        # 'vector': ['cqi', 'se', 'prior', 'sched_rbg'],
        'vector': [],
    }
    state_dim = len(state_var['scalar']) + RBG_NUM * len(state_var['vector'])
    attr_range = {
        'buffer': (int(1e3), int(1e6)),
        'rsrp': (-120, -89),
        'avg_snr': (1, 32),
        'avg_thp': (0, BANDWIDTH * 0.9 * TTI * np.log2(1+29**2)),
        'avg_cqi': (1, 29),
        'sched_rbg_num': (0, RBG_NUM),
        'cqi': (1, 29),
        'mcs': (1, 29),
        'se': tuple(map(lambda x: np.log2(1+x**2), (1, 29))),
        'prior': (0, np.log2(1 + 29 ** 2)),
        'sched_rbg': (0, 1),
    }

    def __init__(self, ue_id, arrive):
        self.ID = ue_id
        self.arrive = arrive
        self.buffer = np.random.randint(*self.attr_range['buffer'])
        self.rsrp = np.random.randint(*self.attr_range['rsrp'])
        self.avg_snr = self.rsrp + 121
        self.avg_thp = 0
        self.cqi = np.full(RBG_NUM, np.nan)
        self.mcs = np.full(RBG_NUM, np.nan)
        self.se = np.full(RBG_NUM, np.nan)
        self.prior = np.full(RBG_NUM, np.nan)
        self.send_num = 0  # number of packages already sent
        self.arrive_num = 0  # number of packages finished transmission
        # self.is_ack, self.sched_mcs = [], []
        # bool-like vector, whether the ue is assigned by RBG
        self.sched_rbg = np.zeros(RBG_NUM)
        self.tti_end()

    def tti_end(self):
        " the attributes that needs re-initialization every tti "
        self.sched_rbg.fill(0)  # rbg assigned == 1, otherwise 0

    @property
    def avg_cqi(self):
        return (self.mcs*self.sched_rbg).mean()

    @property
    def sched_rbg_num(self):
        return self.sched_rbg.sum()

    def __repr__(self):
        return f"<UE: {self.ID} arrive: {self.arrive}>"

    def __lt__(self, other):
        return self.ID < other.ID

    def __getitem__(self, x):
        return getattr(self, x)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def _scale(self, value, attr_range):
        return (value - attr_range[0])/(attr_range[1] - attr_range[0])

    @property
    def state(self):
        " return attributes scaled in [0, 1] in np.array "
        s = np.empty(self.state_dim)
        i = 0
        for attr in self.state_var['scalar']:
            s[i] = self._scale(self[attr], self.attr_range[attr])
            i += 1
        for attr in self.state_var['vector']:
            s[i:(i+RBG_NUM)] = self._scale(self[attr], self.attr_range[attr])
            i += RBG_NUM
        return s


class Airview(gym.Env):
    """
    Experimental Environment of AirView based on OpenAI Gym

    Args:
        ue_arrival_rate (Int): the rate (probability) of UE arrival
        cqi_report_interval (Int): the length of CQI report interval

    Attributes:
        ues (List[UE]): 
        sim_time (float): simulation time
        select_ue (List[UE]): len(select_ue) == RBG_NUM,
            select_ue[i] is the UE who RBG[i] current serves.

    Examples:
        policy = SomePolicy() # have methods .decide & .learn
        env = Airview()
        state = env.reset()
        for _ in range(TRAIN_STEPS):
            action = policy.decide(state)
            next_state, reward, done, _ = env.step(action)
            policy.learn((state, reward, next_state, done))
            state = next_state
    """

    def __init__(self, episode_length=EPISODE_LENGTH, ue_arrival_rate=UE_ARRIVAL_RATE, cqi_report_interval=CQI_REPORT_INTERVAL):
        self.episode_length = episode_length
        self.ue_arrival_rate = ue_arrival_rate
        self.cqi_report_interval = cqi_report_interval

        self._reset()

        self.action_space = spaces.MultiDiscrete([MAX_MCS-MIN_MCS+1]*RBG_NUM)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(RBG_NUM, UE.state_dim))
        self.reward_range = (0, BANDWIDTH*0.9*UE.attr_range['se'][1]*TTI)

        self.sched_ue_count = OrderedCounter()

    def _reset(self):
        " reset attributes "
        self.ues = []
        self.ue_num = 0
        self.sim_time = 0
        self.select_ue = [None] * RBG_NUM
        self.state = None

    def reset(self):
        " reset the environment at the moment first UE arrives "
        self._reset()
        while len(self.ues) == 0:
            self._run()
        self._run()
        # self.state = np.tile(self.ues[0].state, (RBG_NUM, 1))
        self.state = np.expand_dims(self.ues[0].state, 0)
        self.sched_ue_count[self.ues[0]] = self.ues[0].sched_rbg_num
        return self.state

    def step(self, action):
        """ 
        Args:
            action (List[Int]): len(action) == RBG_NUM, 
                action[i] represents the MCS level of RBG[i]

        Returns:
            state (np.array): state.shape == (ue_num, UE.state_dim)
            reward (float): reward returned after taking the action
            done (bool): whether the episode has ended
            info (dict): extra informaction
        """

        reward = 0
        total = 0
        is_acks = []
        for sched_mcs, ue in zip(action, self.sched_ue_count.keys()):
            # ue.sched_mcs.append((self.sim_time, sched_mcs))
            # reward
            is_ack = self._is_ack(ue, sched_mcs)
            is_acks.append(is_ack)
            # ue['is_ack'].append((self.sim_time, is_ack))
            rbg_se = np.log2(1 + sched_mcs**2)
            # tbs == transmission block size
            rbg_tbs = int(BANDWIDTH * 0.9 *
                          self.sched_ue_count[ue] / RBG_NUM * rbg_se * TTI)
            ue.avg_thp = 0.01*is_ack*rbg_tbs + (1-0.01)*ue.avg_thp
            reward += min(ue.buffer, is_ack * rbg_tbs)
            total += min(ue.buffer, rbg_tbs)
            ue.buffer = max(0, ue.buffer - rbg_tbs)
        self._run()
        self.sched_ue_count = self._sched_ue_count()
        self.state = np.vstack([ue.state for ue in self.sched_ue_count.keys()])
        return self.state, reward, self.done, {'total': total}

    def _sched_ue_count(self):
        count = OrderedCounter()
        for ue in self.select_ue:
            count[ue] += 1
        return count

    @property
    def done(self):
        " judge whether the episode ends "
        return self.sim_time >= self.episode_length

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _run(self):
        self.sim_time += TTI
        self.sim_time = round(self.sim_time, 3)

        self._rm_no_buffer_ue()
        for ue in self.ues:
            ue.tti_end()

        self._arrive()

        " new ue arrives "
        if np.random.uniform(0., 1.) < TTI / self.ue_arrival_rate:
            self.ues.append(UE(self.ue_num, self.sim_time))
            self.ue_num += 1
            logging.debug(f"{self.sim_time}: {self.ues[-1]} arrives.")

        " filter the ues needs calc_prior & select & send "
        ues = list(filter(lambda ue: self._can_send(ue), self.ues))
        if len(ues) == 0:
            return

        " calculate priority "
        ues = self._calc_prior(ues)

        " select ue for every rbg "
        ues = self._select(ues)

        " then waiting for user giving a action based on the state generated from select_ue "
        " call self.take_action later "
        return ues

    def _calc_prior(self, ues):
        " for every ue, update rbg_cqi per 'sqi_report_interval' "
        for ue in ues:
            dt = self.sim_time - ue.arrive
            # if np.isclose(dt % self.cqi_report_interval, 0.0) or np.isclose(dt, self.cqi_report_interval):
            ue.cqi = ue.avg_snr + np.random.randint(-2, 3, size=RBG_NUM)
            np.clip(ue.cqi, *ue.attr_range['cqi'], out=ue.cqi)
            logging.debug(f"{self.sim_time}: cqi reported for {ue}")

            np.copyto(ue.mcs, ue.cqi)
            # ue.mcs[np.isnan(ue.mcs)] = DEFAULT_MCS  # set default mcs
            np.log2(1 + ue.mcs**2.0, out=ue.se)
            ue.prior = ue.se / max(1, ue.avg_thp/PRIOR_THRESHOLD)
        return ues

    def _select(self, ues):
        " for each rbg, find the ue having the maximum priority "
        for rbg in range(RBG_NUM):
            max_prior = -1
            selected_ue = None
            for ue in ues:
                if self._can_send(ue) and ue.prior[rbg] > max_prior:
                    max_prior = ue.prior[rbg]
                    selected_ue = ue
            self.select_ue[rbg] = selected_ue
            if selected_ue is not None:
                selected_ue.sched_rbg[rbg] = 1
        return ues

    def _arrive(self):
        " finish transmission and rewards returned "
        for ue in filter(lambda u: u is not None, self.select_ue):
            if ue.arrive_num < ue.send_num:
                ue.arrive_num += 1

    def _can_send(self, ue):
        " determine whether the ue is avaliable to be selected for transmission "
        return not any([
            ue.arrive == self.sim_time,  # just arrive
            ue.arrive_num < ue.send_num,
            ue.buffer <= 0,
        ])

    def _rm_no_buffer_ue(self):
        " clean up the ues who no longer have buffer "
        self.ues = list(filter(lambda x: x.buffer > 0, self.ues))

    def _is_ack(self, ue, sched_mcs):
        " whether the ue successfully sends the package in the given sched_mcs "
        is_ack = ue.avg_snr + np.random.randint(-2, 3) - sched_mcs
        if is_ack > 0:
            is_ack = 1
        elif is_ack < 0:
            is_ack = 0
        else:
            is_ack = np.random.randint(0,2)
        return is_ack


class Policy:
    " default policy (taking floor after mean) "

    def __init__(self, mode="avg"):
        self.mode = mode

    def decide(self, ues):
        if self.mode == "avg":
            return np.array([np.floor(np.sum(ue.mcs*ue.sched_rbg)/ue.sched_rbg.sum()) for ue in ues])
        assert self.mode == "snr"
        return np.array([ue.avg_snr for ue in ues])

    def learn(self, *args):
        pass


class Policy2:
    def decide(self, ues):
        return [ue.mcs[i] for i, ue in enumerate(ues)]


class OrderedCounter(OrderedDict, Counter):
    pass
