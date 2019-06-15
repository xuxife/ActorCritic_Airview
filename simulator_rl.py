# -*- coding: utf-8 -*-

import gym
import numpy as np
import copy
from sklearn.preprocessing import minmax_scale

"""
Configuration:
"""
BANDWIDTH = 1e7
RBG_NUM = 17
TTI = 0.001
UE_ARRIVAL_RATE = 0.05
PACKET_SIZE = (int(1e3), int(1e6))
CQI_REPORT_INTERVAL = 0.02
PRIOR_THRESHOLD = 1e4
MIN_CQI = 1
MAX_CQI = 29
MIN_MCS = 1
MAX_MCS = 29
EPISODE_TTI = 10.0


class User:
    var = {
        'num': ['buffer', 'rsrp', 'avg_snr', 'avg_thp'],
        'vec': ['cqi', 'se', 'prior', 'sched_rbg']
    }
    attr_range = {
        'buffer': (int(1e3), int(1e6)),
        'rsrp': (-120, -90),
        'avg_snr': (1, 31),
        'avg_thp': (0, BANDWIDTH * 0.9 * TTI * np.log2(1 + 29 ** 2)),
        'cqi': (1, 29),
        'mcs': (1, 29),
        'se': tuple(map(lambda x: np.log2(1 + x ** 2), (1, 29))),
        'prior': (0, np.log2(1 + 29 ** 2)),
        'sched_rbg': (0, 1)
    }

    def __init__(self, user_id, arr_time, buffer, rsrp, is_virtual=False):
        self.ID = user_id
        self.arr_time = arr_time
        self.buffer = buffer
        self.rsrp = rsrp
        self.avg_snr = self.rsrp + 121
        self.avg_thp = 0
        self.cqi = np.full(RBG_NUM, np.nan)
        self.mcs = np.full(RBG_NUM, np.nan)
        self.se = np.full(RBG_NUM, np.nan)
        self.prior = np.full(RBG_NUM, np.nan)
        self.sched_rbg = np.zeros(RBG_NUM)
        self.tbs_list = []
        self.is_virtual = is_virtual

    def reset_rbg(self):
        self.sched_rbg.fill(0)

    def __getitem__(self, x):
        return getattr(self, x)

    def __setitem__(self, key, value):
        self.__dict__[key] = value


"""
Here we set the state of environment from the perspective of RBGs
The dimension is 17*(attr)
attr is the features of user who is assigned to this RBG, which includes:
buffer: the package size of the user
rsrp: the user's rsrp
avg_snr: the user's avg_snr
avg_thp: the user' avg_thp
cqi: the user's cqi in this RBG
se: the user's se in this RBG
prior: the user's prior in this RBG
"""


class Airview(gym.Env):
    user_var = {
        'num': ['buffer', 'rsrp', 'avg_thp', 'cqi'],
        'vec': []
    }

    def __init__(self, ue_arrival_rate=UE_ARRIVAL_RATE, episode_tti=EPISODE_TTI):
        self.ue_arrival_rate = ue_arrival_rate
        self.cqi_report_interval = CQI_REPORT_INTERVAL

        self.episode_tti = episode_tti
        self.packet_list = np.random.uniform(
            1e3, 1e6, int(self.episode_tti*1000))
        self.rsrp_list = np.random.uniform(-120, -90,
                                           int(self.episode_tti*1000))

        self.user_list = []
        self.count_user = 0
        self.sim_time = 0.0

        # Here we calculate the sum of buffer of all true users
        self.all_buffer = 0

        # user_list which is scheduled in RBG
        self.select_user_list = []

        self.state_dim = RBG_NUM * \
            len(self.user_var['num'] + self.user_var['vec'] * RBG_NUM)
        self.state = np.zeros(
            (RBG_NUM, len(self.user_var['num'] + self.user_var['vec'] * RBG_NUM)))

        self.action_dim = RBG_NUM * (MAX_MCS - MIN_MCS + 1)

    def reset(self):
        self.__init__(self.ue_arrival_rate, self.episode_tti)
        self.fill_in_vir_users()
        self.add_new_user(must_add=True)
        self.calc_prior()
        self.select_user()
        return self.get_state()

    def get_user_by_id(self, uid):
        for user in self.user_list:
            if user.ID == uid:
                return user

    def update_user(self, user):
        uid = user.ID
        for i in range(len(self.user_list)):
            if self.user_list[i].ID == uid:
                self.user_list[i] = user
                break

    # define state from the perspective of user
    # def get_state(self):
    #     for i in range(len(self.user_list)):
    #         user = self.user_list[i]
    #         self.state[i] = [user.buffer, user.rsrp, user.avg_thp] + user.cqi + user.prior + user.sched_rbg
    #     return self.state.reshape(-1)

    def calc_prior(self):
        for i in range(len(self.user_list)):
            user = self.user_list[i]
            live_time = self.sim_time - user.arr_time
            if live_time % self.cqi_report_interval == 0.0:
                user.cqi = user.avg_snr + \
                    np.random.randint(-2, 2, size=RBG_NUM)
                user.cqi = np.clip(user.cqi, *user.attr_range['cqi'])

                # if user is virtual, then cqi is set to 0.
                if user.is_virtual:
                    user.cqi = np.zeros(RBG_NUM)

            user.mcs = copy.deepcopy(user.cqi)
            user.se = np.log2(1 + user.mcs ** 2.0)
            user.prior = user.se / max(1, user.avg_thp / PRIOR_THRESHOLD)
            self.user_list[i] = user

    def select_user(self):
        self.select_user_list = []
        # first we need to reset the schedule of user
        for i in range(len(self.user_list)):
            user = self.user_list[i]
            user.sched_rbg = np.zeros(RBG_NUM)
            self.update_user(user)

        # then schedule the user
        for rbg in range(RBG_NUM):
            max_prior = -1
            select_user = None
            for i in range(len(self.user_list)):
                user = self.user_list[i]
                if user.prior[rbg] > max_prior:
                    max_prior = user.prior[rbg]
                    select_user = user
            select_user.sched_rbg[rbg] = 1
            self.update_user(select_user)
            self.select_user_list.append(select_user)

    def get_state(self):
        for i in range(len(self.select_user_list)):
            select_user = self.select_user_list[i]
            self.state[i] = [select_user.rsrp, select_user.buffer,
                             select_user.avg_thp, select_user.cqi[i]]
        return minmax_scale(self.state, axis=0).reshape(-1)

    def add_new_user(self, must_add=False):
        if np.random.uniform(0., 1.) < self.ue_arrival_rate or must_add:
            self.count_user += 1
            user = User(self.count_user, self.sim_time, self.packet_list[self.count_user],
                        self.rsrp_list[self.count_user])
            if not self.user_list[len(self.user_list) - 1].is_virtual:
                self.user_list.append(user)
                self.all_buffer += user.buffer
                return

            for i in range(len(self.user_list)):
                if self.user_list[i].is_virtual:
                    self.user_list[i] = user
                    self.all_buffer += user.buffer
                    break

    def fill_in_vir_users(self):
        fill_count = max(0, RBG_NUM - len(self.user_list))
        for i in range(fill_count):
            self.user_list.append(User(-1, self.sim_time, 1, -122, True))

    def del_empty_user(self):
        self.user_list = list(filter(lambda x: x.buffer > 0, self.user_list))

    def take_action(self, mcs_list):
        reward = 0
        counted_user_list = set()

        for i in range(len(self.select_user_list)):
            user = self.select_user_list[i]
            if user in counted_user_list:
                continue
            counted_user_list.add(user)
            is_succ = 1 if (user.avg_snr + np.random.randint(-2,
                                                             2) - mcs_list[i]) > 0 else 0
            rbg_se = np.log2(1 + mcs_list[i] ** 2)
            rbg_tbs = int(BANDWIDTH * 0.9 *
                          user.sched_rbg.sum() / RBG_NUM * rbg_se * TTI)

            # if current buffer less than tbs: buffer set to 0, rbg_tbs set to buffer
            if rbg_tbs > user.buffer:
                rbg_tbs = user.buffer
                user.buffer = 0
            else:
                user.buffer -= rbg_tbs
            user.tbs_list.append(rbg_tbs)
            user.avg_thp = np.average(user.tbs_list)
            self.update_user(user)
            reward += is_succ * rbg_tbs
        return reward

    def get_current_users(self):
        all_users = 0
        for user in self.user_list:
            if user.ID != -1:
                all_users += 1
        selected_users = set()
        for user in self.select_user_list:
            selected_users.add(user)
        return all_users, len(selected_users)

    def step(self, action):
        self.sim_time += TTI

        # del user with empty buffer
        self.del_empty_user()

        # new user comes with probability
        self.add_new_user()

        # create virtual users to fill in user_list, this will be executed only when len(user_list)<RBG_NUM
        self.fill_in_vir_users()

        # calculate priority
        self.calc_prior()

        # select users for each RBG
        self.select_user()

        # take action
        action = action.reshape((RBG_NUM, MAX_MCS - MIN_MCS + 1))
        mcs_list = np.argmax(action, axis=-1)
        reward = self.take_action(mcs_list)

        done = int(self.sim_time) == int(self.episode_tti)

        next_state = self.get_state()

        # check current number true/selected users
        num_all_users, num_selected_users = self.get_current_users()

        return next_state, reward, done, self.all_buffer, num_all_users, num_selected_users, mcs_list

    def get_action(self):
        # reward by Huawei Policy, compare with the policy network we trained
        mcs_list = [np.floor(np.sum(ue.mcs * ue.sched_rbg) / ue.sched_rbg.sum()) for ue in
                    self.select_user_list]
        action = np.zeros((RBG_NUM, MAX_MCS - MIN_MCS + 1))
        for i in range(len(mcs_list)):
            action[i][int(mcs_list[i] - 1)] = 1
        action = action.reshape(RBG_NUM * (MAX_MCS - MIN_MCS + 1))

        return action
