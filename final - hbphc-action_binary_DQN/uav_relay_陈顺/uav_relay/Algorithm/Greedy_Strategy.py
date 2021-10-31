from environment import *


def greedy_choose(power,Action,h_user2bs1,h_user2uav):
    utility_j_set = []
    for jam_power in P_jam_set:
        sinr_user2bs1 = P_user * h_user2bs1 / (noise + jam_power * h_jam2bs1)
        sinr_user2uav = P_user * h_user2uav / (noise + jam_power * h_jam2uav)
        sinr_uav2bs2 = power * h_uav2bs2 / noise
        sinr_temp = max(sinr_user2bs1, min(sinr_user2uav, sinr_uav2bs2))
        utility = Action*sinr_temp - Action*C_u +(1-Action)*sinr_user2bs1
        utility_j = -utility - C_j * jam_power
        utility_j_set.append(utility_j)
    return utility_j_set.index(max(utility_j_set))
