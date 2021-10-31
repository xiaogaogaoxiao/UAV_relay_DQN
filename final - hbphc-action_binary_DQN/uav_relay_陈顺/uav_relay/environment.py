import numpy as np


P_user = 10
P_uav_set = np.linspace(1, 3, 20)
P_jam_set = np.linspace(0, 2, 20)
noise = 0.1
C_u = 1
C_j = 0.5
h_user2bs1_set = np.array([0.2,0.6])
h_user2uav_set = np.array([0.6,0.8])
h_uav2bs2 = 0.5
h_jam2bs1 = 0.4
h_jam2uav = 0.2
sinr_user2bs1_set = P_user * h_user2bs1_set[0] / (noise + P_jam_set * h_jam2bs1)
sinr_user2bs1_set = np.append(sinr_user2bs1_set,(P_user * h_user2bs1_set[1] / (noise + P_jam_set * h_jam2bs1)))
sinr_user2uav_set = P_user * h_user2uav_set[0] / (noise + P_jam_set * h_jam2uav)
sinr_user2uav_set = np.append(sinr_user2uav_set,(P_user * h_user2uav_set[1] / (noise + P_jam_set * h_jam2bs1)))

sinr_uav2bs2_set = 2 * h_uav2bs2 / noise


