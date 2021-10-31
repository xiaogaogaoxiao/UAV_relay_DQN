import numpy as np

fai = 0.9
Vmax = 5
v_set = np.array([0,1,2,3,4,Vmax])
p_v_set = np.array([0.02,0.04,0.2,0.3,0.4,0.04])
h_user2bs1_set =np.array([0.2,0.6])
p_h_user2bs1_set = np.array([0.6,0.4])

h_user2uav_set = np.array([0.6,0.8])
p_h_user2uav_set = np.array([0.3,0.7])


def gain_obtain_h_user2bs1(v_obu,h):
    p = fai * v_obu / Vmax
    p_h_user2bs1_set = np.array([[1-p,p],[p,1-p]])
    position_now = np.where(h == h_user2bs1_set)
    temp_p = np.random.random()
    if temp_p < p_h_user2bs1_set[position_now,0]:
        h_now = 0.2
    else:
        h_now = 0.6
    return h_now

def gain_obtain_h_user2uav(v_obu,h):
    p = fai * v_obu / Vmax
    p_h_user2uav_set = np.array([[1-p,p],[p,1-p]])
    position_now = np.where(h == h_user2uav_set)
    temp_p = np.random.random()
    if temp_p < p_h_user2uav_set[position_now,0]:
        h_now = 0.6
    else:
        h_now = 0.8
    return h_now







