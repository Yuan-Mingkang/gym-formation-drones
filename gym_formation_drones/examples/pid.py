# 这段代码实现了一个多无人机仿真，结合了 CtrlAviary 环境（用于飞行模拟）和 DSLPIDControl 控制器（PID 控制器），
# 模拟多个无人机执行圆形轨迹飞行，并进行实时控制、日志记录、可视化和结果保存
"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
# 无人机模型
DEFAULT_DRONES = DroneModel("cf2x")
# 无人机数量
DEFAULT_NUM_DRONES = 30
# 物理引擎
DEFAULT_PHYSICS = Physics("pyb")
# 是否显示图形界面
DEFAULT_GUI = True
# 是否录制视频
DEFAULT_RECORD_VISION = False
# 是否绘制仿真结果
DEFAULT_PLOT = True
# 是否在 GUI 中显示调试信息
DEFAULT_USER_DEBUG_GUI = False
# 是否添加障碍物，默认为 True。
DEFAULT_OBSTACLES = True
# 仿真频率
DEFAULT_SIMULATION_FREQ_HZ = 240
# 控制频率
DEFAULT_CONTROL_FREQ_HZ = 48
# 仿真持续时间
DEFAULT_DURATION_SEC = 12
# 仿真结果保存文件夹
DEFAULT_OUTPUT_FOLDER = 'results'
# 是否运行于 Colab 环境
DEFAULT_COLAB = False
# run 函数接受多个参数，其中包括仿真配置（如无人机模型、数量、控制频率等），这些配置控制仿真行为。
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    # H, H_STEP, R: 设置初始高度、步长和圆形轨迹半径。
    H = .1
    H_STEP = .05
    R = .3
    # INIT_XYZS 是一个包含 num_drones 个无人机初始位置的数组，每行是一个无人机的 [x, y, z] 坐标。
    # 这些坐标会将所有的无人机分布在一个半径为 R 的圆上，并且每个无人机有不同的高度（H + i * H_STEP），确保它们在 Z 轴上是逐层分布的。
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    # INIT_RPYS: 为每个无人机设置初始的滚转、俯仰、偏航角（RPY）。
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    # 设置轨迹的周期。
    PERIOD = 10
    # 根据控制频率计算目标轨迹的点数
    NUM_WP = control_freq_hz*PERIOD
    # 为每个时刻计算目标位置，形成一个圆形轨迹。
    # TARGET_POS：这是一个 NUM_WP × 3 的数组，NUM_WP 是目标位置点的数量，通常是根据控制频率和飞行周期来计算的（即：目标轨迹的总时间步数）。
    # 数组的每一行代表一个目标位置，每行包含 3 个值，分别表示目标位置的 x、y 和 z 坐标。初始化时，所有位置设置为 (0, 0, 0)。
    TARGET_POS = np.zeros((NUM_WP,3))
    # 这段代码是计算每个目标位置 TARGET_POS[i, :] 的具体坐标。
    #     R：目标轨迹的半径，表示无人机飞行的圆形轨迹半径。
    #     i / NUM_WP * (2 * np.pi)：这一项计算了目标轨迹在圆周上的当前角度。i 是目标点的索引，NUM_WP 是目标点的总数，所以这个公式会将目标点均匀分布在圆周上。
    #     np.pi/2：这是一个偏移量，确保目标轨迹在正确的起始角度（通常是沿着 X 轴方向开始）。
    #     np.cos 和 np.sin：这些三角函数分别计算圆形轨迹中每个点的 X 和 Y 坐标。
    #     INIT_XYZS[0, 0] 和 INIT_XYZS[0, 1]：这些是第一个无人机的初始 X 和 Y 坐标，用作目标轨迹的基准点。
    #     R*np.sin(...) - R + INIT_XYZS[0, 1]：这个公式计算了 Y 轴上的目标位置，并包含了一个偏移量 -R，确保 Y 坐标在起始位置时是正确的。
    #     最后，0 用于设置目标位置的高度 Z 坐标，意味着所有的目标位置都位于同一高度，假设轨迹是在 X-Y 平面上。
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    # wp_counters：这个数组用于记录每个无人机当前的目标点（waypoint）。数组的长度为 num_drones，表示每个无人机的目标点索引。
    #     (i * NUM_WP / 6) % NUM_WP：这是计算每个无人机起始目标点的索引。i 是无人机的编号，NUM_WP 是目标位置的总数。通过 (i * NUM_WP / 6) 使得每个无人机从不同的目标点开始，并且每个无人机的目标点索引按照 NUM_WP 循环。这样做的目的是让无人机的飞行轨迹不完全重合。
    #     比如，i = 0 时，第一个无人机的起始点可能是第一个目标点；i = 1 时，第二个无人机的起始点可能是第六个目标点，等等。
    #     % NUM_WP 确保了目标点索引在 [0, NUM_WP-1] 范围内循环。
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    # 使用 CtrlAviary 环境初始化仿真。传入的参数包括无人机模型、数量、初始位置、物理引擎等。
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    # 获取 PyBullet 客户端的 ID，以便后续操作。
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    # 初始化一个日志记录器，用于记录每个时间步的状态、控制命令等信息。
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    # 根据无人机模型创建多个 PID 控制器。
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    # 初始化每个无人机的动作为零向量。
    action = np.zeros((num_drones,4))
    # 使用 time.time() 获取当前时间，方便计算仿真运行时间。
    START = time.time()
    # 这段代码的主要作用是在模拟环境中控制和更新无人机的位置和状态。它通过不断地调用 env.step(action) 来推进模拟，
    # 并根据控制器计算出的动作来调整无人机的状态。每一行都涉及到了模拟的具体步骤，我们逐步解析它：
    # duration_sec 是模拟的持续时间（秒），env.CTRL_FREQ 是控制频率（每秒的控制步骤数）。
    # int(duration_sec * env.CTRL_FREQ) 计算了模拟的总控制步骤数（即循环执行的次数）。假设控制频率是 48 Hz，并且模拟持续 12 秒，那么循环会运行 12 * 48 = 576 次。
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        # env.step(action) 负责推进仿真一步。它返回：
        #     obs: 无人机的当前观测数据（如位置、速度等状态信息）。
        #     reward: 当前步骤的奖励，表示无人机在当前状态下的表现。
        #     terminated: 一个布尔值，表示模拟是否已经结束。
        #     truncated: 一个布尔值，表示模拟是否由于超时等原因被截断。
        #     info: 包含其他额外信息的字典，通常包括调试信息或其他有用的状态数据。
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            # 这是每个无人机控制器（如 DSLPIDControl）根据当前的状态计算出所需的控制命令（如速度、力矩等）。
            # control_timestep=env.CTRL_TIMESTEP：控制时间步长，表示每次控制计算的时间间隔。
            # state=obs[j]：无人机的当前状态（来自 env.step(action) 的 obs）。
            # target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]])：目标位置，由圆形轨迹 TARGET_POS 和初始高度 INIT_XYZS[j, 2] 组成。目标位置是一个 [x, y, z] 的三维坐标。
            # target_rpy=INIT_RPYS[j, :]：目标滚转角、俯仰角和偏航角（由 INIT_RPYS 提供）。
            # action[j, :] 是当前计算出的控制命令，通常是四个值，对应无人机的油门、滚转、俯仰和偏航角度。
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        # 这段代码控制每个无人机的目标位置进度。wp_counters[j] 是无人机 j 当前的目标位置索引（waypoint）。
        # 如果当前的目标位置索引没有到达最后一个位置（wp_counters[j] < NUM_WP - 1），则将其加 1，表示前进到下一个位置。
        # 如果已经到达最后一个位置（wp_counters[j] == NUM_WP - 1），则重置为 0，表示回到起点，形成循环。
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        # 记录每个无人机的状态和控制数据：
        # logger.log(...) 将数据记录到日志中。数据包括：
        # drone=j: 当前无人机的索引。
        # timestamp=i / env.CTRL_FREQ: 当前时间戳（以秒为单位）。
        # state=obs[j]: 当前状态（位置、速度等）。
        # control=np.hstack(...): 控制命令，包括目标位置和目标姿态（例如目标位置 TARGET_POS、初始高度 INIT_XYZS[j, 2] 和姿态 INIT_RPYS[j, :]）。
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        # 调用 env.render() 用于渲染环境，通常是显示图形界面来观察仿真过程。这个操作会显示在 PyBullet 的 GUI 窗口中（如果启用了 GUI）。
        env.render()

        #### Sync the simulation ###################################
        # sync(i, START, env.CTRL_TIMESTEP) 是一个用于确保仿真与可视化的时间同步的函数。
        # 它使得每一帧的渲染与仿真步进保持一致，特别是在启用了 GUI 的情况下，确保仿真进程与显示进度一致。
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
