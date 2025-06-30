# 这段代码实现了一个多无人机仿真，结合了 CtrlAviary 环境（用于飞行模拟）和 DSLPIDControl 控制器（PID 控制器），
# 模拟多个无人机执行圆形轨迹飞行，并进行实时控制、日志记录、可视化和结果保存

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pandas as pd
import formation as p
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from gym_formation_drones.utils.enums import DroneModel, Physics
from gym_formation_drones.envs.CtrlAviary import CtrlAviary
from gym_formation_drones.control.DSLPIDControl import DSLPIDControl
from gym_formation_drones.utils.Logger import Logger
from gym_formation_drones.utils.utils import sync, str2bool
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
DEFAULT_OBSTACLES = False
# 仿真频率
DEFAULT_SIMULATION_FREQ_HZ = 240
# 控制频率
DEFAULT_CONTROL_FREQ_HZ = 48
# 仿真持续时间
DEFAULT_DURATION_SEC = 100
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

    INIT_XYZS = np.zeros((DEFAULT_NUM_DRONES, 3))  # 30个无人机，每个初始位置为3D坐标
    INIT_RPYS = np.zeros((DEFAULT_NUM_DRONES, 3))
    TARGET_POS = []  # 每个元素为无人机对应的航点数组（形状：n×3）
    # 读取文件夹并处理文件
    folder_path = "Drones_csv"
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("Drone ") and f.endswith(".csv")],
        key=lambda x: int(x.split()[1].split(".")[0])  # 按文件名中的数字排序
    )

    # 遍历每个文件
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取初始位置（第一行数据）
        initial_pos = (df[["x [m]", "y [m]", "z [m]"]].iloc[0].values) / 6.0
        # print(initial_pos)
        INIT_XYZS[i] = initial_pos

        # 提取后续航点（从第二行开始）
        raw_waypoints = (df[["x [m]", "y [m]", "z [m]"]].iloc[1:].values) / 6.0
        # 获取原始时间戳（单位：秒）
        time_raw = df["Time [msec]"].iloc[1:].values / 1000.0  # 从第二行开始
        # print(waypoints)
        # 生成密集时间轴（匹配控制频率）
        start_time = time_raw[0]
        end_time = start_time + duration_sec  # 根据仿真总时长生成
        time_dense = np.linspace(start_time, end_time,
                                 int(control_freq_hz * duration_sec))

        # 三次样条插值（x,y,z分别插值）
        cs_x = CubicSpline(time_raw, raw_waypoints[:, 0])
        cs_y = CubicSpline(time_raw, raw_waypoints[:, 1])
        cs_z = CubicSpline(time_raw, raw_waypoints[:, 2])

        # 生成密集航点
        dense_waypoints = np.column_stack((
            cs_x(time_dense),
            cs_y(time_dense),
            cs_z(time_dense)
        ))

        # 动态曲率限制（防止突变）
        velocity = np.gradient(dense_waypoints, axis=0) * control_freq_hz
        acceleration = np.gradient(velocity, axis=0) * control_freq_hz
        acc_magnitude = np.linalg.norm(acceleration, axis=1)

        # 加速度超过2g的航点平滑处理
        max_acc = 2 * 9.81  # 2g加速度阈值
        for idx in np.where(acc_magnitude > max_acc)[0]:
            if idx > 0 and idx < len(dense_waypoints) - 1:
                dense_waypoints[idx] = 0.5 * (dense_waypoints[idx - 1] + dense_waypoints[idx + 1])

        TARGET_POS.append(dense_waypoints)
        if i == DEFAULT_NUM_DRONES-1:
            break
    # print("无人机0的第1个航点:", TARGET_POS[0])
    #### Initialize a circular trajectory ######################
    # 设置轨迹的周期。
    PERIOD = 40
    # 根据控制频率计算目标轨迹的点数
    NUM_WP = control_freq_hz*PERIOD

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

    #### Obtain the formation Client ID from the environment ####
    # 获取 formation 客户端的 ID，以便后续操作。
    PYB_CLIENT = env.getformationClient()

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
                                                                    target_pos=np.hstack(TARGET_POS[j][i]),
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        # 这段代码控制每个无人机的目标位置进度。wp_counters[j] 是无人机 j 当前的目标位置索引（waypoint）。
        # 如果当前的目标位置索引没有到达最后一个位置（wp_counters[j] < NUM_WP - 1），则将其加 1，表示前进到下一个位置。
        # 如果已经到达最后一个位置（wp_counters[j] == NUM_WP - 1），则重置为 0，表示回到起点，形成循环。
        # for j in range(num_drones):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

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
                       control=np.hstack([TARGET_POS[j][i], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        # 调用 env.render() 用于渲染环境，通常是显示图形界面来观察仿真过程。这个操作会显示在 formation 的 GUI 窗口中（如果启用了 GUI）。
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
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use formation GUI (default: True)', metavar='')
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
