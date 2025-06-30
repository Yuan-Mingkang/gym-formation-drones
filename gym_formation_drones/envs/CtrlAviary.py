import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
# CtrlAviary 继承自 BaseAviary 类，表示一个多无人机的仿真环境，专门用于控制任务。
class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

    ################################################################################
    # 这一段代码定义了 CtrlAviary 环境中的动作空间，具体是无人机的电机转速（RPM）指令的空间 
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        _actionSpace(self)：这是一个定义动作空间的方法，表示环境允许的所有可能的动作。
        方法返回的动作空间是一个 Box 对象，它代表了一个连续的空间。

        spaces.Box 表示一个可以容纳连续数值的空间，这里用于表示四个电机的转速（RPM）。每个无人机有四个电机，因此动作空间的维度是 (NUM_DRONES, 4)。

        """
        #### Action vector ######## P0            P1            P2            P3
        # act_lower_bound：这个变量定义了每个无人机每个电机转速的下限。这里每个无人机的电机转速的最小值是 0（即电机停止转动）。
        # np.array([[0., 0., 0., 0.] for i in range(self.NUM_DRONES)])：生成一个形状为 (NUM_DRONES, 4) 的数组，其中 NUM_DRONES 表示无人机的数量，每行表示一个无人机的四个电机的下限（都为 0）。
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        # act_upper_bound：这个变量定义了每个无人机每个电机转速的上限。self.MAX_RPM 是每个电机的最大转速。
        # np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])：生成一个形状为 (NUM_DRONES, 4) 的数组，每行表示一个无人机的四个电机的上限（都为 MAX_RPM）。
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        # spaces.Box：gym 中用来定义一个连续动作空间的类。在这里，low=act_lower_bound 和 high=act_upper_bound 定义了每个电机转速的最小值和最大值。
        # dtype=np.float32：动作空间的元素类型设置为 float32，即每个电机转速的数值是浮动的，采用 32 位浮点数表示。
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################
    # 这段代码定义了 CtrlAviary 环境中的观测空间，即无人机每个时间步产生的观测数据的范围和形状。
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).
        
        _observationSpace(self)：定义了一个方法来返回环境的观测空间。
        返回值是一个 spaces.Box 对象，它表示观测空间的维度和边界。

        返回的形状是 (NUM_DRONES, 20)，意味着每个无人机的观测数据是一个长度为 20 的向量。

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        # 观测向量的组成：每个无人机的观测向量包含 20 个元素，具体包括以下信息：
        #     位置 (X, Y, Z): 无人机的三维位置坐标。
        #     四元数 (Q1, Q2, Q3, Q4): 无人机的方向（旋转表示），使用四元数来表示。
        #     欧拉角 (R, P, Y): 无人机的姿态角（滚转、俯仰、偏航）。
        #     速度 (VX, VY, VZ): 无人机在三维空间中的线速度。
        #     角速度 (WX, WY, WZ): 无人机的角速度（绕各轴旋转的速率）。
        #     电机转速 (P0, P1, P2, P3): 四个电机的转速（RPM）。
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################
    # 这段代码定义了一个名为 _computeObs 的方法，用于计算并返回当前环境中每个无人机的观测状态。
    # 它会返回一个形状为 (NUM_DRONES, 20) 的 ndarray，其中每一行代表一个无人机的状态（包含 20 个值）。
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        _computeObs(self)：这是该类中的一个方法，负责返回环境中所有无人机的观测数据。
Returns：该方法返回一个 ndarray（即 NumPy 数组），它的形状为 (NUM_DRONES, 20)，即每个无人机的观测数据由 20 个值构成（前面已经定义过这些值的意义，如位置、速度、四元数等）。

        """
        # 这行代码通过 列表推导式 来遍历所有的无人机并获取它们的状态。
        # self._getDroneStateVector(i)：对每个无人机调用 _getDroneStateVector(i) 方法，获取第 i 个无人机的状态。这个方法（在别的地方定义）会返回一个包含该无人机所有状态信息的向量（一般为 20 个值，如位置、速度、姿态等）。
        # 使用 np.array(...) 将这些状态向量转换为一个 NumPy 数组。最终返回的是一个二维数组，其中每一行代表一个无人机的状态信息。
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################
    # 这段代码定义了一个名为 _preprocessAction 的方法，用于将传入的控制动作（action）处理为适合控制无人机电机转速（RPM）的数据。
    # 具体来说，它对每个无人机的控制命令进行裁剪，使其落在合理的范围内，并将动作转化为一个二维数组，最终返回这个数组。
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # np.array([...])：列表推导式生成一个新的二维数组，其中每个元素代表一个无人机的控制命令。这里的每一行对应一个无人机的 4 个电机的转速命令。
        #     action[i, :]：这是对第 i 个无人机的控制命令进行访问，action[i, :] 返回的是一个 1x4 的数组，代表该无人机的 4 个电机的转速命令。
        #     np.clip(action[i, :], 0, self.MAX_RPM)：np.clip() 函数用于将数组中的元素限制在指定的最小值和最大值之间。这里的作用是：
        #     最小值：0，表示电机转速不能为负值。
        #     最大值：self.MAX_RPM，表示每个电机的最大转速（通常会在类初始化时定义 MAX_RPM 的值，表示最大转速）。
        #     通过 np.clip，如果 action[i, :] 中的某个转速值超出了这个范围，它就会被限制在有效范围内。也就是说，如果某个命令超出了 [0, MAX_RPM] 的范围，它就会被“修剪”到这个范围。
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
