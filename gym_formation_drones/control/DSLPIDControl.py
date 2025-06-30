import math
import numpy as np
import pybullet as p
# 用于处理旋转矩阵和四元数转换。
from scipy.spatial.transform import Rotation

from gym_formation_drones.control.BaseControl import BaseControl
from gym_formation_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        # P_COEFF_FOR、I_COEFF_FOR 和 D_COEFF_FOR 分别是位置控制的比例、积分和微分系数。
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        # 用于姿态控制的比例、积分和微分系数。
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # 定义从 PWM 转换到 RPM 的比例和常数，以及 PWM 的最小值和最大值。
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        # 根据不同的无人机模型（CF2X 或 CF2P），定义不同的混合矩阵。混合矩阵用于将各个电机的力矩和推力转换为各个电机的控制信号（PWM）。
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################
    # 重置控制类的状态，包括位置误差、姿态误差以及它们的积分项。
    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    # 该方法的目标是通过计算位置和姿态的 PID 控制输入，生成适用于四个电机的 RPM 输出
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.控制计算的时间步长，单位为秒，表示控制更新的频率。
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.当前无人机的位置坐标，格式为 [x, y, z]。
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.当前无人机的姿态，以四元数表示，格式为 [w, x, y, z]。
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.当前无人机的线速度，格式为 [vx, vy, vz]。
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.当前无人机的角速度，格式为 [p, q, r]，但在该函数中未被使用。
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.目标位置，格式为 [x_target, y_target, z_target]。
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.目标姿态，以欧拉角表示，格式为 [roll, pitch, yaw]。
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.目标速度，格式为 [vx_target, vy_target, vz_target]。
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.目标姿态速率（目标欧拉角的变化率），格式为 [roll_rate, pitch_rate, yaw_rate]。

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.四个电机的目标 RPM（转速），每个值代表一个电机的转速。
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.当前位置误差（目标位置与当前实际位置的差异），格式为 [x_e, y_e, z_e]。
        float
            The current yaw error.当前的偏航误差，即目标偏航角与当前实际偏航角的差异。

        """
        # 增加了一个控制计数器，通常用于跟踪控制周期的次数。
        self.control_counter += 1
        # 这一行调用了 _dslPIDPositionControl 函数，计算基于目标位置、目标速度等输入的 PID 控制，并返回：
        #     thrust：所需的推力值（沿着无人机 Z 轴）。
        #     computed_target_rpy：通过位置控制计算的目标姿态（欧拉角形式）。
        #     pos_e：当前位置误差（目标位置与当前实际位置的差异）。
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        # 这一行调用了 _dslPIDAttitudeControl 函数，计算基于目标姿态（包括目标欧拉角、目标姿态速率等）的 PID 控制，并返回：
        #     rpm：目标 RPM（每个电机的转速），这是最终控制信号。
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        # 该行将当前四元数 cur_quat 转换为欧拉角（roll, pitch, yaw），方便计算偏航误差。
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        # 返回三个值：
        #     rpm：四个电机的目标 RPM。
        #     pos_e：当前位置误差（目标位置与实际位置的差异）。
        #     yaw_error：当前偏航误差，即目标偏航（computed_target_rpy[2]）与当前偏航（cur_rpy[2]）之间的差异。
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################
    # 根据当前状态（位置、速度、姿态）和目标状态（位置、速度、姿态）计算无人机所需的目标推力和目标姿态（滚转、俯仰、偏航）。
    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.目标推力，沿无人机的 z 轴方向。这是一个标量，表示需要施加的推力大小。
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.目标姿态，表示为滚转、俯仰、偏航角度（单位：弧度）。
        float
            The current position error.当前的位移误差，计算方法是目标位置减去当前的位置。

        """
        # 旋转矩阵的计算:
        # 将当前的四元数 cur_quat 转换为 3x3 的旋转矩阵 cur_rotation，该矩阵用于将向量（如推力向量）从机体坐标系转换到世界坐标系。
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # 位置误差和速度误差：
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        # 积分项：位置误差的积分项是用来消除稳态误差的。每次控制周期会根据误差进行积分累加，控制时长 control_timestep 用来决定每次累加的步长。
        # 限制积分值：为了避免积分值过大，积分项会被限制在一定范围内，尤其是 z 轴的积分项被限制在 [-0.15, 0.15] 范围内。
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        # 这是标准的 PID 控制公式：
        #     P（比例）：位置误差 pos_e 的比例。
        #     I（积分）：位置误差的积分项 self.integral_pos_e。
        #     D（微分）：速度误差 vel_e。
        # 还加上了一个重力项 self.GRAVITY，使得控制能够适应重力的影响。
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        # 标量推力 (scalar_thrust)：通过计算目标推力与旋转矩阵的 z 轴分量的点积，得到沿 z 轴的有效推力分量。
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        # 推力转换：根据推力的大小，使用某些常数将推力值转换为 PWM 信号，并最终转换为 RPM（转速）。
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # 目标轴：通过目标推力的方向来计算无人机的三个轴（x, y, z）。首先，z 轴是目标推力的单位向量；然后，基于目标姿态的偏航角（target_rpy[2]），计算出 x 轴和 y 轴，保证它们与 z 轴正交。
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        # 通过旋转矩阵 target_rotation，计算出目标姿态的欧拉角（滚转、俯仰、偏航），并确保其在 [-π, π] 范围内。
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        # 这一步是为了确保计算出来的欧拉角（roll, pitch, yaw）都在合法范围 [-π, π] 之内。如果超出了范围，打印错误信息。
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################
    # 实现了一个PID姿态控制器的核心部分，用于调整四旋翼无人机的姿态控制。具体来说，
    # 这个函数计算的是给定目标姿态（目标欧拉角）和目标旋转速率（目标的滚转、俯仰和偏航角速率）下，如何通过PID控制计算出每个电机的转速（RPM）。
    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.目标的推进力，通常沿着无人机的Z轴方向（垂直方向）。
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.当前无人机的四元数表示的姿态（单位四元数）。
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.目标的欧拉角（滚转、俯仰、偏航），用来表示目标的旋转。
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.目标的滚转、俯仰和偏航角速率。

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # 从四元数到旋转矩阵： 首先，当前姿态通过四元数转换为旋转矩阵，p.getMatrixFromQuaternion(cur_quat) 将当前四元数 cur_quat 转换为旋转矩阵 cur_rotation。
        # 同样，目标姿态 target_euler 被转换为目标四元数，并进一步转换为目标旋转矩阵 target_rotation。
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        # 旋转误差： 然后，通过计算旋转误差矩阵 rot_matrix_e 来得到当前旋转矩阵与目标旋转矩阵之间的差异。
        # 这个误差反映了目标姿态与当前姿态之间的旋转差异。通过提取旋转矩阵中的元素，可以得到三个旋转误差分量：
        # 这些误差代表了滚转、俯仰和偏航角的误差，rot_e 即为这三个角度的误差。
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        # 姿态角速率误差： 目标的姿态角速率 target_rpy_rates 是期望的滚转、俯仰、偏航角的变化速率。然后，通过比较当前姿态角与目标姿态角的差异，计算出姿态角速率的误差 rpy_rates_e。
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        # 积分误差： 对姿态误差进行积分，得到控制器的积分项，self.integral_rpy_e 存储了姿态误差的历史累计值。这部分帮助控制器消除长期存在的小偏差。
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        # PID控制： 使用PID控制器计算目标的转矩。根据姿态误差、姿态角速率误差和积分误差，计算出每个轴（滚转、俯仰和偏航）的转矩：
        # target_torques 代表了每个轴（滚转、俯仰和偏航）对应的控制力矩。
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        # PWM控制信号： 计算得到的目标转矩与推力 thrust 一起通过混合矩阵 self.MIXER_MATRIX 进行加权，以得到控制每个电机的PWM信号。
        # 最后，通过 self.PWM2RPM_SCALE 和 self.PWM2RPM_CONST 将PWM信号转换为电机转速（RPM）。
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        # 最终，返回的是根据PID控制器计算得到的四个电机的转速（RPM）。
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################
    # 定义了一个实用函数 _one23DInterface，用于将不同维度的推力输入（1D、2D、3D推力输入）转化为四旋翼每个电机的PWM值。
    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.
        一个数组，表示希望输入的推力值。根据不同维度的输入，它可以有以下几种形态：
            1D推力（长度为1）：所有电机获得相同的推力。
            2D推力（长度为2）：推力输入将被分配到相对的电机对，通常用于更简单的飞行控制。
            4D推力（长度为4）：每个电机有不同的推力值，用于更复杂的控制策略。
        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            返回一个长度为4的数组，表示给定推力输入下的每个电机的PWM值，这些值用于控制电机的转速。
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        # 推力输入维度判断： 该函数首先通过 DIM = len(np.array(thrust)) 确定输入推力的维度（即输入数组的长度）。根据 DIM 的不同值，函数处理方式有所不同。
        DIM = len(np.array(thrust))
        # PWM值计算： 接下来，推力值 thrust 被用于计算PWM值。计算公式为：
        # 该公式将推力值转换为电机控制信号（PWM）。其中：
        #     self.KF 是推力常数，用来将推力转换为电机的旋转量。
        #     self.PWM2RPM_CONST 是一个常数，用于将PWM信号转换为RPM信号。
        #     self.PWM2RPM_SCALE 是另一个常数，用于调节PWM值的比例。
        #     np.sqrt(np.array(thrust)) 通过开方的方式对输入的推力值进行转换，这是因为推力和电机速度之间的关系是平方关系。
        # 最终，np.clip 会确保PWM值在 self.MIN_PWM 和 self.MAX_PWM 的范围内。
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        # 不同维度的推力处理：
        #     1D推力：如果输入推力维度是1（即 DIM == 1），则所有四个电机会获得相同的PWM值。这是通过 np.repeat(pwm, 4/DIM) 实现的，4/DIM 为4（电机数量）除以输入推力的维度（1），意味着将相同的PWM值复制4次。
        #     2D推力：如果输入推力是2D（即 DIM == 2），则推力会被分配到两个相对的电机上。具体来说，PWM值的前两个和后两个分别赋值给两个电机对，并且后两个电机的PWM值是前两个的翻转（np.flip(pwm)）。
        #     4D推力：如果输入推力维度是4（即 DIM == 4），则每个电机的PWM值都可以由输入推力直接计算。通过 np.repeat(pwm, 4/DIM)，每个电机都有不同的PWM值。
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
