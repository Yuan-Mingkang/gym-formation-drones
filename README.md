### **整体流程概述**
代码用于处理多个CSV文件（每个文件代表一个无人机的轨迹数据），核心目标是为每个无人机生成密集、平滑的航点轨迹。流程包括：
1. **数据读取与初始化**：从CSV提取初始位置和原始航点。
2. **时间轴生成**：创建匹配控制频率的密集时间序列。
3. **三次样条插值**：生成连续、平滑的密集航点。
4. **动态曲率限制**：通过加速度阈值检测和平滑处理，防止轨迹突变。
5. **输出存储**：存储处理后的航点用于后续控制。

数学上，这是一个**轨迹优化问题**：输入离散的位置-时间数据，输出物理可行的连续轨迹，确保加速度不超过物理极限（如无人机动力学约束）。最终轨迹需满足：
- **位置连续性**（$`C^0`$连续）。
- **速度与加速度平滑性**（$`C^1`$和$`C^2`$连续）。
- **加速度有界**（如≤2g）。

---

### CSV文件通过Blender Skybrush获取
<img src="/gym_formation_drones/assets/CSV.gif" alt="formation flight" width="1000"> 

### **pid_circle控制示例**
```sh
cd gym_pybullet_drones/examples/
python3 pid_circle.py
```
<img src="/gym_formation_drones/assets/pid_circle.gif" alt="formation flight" width="1000"> 

### **分步骤数学公式与原理**
#### **1. 数据读取与初始化**
```python
df = pd.read_csv(file_path)
initial_pos = (df[["x [m]", "y [m]", "z [m]"]].iloc[0].values) / 6.0
raw_waypoints = (df[["x [m]", "y [m]", "z [m]"]].iloc[1:].values) / 6.0
```

- **数学公式**：
  - 初始位置：$`\mathbf{p}_0 = \frac{1}{6} \mathbf{d}_0`$，其中 $`\mathbf{d}_0 = [x_0, y_0, z_0]^T`$ 是CSV第一行的坐标。
  - 原始航点：$`\mathbf{p}_i = \frac{1}{6} \mathbf{d}_i`$（$`i = 1, 2, \ldots, n-1`$)，$`n`$为数据行数，$`\mathbf{d}_i`$为第$`i`$行坐标。
  - 时间戳：$`t_j = \frac{\text{Time}_j}{1000}`$（$`j = 1, 2, \ldots, n-1`$)，单位为秒。
- **原理**：
  - `pd.read_csv`将CSV文件解析为DataFrame，支持表格化数据处理。
  - `iloc[0]`选择第一行（索引0），`iloc[1:]`选择第二行至末尾，`.values`转换为NumPy数组。除以6.0可能是单位转换（如从毫米到米）。
- **作用**：提取离散航点作为插值基础，确保数据可操作。初始位置用于无人机初始化，后续航点定义轨迹形状。

#### **2. 密集时间轴生成**
```python
time_dense = np.linspace(start_time, end_time, int(control_freq_hz * duration_sec))
```

- **数学公式**：
  - $`t_k = t_0 + k \cdot \Delta t`$，其中 $`\Delta t = \frac{1}{f_{\text{ctrl}}}`$，$`k = 0, 1, \ldots, m-1`$，$`m = f_{\text{ctrl}} \times T`$。
  - $`t_0`$是第一个航点时间，$`T`$是总时长（`duration_sec`），$`f_{\text{ctrl}}`$是控制频率（`control_freq_hz`）。
- **原理**：生成等间隔时间序列，步长$`\Delta t`$由控制频率决定。例如，若$`f_{\text{ctrl}} = 100\,\text{Hz}`$，则$`\Delta t = 0.01\,\text{s}`$。
- **作用**：为插值提供高分辨率时间网格，匹配无人机控制器的更新频率，确保轨迹可实时跟踪。

#### **3. 三次样条插值生成密集航点**
```python
cs_x = CubicSpline(time_raw, raw_waypoints[:, 0])
dense_waypoints = np.column_stack((cs_x(time_dense), cs_y(time_dense), cs_z(time_dense)))
```

- **数学公式**：
  - 对每个坐标轴（x, y, z）构建分段三次多项式：
    $$`
    s_x(t) = a_i (t - t_i)^3 + b_i (t - t_i)^2 + c_i (t - t_i) + d_i \quad \text{for} \quad t \in [t_i, t_{i+1}]
    `$$

系数$`a_i, b_i, c_i, d_i`$通过以下条件求解：
- **插值条件**：$`s_x(t_j) = p_{j,x}`$（位置匹配）。
- **连续性条件**：$`s_x \in C^2[t_{\min}, t_{\max}]`$（一阶导数$`s_x'`$和二阶导数$`s_x''`$连续）。
  - 密集航点：$`\mathbf{p}_k = [s_x(t_k), s_y(t_k), s_z(t_k)]^T`$。
- **原理**：
  - 三次样条插值保证轨迹光滑（加速度连续），优于线性插值或多项式插值（后者易振荡）。`CubicSpline`默认使用"not-a-knot"边界条件，避免端点突变。
  - 误差收敛速度为$`O(h^4)`$（$`h`$为节点间距），高精度适合动力学仿真。
- **作用**：将稀疏航点转化为连续轨迹，确保无人机运动平滑（减少急停或抖动），同时保留原始路径形状。

#### **4. 速度、加速度计算与动态曲率限制**
```python
velocity = np.gradient(dense_waypoints, axis=0) * control_freq_hz
acceleration = np.gradient(velocity, axis=0) * control_freq_hz
acc_magnitude = np.linalg.norm(acceleration, axis=1)
for idx in np.where(acc_magnitude > max_acc)[0]:
    dense_waypoints[idx] = 0.5 * (dense_waypoints[idx - 1] + dense_waypoints[idx + 1])
```

- **数学公式**：
  - **速度计算**：
    $$`
    \mathbf{v}_k = \frac{\Delta \mathbf{p}}{\Delta t} \approx \frac{\mathbf{p}_{k+1} - \mathbf{p}_{k-1}}{2 \Delta t} \quad \text{(中心差分)}
    `$$

其中$`\Delta t = 1 / f_{\text{ctrl}}`$。
  - **加速度计算**：
    $$`
    \mathbf{a}_k = \frac{\Delta \mathbf{v}}{\Delta t} \approx \frac{\mathbf{v}_{k+1} - \mathbf{v}_{k-1}}{2 \Delta t}
    `$$

  - **加速度大小**：$`a_k = \|\mathbf{a}_k\| = \sqrt{a_{x,k}^2 + a_{y,k}^2 + a_{z,k}^2}`$。
  - **平滑处理**（当$`a_k > 2g`$）：
    $$`
    \mathbf{p}_k \leftarrow \frac{1}{2} (\mathbf{p}_{k-1} + \mathbf{p}_{k+1})
    `$$

其中$`g = 9.81\,\text{m/s}^2`$。
- **原理**：
  - `np.gradient`使用中心差分（内部点）和单侧差分（边界）计算梯度，精度为$`O(\Delta t^2)`$。乘以$`f_{\text{ctrl}}`$将离散差分转为物理导数（因$`\Delta t = 1/f_{\text{ctrl}}`$)。
  - 加速度阈值（2g）源于工程实践：
- 超过2g的加速度可能导致无人机失稳或结构损伤。
- 平滑处理通过局部平均降低曲率突变，等效于低通滤波，确保动力学可行性。
- **作用**：
  - **动态曲率限制**：加速度与轨迹曲率$`\kappa`$相关（$`\kappa \propto \|\mathbf{a}\|`$)，限制$`a_k \leq 2g`$防止侧滑或翻车（尤其在高速转弯时）。
  - **实时修正**：处理后的轨迹满足$`\max \|\mathbf{a}\| \leq 2g`$，符合安全标准。

#### **5. 输出存储**
```python
INIT_XYZS[i] = initial_pos
TARGET_POS.append(dense_waypoints)
```

- **数学表示**：存储初始位置$`\mathbf{p}_0`$和修正后的密集航点序列$`\{\mathbf{p}_k\}_{k=0}^{m-1}`$。
- **作用**：为每个无人机提供可行轨迹，用于后续控制算法（如PID或模型预测控制）。

---

### **整体原理与作用**
#### **原理**
- **轨迹平滑性**：三次样条插值确保位置、速度、加速度连续（$`C^2`$连续），避免不连续点导致的控制振荡。
- **物理可行性**：
  - 加速度阈值（2g）基于无人机动力学极限，防止电机饱和或失稳。
  - 动态曲率限制通过局部平滑处理，保证轨迹曲率有界（$`\kappa \leq \kappa_{\max}`$)，符合最小转弯半径约束。
- **数值稳定性**：`np.gradient`使用自适应差分（边界用一阶，内部用二阶），避免数值发散。

#### **作用**
1. **轨迹优化**：将粗糙的离散航点转化为高分辨率、平滑的轨迹，提升跟踪精度。
2. **安全保证**：通过加速度限制，确保轨迹在物理约束内（如最大推力），防止事故。
3. **实时性**：插值和平滑处理计算高效，适合嵌入式系统或仿真。
4. **通用性**：适用于多无人机协同（`file_list`遍历），可扩展至机器人路径规划。

#### **工程背景**
- **无人机/车辆轨迹规划**：代码类似自动驾驶中的轨迹优化，其中曲率限制防止高速侧滑，加速度阈值确保乘客舒适性。
- **信号处理**：加速度阈值法源于小波去噪和冲击检测（如2g阈值广泛用于惯性导航）。
- **控制理论**：密集航点匹配控制频率（如100Hz），避免欠采样导致的跟踪误差。

   
