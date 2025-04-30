import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
import copy
#from Draw_dynamic_cdf import Draw_dynamic_cdf


import rospy
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
import tf
import math
import pymap3d
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from threading import Lock
from scipy.spatial.transform import Rotation as R

class CBFMPCController:
    def __init__(self):
        rospy.init_node('CBF_MPC', anonymous=True)
        # 系统参数初始化
        self.initialize_system_parameters()
        # MPC参数初始化
        self.initialize_mpc_parameters()
        # 环境设置初始化
        self.initialize_environment()
        # 构建动力学模型
        self.build_dynamics_model()
        # 设置优化问题
        self.setup_optimization_problem()


        self.t = np.arange(0, self.time_total+self.dt, self.dt)
        self.mpciter = 0
        # 初始化控制输入和状态预测
        self.u0 = np.zeros((self.N, self.n_controls))
        self.X0 = np.tile(self.x0.T, (self.N+1, 1))
        self.C_0 = np.ones(self.N).reshape(1, -1) * self.C_t
        # 存储历史数据
        self.t0 = 0
        self.xx1 = []
        self.u_cl = []
        self.u_cl1 = []
        self.su1 = 0.75
        self.solve_times = []
        self.obs1_log = []
        self.x = []
        self.y = []
        self.data_lock1 = Lock()
        

        rospy.loginfo('MPC initialized')
        rospy.Subscriber("/wamv1/sensors/position/p3d_wamv", Odometry, self.odomCallback1)
        #rospy.Subscriber("/wamv2/sensors/position/p3d_wamv", Odometry, self.odomCallback2)
        # rospy.Subscriber("/wamv3/sensors/position/p3d_wamv", Odometry, self.odomCallback3)

        self.control_left_pub = rospy.Publisher('/wamv1/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.control_conter_pub = rospy.Publisher('/wamv1/thrusters/lateral_thrust_cmd', Float32, queue_size=10)
        self.control_right_pub = rospy.Publisher('/wamv1/thrusters/right_thrust_cmd', Float32, queue_size=10)
        

        # self.control_left_pub2 = rospy.Publisher('/wamv2/thrusters/left_thrust_cmd', Float32, queue_size=10)
        # self.control_right_pub2 = rospy.Publisher('/wamv2/thrusters/right_thrust_cmd', Float32, queue_size=10)

        # self.control_left_pub3 = rospy.Publisher('/wamv3/thrusters/left_thrust_cmd', Float32, queue_size=10)
        # self.control_right_pub3 = rospy.Publisher('/wamv3/thrusters/right_thrust_cmd', Float32, queue_size=10)

        self.u_pub = rospy.Publisher('u', Float32, queue_size=10)
        self.v_pub = rospy.Publisher('v', Float32, queue_size=10)
        self.r_pub = rospy.Publisher('r', Float32, queue_size=10)

        self.motion_path_pub = rospy.Publisher("/run_path", Path, queue_size=1)
        # self.obs_motion_path_pub = rospy.Publisher("/obs_path", Path, queue_size=1)
        # self.obs_motion_path_pub1 = rospy.Publisher("/obs_path1", Path, queue_size=1)
        self.predicted_path_pub = rospy.Publisher('/predicted_path', MarkerArray, queue_size=10)

        self.motion_path = Path()
        # self.obs_motion_path = Path()
        # self.obs_motion_path1 = Path()
        self.marker_array = MarkerArray()

        rospy.Timer(rospy.Duration(0.02), self.publish_control_cmd)
        self.static_obs()
        #self.pub_markarry()
        
    def initialize_system_parameters(self):
        """初始化系统参数"""
        # 定义符号变量
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        u = ca.SX.sym('u')
        v = ca.SX.sym('v')
        r = ca.SX.sym('r')
        self.states = ca.vertcat(u, v, r, x, y, psi)
        self.n_states = self.states.size1()

        # 控制输入
        Tp = ca.SX.sym('Tp')
        Ts = ca.SX.sym('Ts')
        Tm = ca.SX.sym('Tm')
        self.controls = ca.vertcat(Tp, Ts, Tm)
        self.n_controls = self.controls.size1()

    def initialize_mpc_parameters(self):
        """初始化MPC参数"""
        self.time_total = 100.0  # 总仿真时间(s)
        self.dt = 0.02       # 时间步长
        o = 10
        self.Q = ca.diag(ca.vertcat(o, o, o, 100, 100, 100))  # 状态权重
        self.P_weight = ca.diag(ca.vertcat(o, o, o, 100, 100, 10))  # 终端权重
        self.R = ca.diag(ca.vertcat(0.5, 0.5, 0.5))  # 控制权重
        self.N = 7          # 预测步长
        self.C_t = 0.1       # 松弛变量初值

        # 状态约束
        self.xmin = -np.inf * np.ones(self.n_states).reshape(-1, 1)
        self.xmax = np.inf * np.ones(self.n_states).reshape(-1, 1)

        # 控制约束
        self.umin = np.array([-400, -400, -400]).reshape(-1, 1)
        self.umax = np.array([400, 400, 400]).reshape(-1, 1)

    def initialize_environment(self):
        """初始化环境设置"""
        self.x0 = np.array([0, 0, 0, 0, 0, -2.3]).reshape(-1, 1)  # 初始状态
        self.xf = np.array([0, 0, 0, -90, -100, -2.3]).reshape(-1, 1)  # 目标状态
        self.gamma = 0.5  # CBF参数
        self.obs_u = 0.41
        # 障碍物定义
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')
        obs_s = ca.SX.sym('obs_s')
        self.obs = ca.vertcat(obs_x, obs_y, obs_r, obs_s)

        # 障碍物参数
        self.num_obs = 2
        obs_rad = np.array([4.0, 3.0]).reshape(1, -1)
        obs_sens = np.array([obs_rad[0, 0]+ 4, obs_rad[0, 1]+4]).reshape(1, -1)
        self.obs1_initial = np.array([-42.0, -27.0, 25.0, obs_sens[0, 0]]).reshape(-1, 1)
        self.obs2_initial = np.array([-52.0, -77.0, 20, obs_sens[0, 0]]).reshape(-1, 1)
        #self.obs3_initial = np.array([-25.0, -32.0, obs_rad[0, 0]+4, obs_sens[0, 0]]).reshape(-1, 1)
        self.obs1 = copy.deepcopy(self.obs1_initial)
        self.obs2 = copy.deepcopy(self.obs2_initial)
        #self.obs3 = copy.deepcopy(self.obs3_initial)

        # CBF函数
        self.rho_circle1 = self.CBF_circle1(self.states, self.obs)
        self.rho_circle1_func = ca.Function('rho', [self.states, self.obs], [self.rho_circle1])

    def pub_uvr_date(self, state, control,f):
        u_date = Float32()
        v_date = Float32()
        r_date = Float32()
        next_state = state + self.dt * f(state, control)
        next_state = np.array(next_state)
        u_date.data =   next_state[0]   
        v_date.data =   next_state[1]
        r_date.data =   next_state[2]
        self.u_pub.publish(state[0])
        self.v_pub.publish(v_date)
        self.r_pub.publish(r_date)

    def build_dynamics_model(self):
        """构建动力学模型"""
        # 使用USVCS_dynamics函数
        self.dx_dt, self.f, self.g = self.USVCS_dynamics(self.states, self.controls)


        # 定义动力学函数
        self.F_func = ca.Function('F', [self.states, self.controls], [self.dx_dt])
        

    def setup_optimization_problem(self):
        """设置优化问题"""
        # 定义优化变量
        self.X = ca.SX.sym('X', self.n_states, self.N+1)       # 状态轨迹
        self.U = ca.SX.sym('U', self.n_controls, self.N)       # 控制序列
        self.C = ca.SX.sym('C', self.N)                   # 松弛变量

        # 参数（初始状态、目标状态、障碍物信息）
        self.P = ca.SX.sym('P', self.n_states + self.n_states + self.num_obs*4)

        # 初始化目标函数和约束
        self.obj = 0
        self.constraints = []

        # 初始状态约束
        st = self.X[:, 0]
        self.constraints = (st - self.P[:self.n_states])

        # 构建MPC问题
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            
            # 运行代价
            self.obj += (st - self.P[self.n_states:2*self.n_states]).T @ self.Q @ (st - self.P[self.n_states:2*self.n_states]) 
            self.obj += con.T @ self.R @ con
            
            # 动力学约束
            st_next = self.X[:, k+1]
            st_next_euler = st + self.dt * self.F_func(st, con)
            self.constraints = ca.vertcat(self.constraints, st_next - st_next_euler)

        for i in range(self.num_obs):
            for k in range(self.N):
                obs_loc1 = self.P[self.n_states + self.n_states + 4*i : self.n_states + self.n_states + 4*i + 4]
                rho = self.rho_circle1_func(self.X[:, k], obs_loc1)
                rho_next = self.rho_circle1_func(self.X[:, k+1], obs_loc1)
                
                CBF_constraint = rho_next - rho + self.gamma*rho
                self.constraints = ca.vertcat(self.constraints, CBF_constraint)

        # 终端代价
        self.obj += (self.X[:, self.N] - self.P[self.n_states:2*self.n_states]).T @ self.P_weight @ (self.X[:, self.N] - self.P[self.n_states:2*self.n_states])

        # 合并优化变量
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # 按列展开
            self.U.reshape((-1, 1)),
            self.C.reshape((-1, 1))
        )

        # 创建NLP问题
        self.nlp_prob = {
            'f': self.obj,
            'x': self.OPT_variables,
            'g': ca.vertcat(*ca.vertsplit(self.constraints, 1)),
            'p': self.P
        }

        # 求解器选项
        self.opts = {
            'ipopt': {
                'max_iter': 100,
                'print_level': 0,
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-5
            },
            'print_time': 0
        }

        # 创建求解器
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        # 约束边界设置
        self.total_eq_constraints = self.n_states * (self.N + 1)
        self.total_ineq_constraints = self.num_obs * self.N
        self.total_constraints = self.total_eq_constraints + self.total_ineq_constraints

        self.args = {
            'lbg': np.zeros((1, self.total_constraints)),
            'ubg': np.zeros((1, self.total_constraints)),
            'lbx': np.concatenate([
                np.tile(self.xmin, (self.N+1, 1)),  # 状态约束
                np.tile(self.umin, (self.N, 1)),    # 控制约束
                np.zeros((self.N, 1)),              # 松弛变量约束
            ]),
            'ubx': np.concatenate([
                np.tile(self.xmax, (self.N+1, 1)),
                np.tile(self.umax, (self.N, 1)),
                np.inf * np.ones((self.N, 1)) 
            ])
        }

        self.args['lbg'][0, :self.total_eq_constraints] = 0
        self.args['ubg'][0, :self.total_eq_constraints] = 0
        # 设置不等式约束 (CBF约束)
        start_idx = self.total_eq_constraints
        end_idx = self.total_eq_constraints + self.total_ineq_constraints
        self.args['lbg'][0, start_idx:end_idx] = 0
        self.args['ubg'][0, start_idx:end_idx] = np.inf

    
    def odomCallback1(self, msg):
        with self.data_lock1:
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, yaw = euler_from_quaternion(orientation_list)
            # 速度转换
            psi = yaw
            R_b_n = np.array([[math.cos(psi), -math.sin(psi)], 
                            [math.sin(psi), math.cos(psi)]])
            vel_n = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
            vel_b = R_b_n.T @ vel_n
            self.x0[0] = vel_b[0]
            self.x0[1] = vel_b[1]
            self.x0[2] = msg.twist.twist.angular.z
            
            self.x0[3] = msg.pose.pose.position.x
            self.x0[4] = msg.pose.pose.position.y
            self.x0[5] = yaw
            # 发布运动轨迹
            self.motion_path.header.frame_id = "world"
            self.motion_path.header.stamp = rospy.Time.now()
            pose_msg = PoseStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = msg.header.frame_id
            pose_msg.pose.position.x = msg.pose.pose.position.x
            pose_msg.pose.position.y = msg.pose.pose.position.y
            pose_msg.pose.position.z = 0
            pose_msg.pose.orientation = msg.pose.pose.orientation
            self.motion_path.poses.append(pose_msg)

    

    def publish_control_cmd(self, event):
        if np.linalg.norm(self.x0 - self.xf) > 0.6 and self.mpciter < int(self.time_total/self.dt):
            
            # 设置参数
            if np.linalg.norm(self.x0[3:5] - self.xf[3:5]) < 3.0:  # 5米内
                current_xf = copy.deepcopy(self.xf)
                current_xf[5] = self.x0[5]  # 使用当前航向作为目标
                self.args['p'] = np.concatenate([self.x0, current_xf, self.obs1, self.obs2])
            else:
                self.args['p'] = np.concatenate([self.x0, self.xf, self.obs1, self.obs2])
                    
            
            # 初始猜测
            self.args['x0'] = np.concatenate([
                self.X0.reshape((-1, 1)),  # 状态初始猜测
                self.u0.reshape((-1, 1)),  # 控制初始猜测
                self.C_0.reshape((-1, 1))    # 松弛变量初始猜测
            ])
            
            # 求解优化问题
            sol = self.solver(
                x0=self.args['x0'],
                lbx=self.args['lbx'],
                ubx=self.args['ubx'],
                lbg=self.args['lbg'],
                ubg=self.args['ubg'],
                p=self.args['p']
            )
            
            # 提取控制输入
            u = sol['x'][self.n_states*(self.N+1):self.n_states*(self.N+1)+self.n_controls*self.N]
            uuu = np.array(u)
            uuu = uuu.reshape(self.N, self.n_controls, order='C')
            #self.u_cl.append(u[0, :])
            self.u0 = np.vstack([uuu[1:], uuu[-1]])
            self.mpciter += 1
            rospy.loginfo(f"Iteration {self.mpciter}/{int(self.time_total/self.dt)} - Current error: {np.linalg.norm(self.x0-self.xf):.2f}")
            #print(f"Iteration {self.mpciter}/{int(self.time_total/self.dt)} - Current error: {np.linalg.norm(self.x0-self.xf):.2f}")
            
            self.X0 = sol['x'][:self.n_states*(self.N+1)].reshape((self.n_states, self.N+1)).T
            
            optimal_left = Float32()
            optimal_right = Float32()
            optimal_conter = Float32()
            optimal_left.data = uuu[0, 0]
            optimal_right.data = uuu[0, 1]
            optimal_conter.data = uuu[0, 2]
            #optimal_ts.data = self.uuu[0, 0]
            
            self.control_left_pub.publish(optimal_left)
            self.control_right_pub.publish(optimal_right)
            self.control_conter_pub.publish(optimal_conter)

            
            self.pub_uvr_date(self.x0, uuu[0, :], self.F_func)

            self.visualize_predicted_trajectory(np.array(self.X0))

            self.X0 = np.vstack([self.X0[1:, :], self.X0[-1, :]])
            self.C_0 = sol['x'][-self.N:].full().flatten()
        else:
            optimal_left = Float32()
            optimal_right = Float32()
            optimal_conter = Float32()
            optimal_left.data = 0.0
            optimal_right.data = 0.0
            optimal_conter.data = 0.0
            #optimal_ts.data = self.uuu[0, 0]
            
            self.control_left_pub.publish(optimal_left)
            self.control_right_pub.publish(optimal_right)
            self.control_conter_pub.publish(optimal_conter)
    
    def static_obs(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "solid_circle_markers"
        marker.id = 11
        marker.type = Marker.SPHERE  # 使用球体表示圆
        marker.action = Marker.ADD
        
        # 设置位置和方向（略微偏移Z轴避免重叠闪烁）
        marker.pose.position = Point(-42, -27, 0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)  # 无旋转
        
        # 设置尺寸（X/Y为直径，Z非常小以压扁成2D圆盘）
        marker.scale.x = 30.0  # 直径
        marker.scale.y = 30.0  # 直径
        marker.scale.z = 1.0        # 厚度（越小越接近2D）
        
        # 设置颜色 (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a =  1.0
        
        # 设置 Marker 生命周期（永久显示）
        marker.lifetime = rospy.Duration()
        self.marker_array.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "solid_circle_markers"
        marker.id = 12
        marker.type = Marker.SPHERE  # 使用球体表示圆
        marker.action = Marker.ADD
        
        # 设置位置和方向（略微偏移Z轴避免重叠闪烁）
        marker.pose.position = Point(-52, -77, 0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)  # 无旋转
        
        # 设置尺寸（X/Y为直径，Z非常小以压扁成2D圆盘）
        marker.scale.x = 20.0  # 直径
        marker.scale.y = 20.0  # 直径
        marker.scale.z = 1.0        # 厚度（越小越接近2D）
        
        # 设置颜色 (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a =  1.0
        
        # 设置 Marker 生命周期（永久显示）
        marker.lifetime = rospy.Duration()
        self.marker_array.markers.append(marker)

    
    def visualize_predicted_trajectory(self, predicted_traj):
        # 清除之前的预测轨迹标记
        for marker in self.marker_array.markers:
            if marker.ns == "predicted_trajectory":
                self.marker_array.markers.remove(marker)
        
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "predicted_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # 线宽
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0  # 绿色表示预测轨迹
        marker.color.b = 0.0
        
        # 添加轨迹点
        for k in range(predicted_traj.shape[0]):  # 遍历所有预测步
            point = Point()
            point.x = predicted_traj[k, 3]
            point.y = predicted_traj[k, 4]
            point.z = 0
            marker.points.append(point)
            
        
        self.marker_array.markers.append(marker)


        self.predicted_path_pub.publish(self.marker_array)
        self.motion_path_pub.publish(self.motion_path)

    
    


    def USVCS_dynamics(self, states, controls):
        """
        USV动力学模型 - MATLAB转Python版本
        输入:
            states: [u, v, r, x, y, psi] 状态向量
            controls: [Tp, Ts, Tm] 控制输入
        输出:
            dx_dt: 状态导数
            F_sys: 系统动态矩阵
            G_sys: 控制输入矩阵
        """
        # 参数定义 (与MATLAB保持一致)
        m = 250       # 质量
        Xdu = 0       # 纵向附加质量系数
        Ydv = 0       # 横向附加质量系数
        Xu = 51.3     # 纵向线性阻尼系数
        X_abs_u_u = 72.4  # 纵向非线性阻尼系数
        Yv = 40       # 横向线性阻尼系数
        Y_abs_v_v = 0 # 横向非线性阻尼系数
        Iz = 446      # 转动惯量
        Ndr = 0       # 转舵附加质量系数
        Nr = 400      # 转向线性阻尼系数
        N_abs_r_r = 0 # 转向非线性阻尼系数
        
        # 解包状态变量 (注意Python是0-based索引)
        u = states[0]
        v = states[1]
        r = states[2]
        x = states[3]
        y = states[4]
        psi = states[5]
        
        # 解包控制输入
        Tp = controls[0]
        Ts = controls[1]
        Tm = controls[2]
        
        # 系统动态部分 F_sys (与MATLAB公式一致)
        F_u = 0.004 * (m * v * r - Xu * u - 72.4 * ca.fabs(u) * u)
        F_v = 0.004 * (-m * v * r - Yv * v)
        F_r = 0.002 * (-Nr * r)
        
        F_sys = np.array([
            [F_u],
            [F_v],
            [F_r],
            [u * np.cos(psi) - v * np.sin(psi)],  # x方向运动学
            [u * np.sin(psi) + v * np.cos(psi)],  # y方向运动学
            [r]                                  # 航向角变化率
        ])
        
        # 控制输入矩阵 G_sys (与MATLAB最终版本一致)
        G_sys = np.array([
            [0.004*250, 0.004*250, 0],
            [0, 0, 0.004*250],
            [0.002*1.02*250, -0.002*1.02*250, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        # 计算状态导数 dx/dt = F_sys + G_sys * controls
        controls_vec = np.array([[Tp], [Ts], [Tm]])  # 转换为列向量
        dx_dt = F_sys + np.dot(G_sys, controls_vec)
        
        return dx_dt, F_sys, G_sys
# 使用示例
if __name__ == '__main__':
    try:
        mpc = CBFMPCController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass