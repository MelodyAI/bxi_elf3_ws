#!/usr/bin/env python3
"""
Joint T-N Curve Visualizer (High Performance)
Subscribe /simulation/joint_states, monitor torque-speed operating points.
Static TN curves drawn once; only dynamic points updated per frame.

Usage:
  source install/setup.bash
  python3 src/bxi_example_py_elf3/bxi_example_py_elf3/joint_tn_visualizer.py
  python3 src/bxi_example_py_elf3/bxi_example_py_elf3/joint_tn_visualizer.py 2>&1 | head -30

Keys (in matplotlib window):
  Left/Right: switch joint
  Up/Down:    toggle single/all view
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import sensor_msgs.msg
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from collections import deque
from threading import Lock, Thread
import signal
import sys

# ==================== Joint Definitions ====================
JOINT_NAMES = (
    "waist_y_joint",  "waist_x_joint",  "waist_z_joint",
    "l_hip_y_joint",  "l_hip_x_joint",  "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint",  "r_hip_x_joint",  "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
)
DOF_NUM = 29

# ==================== Ankle Parallel Linkage ====================
# 脚踝采用并联连杆机构，两个电机(A,B)耦合驱动 ankle_y(pitch) 和 ankle_x(roll)
# 运动学:  vel_A = vel_y + vel_x,   vel_B = vel_y - vel_x
# 力矩:   tau_A = (tau_y + tau_x)/2, tau_B = (tau_y - tau_x)/2
# ankle_y 位置显示 Motor A 工作点, ankle_x 位置显示 Motor B 工作点
ANKLE_PAIRS = [
    (7, 8),   # l_ankle_y_joint, l_ankle_x_joint
    (13, 14), # r_ankle_y_joint, r_ankle_x_joint
]
# 构建快速查找: ankle索引 -> (pair_y_idx, pair_x_idx, is_motor_A)
ANKLE_LOOKUP = {}
for iy, ix in ANKLE_PAIRS:
    ANKLE_LOOKUP[iy] = (iy, ix, True)   # ankle_y -> Motor A
    ANKLE_LOOKUP[ix] = (iy, ix, False)  # ankle_x -> Motor B

def ankle_parallel_transform(vel_y, vel_x, tau_y, tau_x):
    """将等效关节空间 (ankle_y, ankle_x) 转换到电机空间 (Motor A, Motor B)"""
    vel_A = vel_y + vel_x
    vel_B = vel_y - vel_x
    tau_A = (tau_y + tau_x) / 2.0
    tau_B = (tau_y - tau_x) / 2.0
    return vel_A, vel_B, tau_A, tau_B

# ==================== Motor T-N Curves ====================
TN_MOTOR_85 = np.array([
    [1.05,96.0],[3.08,96.0],[5.12,96.0],[7.16,96.0],[9.19,94.8],
    [11.23,90.9],[13.26,62.5],[15.3,32.2],[17.34,12.4],[19.37,8.0]])
TN_MOTOR_70 = np.array([
    [0.84,48.4],[2.78,47.3],[4.72,45.6],[6.67,42.8],[8.61,37.8],
    [10.55,30.6],[12.5,22.1],[14.44,14.5],[16.38,8.1],[18.33,3.5]])
TN_MOTOR_50 = np.array([
    [1.57,25.0],[3.43,25.2],[5.29,24.8],[7.16,23.5],[9.02,20.8],
    [10.88,17.2],[12.74,13.5],[14.6,9.8],[16.46,6.2],[18.33,3.1]])

def _motor_type(name):
    n = name.lower()
    if 'wrist_' in n or 'shoulder_z' in n or 'ankle_' in n:
        return 50
    if 'hip_z' in n or 'waist_x' in n or 'waist_y' in n or \
       'shoulder_x' in n or 'shoulder_y' in n or 'elbow_' in n:
        return 70
    if 'hip_x' in n or 'hip_y' in n or 'waist_z' in n or 'knee_' in n:
        return 85
    return 70

_TN_MAP = {85: TN_MOTOR_85, 70: TN_MOTOR_70, 50: TN_MOTOR_50}
_CLR_MAP = {85: '#2196F3', 70: '#FF9800', 50: '#4CAF50'}

# Precompute per-joint
MTYPE  = [_motor_type(n) for n in JOINT_NAMES]
MTN    = [_TN_MAP[m] for m in MTYPE]
MCLR   = [_CLR_MAP[m] for m in MTYPE]
MINTERP = [interp1d(t[:,0], t[:,1], kind='cubic', fill_value='extrapolate') for t in MTN]
SMOOTH = []
for t in MTN:
    f = interp1d(t[:,0], t[:,1], kind='cubic')
    xs = np.linspace(t[0,0], t[-1,0], 100)
    SMOOTH.append((xs, f(xs)))

HIST_LEN = 50

# ==================== ROS2 Node ====================
class Listener(Node):
    def __init__(self):
        super().__init__('joint_tn_visualizer')
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability,
                         reliability=qos_profile_sensor_data.reliability)
        self.create_subscription(sensor_msgs.msg.JointState,
                                 '/simulation/joint_states', self._cb, qos)
        self._lock = Lock()
        self.vel = np.zeros(DOF_NUM)
        self.tor = np.zeros(DOF_NUM)
        self.ok = False
        self.vh = [deque(maxlen=HIST_LEN) for _ in range(DOF_NUM)]
        self.th = [deque(maxlen=HIST_LEN) for _ in range(DOF_NUM)]
        # 超载点持久存储
        self.ol_v = [[] for _ in range(DOF_NUM)]  # overload velocity
        self.ol_t = [[] for _ in range(DOF_NUM)]  # overload torque

    def _cb(self, msg):
        with self._lock:
            nv = min(len(msg.velocity), DOF_NUM)
            self.vel[:nv] = msg.velocity[:nv]
            ne = min(len(msg.effort), DOF_NUM)
            self.tor[:ne] = msg.effort[:ne]

            # 脚踝并联连杆转换: 将等效关节空间转换到电机空间
            for iy, ix in ANKLE_PAIRS:
                vel_A, vel_B, tau_A, tau_B = ankle_parallel_transform(
                    self.vel[iy], self.vel[ix], self.tor[iy], self.tor[ix])
                self.vel[iy] = vel_A   # ankle_y 位置 -> Motor A
                self.vel[ix] = vel_B   # ankle_x 位置 -> Motor B
                self.tor[iy] = tau_A
                self.tor[ix] = tau_B

            self.ok = True
            for i in range(DOF_NUM):
                av = abs(self.vel[i])
                at = abs(self.tor[i])
                self.vh[i].append(av)
                self.th[i].append(at)
                # 检测超载并记录
                if av > 0:
                    rated = float(max(0, MINTERP[i](av)))
                    if at > rated:
                        self.ol_v[i].append(av)
                        self.ol_t[i].append(at)

    def clear_overload(self):
        """清除所有超载点记录"""
        with self._lock:
            for i in range(DOF_NUM):
                self.ol_v[i].clear()
                self.ol_t[i].clear()

    def snap(self):
        with self._lock:
            return (self.vel.copy(), self.tor.copy(), self.ok,
                    [np.array(h) for h in self.vh],
                    [np.array(h) for h in self.th],
                    [np.array(v) for v in self.ol_v],
                    [np.array(t) for t in self.ol_t])


# ==================== Single Joint View ====================
class SingleView:
    def __init__(self, fig):
        self.fig = fig
        self.ax = fig.add_axes([0.08, 0.12, 0.58, 0.78])
        self.ax_info = fig.add_axes([0.72, 0.12, 0.26, 0.78])
        self.ax_info.axis('off')
        self._idx = -1
        self.trail = None
        self.dot = None
        self.anno = None
        self.ox = None
        self.otxt = None
        self.itxt = None

    def show(self):
        self.ax.set_visible(True)
        self.ax_info.set_visible(True)
        self._idx = -1

    def hide(self):
        self.ax.set_visible(False)
        self.ax_info.set_visible(False)

    def _build(self, idx):
        ax = self.ax
        ax.clear()
        tn, clr, mt = MTN[idx], MCLR[idx], MTYPE[idx]
        xs, ys = SMOOTH[idx]

        ax.set_facecolor('#2b2b2b')
        ax.fill_between(xs, 0, ys, alpha=0.15, color=clr)
        ax.plot(xs, ys, '-', color=clr, lw=2.5, label=f'Motor{mt} T-N')
        ax.plot(tn[:,0], tn[:,1], 'o', color=clr, ms=6, zorder=5)
        ax.fill_between(-xs, 0, ys, alpha=0.08, color=clr)
        ax.plot(-xs, ys, '--', color=clr, lw=1.5, alpha=0.5)

        ax.set_xlabel('Speed |N| (rad/s)', color='white', fontsize=13)
        ax.set_ylabel('Torque |T| (Nm)', color='white', fontsize=13)
        # 脚踝并联连杆标注
        if idx in ANKLE_LOOKUP:
            _, _, is_A = ANKLE_LOOKUP[idx]
            motor_label = 'Motor A (Y+X)' if is_A else 'Motor B (Y-X)'
            ax.set_title(f'{JOINT_NAMES[idx]}  [Motor{mt}] [{motor_label}]',
                         color='#FFD700', fontsize=14, fontweight='bold', pad=10)
        else:
            ax.set_title(f'{JOINT_NAMES[idx]}  [Motor{mt}]',
                         color='white', fontsize=16, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=10,
                  facecolor='#333', edgecolor='#555', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#666')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#555')
        mx = tn[-1,0]
        ax.set_xlim(-mx*0.15, mx*1.15)
        ax.set_ylim(-2, tn[0,1]*1.25)

        self.trail, = ax.plot([], [], '.', color='#aaaaaa', ms=3, alpha=0.5)
        self.dot, = ax.plot([], [], 'o', color='#FF4444', ms=14, zorder=10,
                            markeredgecolor='white', markeredgewidth=2)
        self.anno = ax.annotate('', xy=(0,0), xytext=(15,15),
                                textcoords='offset points', color='#FF6666',
                                fontsize=12, fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color='#FF6666', lw=1.5))
        self.ol_scatter, = ax.plot([], [], 'x', color='#FF6600', ms=8,
                                   markeredgewidth=1.5, zorder=9, alpha=0.7,
                                   label='Overload points')
        self.ox, = ax.plot([], [], 'x', color='#FF0000', ms=20, markeredgewidth=3, zorder=11)
        self.otxt = ax.text(0, 0, 'OVERLOAD!', color='#FF0000', fontsize=12,
                            fontweight='bold', ha='center', va='bottom', visible=False)

        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#1e1e1e')
        self.itxt = self.ax_info.text(
            0.0, 1.0, '', transform=self.ax_info.transAxes,
            fontsize=8, va='top', fontfamily='monospace', color='#cccccc',
            bbox=dict(facecolor='#2b2b2b', edgecolor='#555', boxstyle='round,pad=0.5'))
        self._idx = idx

    def update(self, idx, vel, tor, vh, th, ol_v, ol_t):
        if idx != self._idx:
            self._build(idx)

        av, at = abs(vel[idx]), abs(tor[idx])

        # overload persistent points
        if len(ol_v[idx]) > 0:
            self.ol_scatter.set_data(ol_v[idx], ol_t[idx])
        else:
            self.ol_scatter.set_data([], [])

        # trail
        if len(vh[idx]) > 0:
            self.trail.set_data(vh[idx], th[idx])
        # dot
        self.dot.set_data([av], [at])
        self.anno.xy = (av, at)
        self.anno.set_text(f'({av:.2f}, {at:.2f})')

        # overload
        tn = MTN[idx]
        mx = tn[-1,0]
        rated = 0.0
        usage = 0.0
        ol = False
        if av > 0:
            rated = float(max(0, MINTERP[idx](av)))
            if rated > 0:
                usage = at / rated * 100
            if at > rated:
                ol = True
        if ol:
            self.ox.set_data([av], [at])
            self.otxt.set_position((av, at + rated*0.1))
            self.otxt.set_visible(True)
        else:
            self.ox.set_data([], [])
            self.otxt.set_visible(False)

        # info
        jn, mt = JOINT_NAMES[idx], MTYPE[idx]
        L = [
            f"{'='*32}", "  Joint Info Panel", f"{'='*32}", "",
            f"  Joint: {jn}", f"  Index: {idx} / {DOF_NUM-1}",
            f"  Motor: Type {mt}", "",
        ]
        if idx in ANKLE_LOOKUP:
            _, _, is_A = ANKLE_LOOKUP[idx]
            motor_label = 'Motor A (vel_y+vel_x)' if is_A else 'Motor B (vel_y-vel_x)'
            L += [f"  ** Parallel Linkage **",
                  f"  Showing: {motor_label}",
                  f"  vel_motor = vel_y {'+ ' if is_A else '- '}vel_x",
                  f"  tau_motor = (tau_y {'+ ' if is_A else '- '}tau_x)/2", ""]
        L += [
            f"  Velocity: {vel[idx]:+.4f} rad/s",
            f"  Torque:   {tor[idx]:+.4f} Nm",
            f"  |Vel|:    {av:.4f} rad/s",
            f"  |Torque|: {at:.4f} Nm", "",
        ]
        if av > 0 and av <= mx and rated > 0:
            L += [f"  Rated T:  {rated:.2f} Nm", f"  Load:     {usage:.1f}%"]
            if usage > 100:
                L.append("  ** OVERLOAD! **")
        ol_cnt = len(ol_v[idx])
        L += ["", f"  Overload pts: {ol_cnt}"]
        L += ["", f"{'='*32}", "  Keys:", "  Left/Right : Switch joint",
              "  Up/Down    : Toggle view",
              "  C          : Clear overload pts", f"{'='*32}", "", "  All Joints:"]
        for i, jname in enumerate(JOINT_NAMES):
            m = " >> " if i == idx else "    "
            suffix = ""
            if i in ANKLE_LOOKUP:
                _, _, is_A = ANKLE_LOOKUP[i]
                suffix = " MotorA" if is_A else " MotorB"
            L.append(f"{m}{i:2d}. {jname} [{MTYPE[i]}]{suffix}")
        self.itxt.set_text('\n'.join(L))


# ==================== All Joints View ====================
class AllView:
    def __init__(self, fig):
        self.fig = fig
        self.axes = []
        self.dots = []
        self.trails = []
        self.ol_plots = []
        self._built = False

    def _build(self):
        cols = 6
        for i in range(DOF_NUM):
            r, c = divmod(i, cols)
            ax = self.fig.add_axes([0.05+c*0.155, 0.82-r*0.185, 0.135, 0.155])
            ax.set_facecolor('#2b2b2b')
            tn, clr = MTN[i], MCLR[i]
            xs, ys = SMOOTH[i]
            ax.fill_between(xs, 0, ys, alpha=0.12, color=clr)
            ax.plot(xs, ys, '-', color=clr, lw=1.2)
            short = JOINT_NAMES[i].replace('_joint','')
            if i in ANKLE_LOOKUP:
                _, _, is_A = ANKLE_LOOKUP[i]
                ml = 'A' if is_A else 'B'
                ax.set_title(f'{short} [{MTYPE[i]}] M{ml}', color='#FFD700', fontsize=7, pad=2)
            else:
                ax.set_title(f'{short} [{MTYPE[i]}]', color='white', fontsize=7, pad=2)
            ax.set_xlim(-0.5, tn[-1,0]*1.1)
            ax.set_ylim(-1, tn[0,1]*1.2)
            ax.tick_params(labelsize=5, colors='#aaa')
            ax.grid(True, alpha=0.2, color='#666')
            for sp in ax.spines.values():
                sp.set_color('#444')

            ol, = ax.plot([], [], 'x', color='#FF6600', ms=4,
                         markeredgewidth=1, zorder=9, alpha=0.7)
            tr, = ax.plot([], [], '.', color='#999', ms=1.5, alpha=0.4)
            dt, = ax.plot([], [], 'o', color='#FF4444', ms=5, zorder=10,
                          markeredgecolor='white', markeredgewidth=0.5)
            self.axes.append(ax)
            self.ol_plots.append(ol)
            self.trails.append(tr)
            self.dots.append(dt)
        self._built = True

    def show(self):
        if not self._built:
            self._build()
        for ax in self.axes:
            ax.set_visible(True)

    def hide(self):
        for ax in self.axes:
            ax.set_visible(False)

    def update(self, cur, vel, tor, vh, th, ol_v, ol_t):
        for i in range(DOF_NUM):
            av, at = abs(vel[i]), abs(tor[i])
            self.dots[i].set_data([av], [at])
            if len(vh[i]) > 0:
                self.trails[i].set_data(vh[i], th[i])
            # overload persistent points
            if len(ol_v[i]) > 0:
                self.ol_plots[i].set_data(ol_v[i], ol_t[i])
            else:
                self.ol_plots[i].set_data([], [])
            self.axes[i].title.set_color('#FFD700' if i == cur else 'white')
            pc = '#FF4444'
            tn = MTN[i]
            if av > 0 and av <= tn[-1,0]:
                if at > max(0, MINTERP[i](av)):
                    pc = '#FF0000'
            self.dots[i].set_color(pc)


# ==================== Main Visualizer ====================
class App:
    def __init__(self, node):
        self.node = node
        self.cur = 0
        self.mode = 'single'

        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = plt.figure(figsize=(16, 9), facecolor='#1e1e1e')
        self.fig.canvas.manager.set_window_title('Joint T-N Curve Monitor')

        self.sv = SingleView(self.fig)
        self.av = AllView(self.fig)
        self.av.hide()

        axp = self.fig.add_axes([0.08, 0.02, 0.08, 0.04])
        axn = self.fig.add_axes([0.18, 0.02, 0.08, 0.04])
        axt = self.fig.add_axes([0.30, 0.02, 0.12, 0.04])
        axc = self.fig.add_axes([0.44, 0.02, 0.12, 0.04])
        self.bp = Button(axp, '<< Prev', color='#333', hovercolor='#555')
        self.bn = Button(axn, 'Next >>', color='#333', hovercolor='#555')
        self.bt = Button(axt, 'All Joints', color='#333', hovercolor='#555')
        self.bc = Button(axc, 'Clear OL', color='#662222', hovercolor='#993333')
        for b in (self.bp, self.bn, self.bt, self.bc):
            b.label.set_color('white')
            b.label.set_fontsize(10)
        self.bp.on_clicked(lambda _: self._sw(-1))
        self.bn.on_clicked(lambda _: self._sw(1))
        self.bt.on_clicked(lambda _: self._tog())
        self.bc.on_clicked(lambda _: self._clear_overload())
        self.fig.canvas.mpl_connect('key_press_event', self._key)
        self.fig.canvas.mpl_connect('button_press_event', self._click)

    def _sw(self, d):
        self.cur = (self.cur + d) % DOF_NUM

    def _tog(self):
        if self.mode == 'single':
            self.mode = 'all'
            self.bt.label.set_text('Single Joint')
            self.sv.hide()
            self.av.show()
        else:
            self.mode = 'single'
            self.bt.label.set_text('All Joints')
            self.av.hide()
            self.sv.show()

    def _key(self, ev):
        if ev.key == 'right': self._sw(1)
        elif ev.key == 'left': self._sw(-1)
        elif ev.key in ('up','down'): self._tog()
        elif ev.key == 'c': self._clear_overload()

    def _clear_overload(self):
        """清除所有超载点记录"""
        self.node.clear_overload()
        print("Overload points cleared.")

    def _click(self, ev):
        if self.mode == 'all' and self.av._built and ev.inaxes in self.av.axes:
            i = self.av.axes.index(ev.inaxes)
            if i < DOF_NUM:
                self.cur = i
                self.mode = 'single'
                self.bt.label.set_text('All Joints')
                self.av.hide()
                self.sv.show()

    def _upd(self, frame):
        vel, tor, ok, vh, th, ol_v, ol_t = self.node.snap()
        if not ok:
            return
        if self.mode == 'single':
            self.sv.update(self.cur, vel, tor, vh, th, ol_v, ol_t)
        else:
            self.av.update(self.cur, vel, tor, vh, th, ol_v, ol_t)

    def run(self):
        self.anim = FuncAnimation(self.fig, self._upd, interval=200,
                                  cache_frame_data=False)
        plt.show()


def main():
    rclpy.init()
    node = Listener()
    Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    app = App(node)

    def _sig(s, f):
        print("\nExiting...")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sig)

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
