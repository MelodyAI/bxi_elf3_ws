from pathlib import Path
import contextlib
import io

import numpy as np
import onnxruntime as ort

from utils.tfs import get_gravity_orientation

_CV2 = None
_CV2_IMPORT_TRIED = False


def _get_cv2():
    global _CV2, _CV2_IMPORT_TRIED
    if _CV2_IMPORT_TRIED:
        return _CV2
    _CV2_IMPORT_TRIED = True
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import cv2 as cv2_module
        _CV2 = cv2_module
    except Exception:
        _CV2 = None
    return _CV2


class CircularBuffer:
    def __init__(self, length: int):
        self.length = int(length)
        self._buffer = None
        self._num_pushes = 0

    def append(self, value):
        value = np.asarray(value, dtype=np.float32)
        if self._buffer is None:
            self._buffer = np.zeros((self.length,) + value.shape, dtype=np.float32)
        if self._num_pushes == 0:
            self._buffer[:] = value
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = value
        self._num_pushes += 1

    @property
    def buffer(self):
        return self._buffer

    def reset(self):
        if self._buffer is not None:
            self._buffer[:] = 0.0
        self._num_pushes = 0


class HumanoidGaitDepthPolicy:
    """带深度相机输入的 ELF3 / BX 29DoF 行走策略部署类。"""

    def __init__(self, model_onnx_path: str, cmd_is_joystick_ratio: bool = False):
        """
        Args:
            model_onnx_path: 深度行走 ONNX 模型路径。
            cmd_is_joystick_ratio: True 时按 deploy_mujoco_bx.py 的摇杆比例命令处理；
                False 时把 cmd_vel 当作实际速度命令，适配 bxi_sim.py 的调用方式。
        """
        self.model_onnx_path = self._resolve_policy_path(model_onnx_path)
        self.cmd_is_joystick_ratio = cmd_is_joystick_ratio

        self.num_actions = 29
        self.num_obs = 96
        self.history_length = 10
        self.control_dt = 0.02
        self.if_use_stand = True
        self.force_phase_active = False
        self.clip_action_limit = 100.0

        self.joint2motor_idx = np.array([
            15, 22, 0, 16, 23, 1,
            17, 24, 2, 18, 25, 3,
            9, 19, 26,
            4, 10, 20, 27, 5, 11, 21,
            28, 6, 12, 7, 13, 8, 14,
        ], dtype=np.int32)
        self.mujoco_to_isaac_idx = self.joint2motor_idx

        self.kps = np.array([
            108.448, 162.672, 176.421,
            176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
            176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
            54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
            54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
        ], dtype=np.float32)
        self.kds = np.array([
            6.904, 10.356, 11.231,
            11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
            11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
            3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
            3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
        ], dtype=np.float32)

        self.default_dof_pos = np.array([
            0.0, 0.0, 0.0,
            -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
            -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.default_angles_policy = self.default_dof_pos[self.joint2motor_idx]

        action_scales = np.array([
            0.231, 0.154, 0.213,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
            0.231, 0.231, 0.373, 0.231, 0.373, 0.373, 0.373,
            0.231, 0.231, 0.373, 0.231, 0.373, 0.373, 0.373,
        ], dtype=np.float32)
        self.action_scales_policy = action_scales[self.joint2motor_idx]

        self.range_velx = np.array([0.0, 1.0], dtype=np.float32)
        self.range_vely = np.array([0.0, 0.0], dtype=np.float32)
        self.range_velz = np.array([-1.57, 1.57], dtype=np.float32)
        self.cmd_scale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.ang_vel_scale = 1.0
        self.cmd = np.zeros(3, dtype=np.float32)

        self.camera_name = "depth_cam"
        self.output_resolution = [64, 36]
        self.width = 36
        self.height = 64
        self.crop_region = [6, 9, 16, 16]
        self.gaussian_kernel_size = (3, 3)
        self.gaussian_sigma = 1.0
        self.depth_range = [0.2, 2.5]
        self.depth_normalize = True
        self.depth_output_range = [0.0, 1.0]
        self.depth_history_len = 8
        self.depth_h = 21
        self.depth_w = 32
        self.depth_obs_indices = np.array(
            [-15, -13, -11, -9, -7, -5, -3, -1], dtype=np.int32
        )
        self.depth_image_buffer = CircularBuffer(length=60)

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = self.default_angles_policy.copy()
        self.target_dof_pos = self.default_dof_pos.copy()
        self.actor_obs_buffer = np.zeros(self.history_length * self.num_obs, dtype=np.float32)
        self.first = True
        self.counter = 0

        self.initialize_model(self.model_onnx_path)

    def initialize_model(self, onnx_path):
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
            sess_options=options,
        )
        self._configure_model_io()
        self.reset()
        self._warmup_policy()
        print(f"AMP depth model init finished: {onnx_path}")
        print("########################################################################")

    def reset(self):
        self.actor_obs_buffer = np.zeros(self.history_length * self.num_obs, dtype=np.float32)
        self.depth_image_buffer.reset()
        self.last_action = self.default_angles_policy.copy()
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_dof_pos.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self.first = True
        self.counter = 0

    def inference_step(self, q, dq, quat, omega, cmd_vel, depth_image=None):
        obs = self.compute_observation(q, dq, quat, omega, cmd_vel)
        ort_outputs = self._run_policy(obs, depth_image)
        action_out = np.squeeze(ort_outputs[self.action_output_index]).astype(np.float32)

        self.action = np.clip(
            action_out[:self.num_actions], -self.clip_action_limit, self.clip_action_limit
        )
        self.last_action = self.action.copy()

        target_policy_order = self.default_angles_policy + self.action * self.action_scales_policy
        target_mujoco_order = np.zeros_like(target_policy_order, dtype=np.float32)
        for i, motor_idx in enumerate(self.joint2motor_idx):
            target_mujoco_order[motor_idx] = target_policy_order[i]

        self.target_dof_pos = target_mujoco_order
        return self.target_dof_pos

    def compute_observation(self, qj, dqj, quat, omega, cmd_vel):
        self._update_command(cmd_vel)

        vel_norm = np.linalg.norm(self.cmd)
        if self.if_use_stand:
            if vel_norm > 0.1 or self.force_phase_active:
                self.counter += 1
            else:
                self.counter = 0
        else:
            self.counter += 1

        qj = np.asarray(qj, dtype=np.float32)[:self.num_actions]
        dqj = np.asarray(dqj, dtype=np.float32)[:self.num_actions]
        self.qj_obs[:] = qj[self.joint2motor_idx]
        self.dqj_obs[:] = dqj[self.joint2motor_idx]

        qj_obs = (self.qj_obs - self.default_angles_policy) * self.dof_pos_scale
        dqj_obs = self.dqj_obs * self.dof_vel_scale
        ang_vel = np.asarray(omega, dtype=np.float32) * self.ang_vel_scale
        gravity_orientation = get_gravity_orientation(quat).astype(np.float32)
        cmd = self.cmd * self.cmd_scale

        obs = np.concatenate([
            ang_vel.copy(),
            gravity_orientation.copy(),
            cmd.copy(),
            qj_obs.copy(),
            dqj_obs.copy(),
            self.last_action.copy(),
        ]).astype(np.float32)

        if self.num_obs == obs.shape[0] + 2:
            phase = (self.counter * self.control_dt) % 1.0
            obs = np.concatenate([
                obs,
                np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32),
            ])
        if obs.shape[0] != self.num_obs:
            raise ValueError(f"AMP depth obs dim mismatch: got {obs.shape[0]}, expected {self.num_obs}")
        return np.clip(obs, -100, 100).astype(np.float32)

    def preprocess_depth_image(self, depth_image):
        if depth_image is None:
            return np.zeros((self.depth_h, self.depth_w), dtype=np.float32)

        depth_image = np.asarray(depth_image, dtype=np.float32)
        if depth_image.shape == (self.depth_h, self.depth_w):
            return depth_image.copy()

        depth_image = np.flip(np.transpose(depth_image.copy(), (1, 0)), axis=0)
        if self.crop_region is not None:
            top, _bottom, left, _right = self.crop_region
            target_h = self.depth_w
            target_w = self.depth_h
            start_h = left
            start_w = top
            depth_image = depth_image[start_w:start_w + target_w, start_h:start_h + target_h]

        cv2_module = _get_cv2()
        if self.gaussian_kernel_size is not None and cv2_module is not None:
            depth_image = cv2_module.GaussianBlur(
                depth_image.astype(np.float32),
                self.gaussian_kernel_size,
                self.gaussian_sigma,
            )

        d_min, d_max = self.depth_range
        depth_image = np.clip(depth_image, d_min, d_max)
        if self.depth_normalize:
            depth_image = (depth_image - d_min) / max(d_max - d_min, 1e-6)
            out_min, out_max = self.depth_output_range
            depth_image = depth_image * (out_max - out_min) + out_min

        if depth_image.shape != (self.depth_h, self.depth_w):
            raise ValueError(
                f"AMP depth image shape mismatch: got {depth_image.shape}, "
                f"expected {(self.depth_h, self.depth_w)}"
            )
        return depth_image.copy().astype(np.float32)

    def _run_policy(self, obs, depth_image):
        if self.model_input_mode == "history_only":
            obs_buffer = self._update_obs_history(obs)
            return self.session.run(None, {self.input_name: obs_buffer})

        depth_buffer = self._get_depth_image_downsample_obs(depth_image)
        if self.model_input_mode == "single_obs_depth":
            obs_buffer = np.concatenate(
                [obs.reshape(1, -1), depth_buffer.reshape(1, -1)], axis=1
            ).astype(np.float32)
            return self.session.run(None, {self.input_name: obs_buffer})

        obs_buffer = self._update_obs_history(obs)
        depth_buffer = depth_buffer.reshape(1, -1, self.depth_h, self.depth_w).astype(np.float32)
        return self.session.run(None, {
            self.obs_input_name: obs_buffer,
            self.depth_input_name: depth_buffer,
        })

    def _update_obs_history(self, obs):
        if self.first:
            self.actor_obs_buffer[:] = np.tile(obs, self.history_length)
            self.first = False
        else:
            self.actor_obs_buffer = np.concatenate(
                (self.actor_obs_buffer[self.num_obs:], obs), axis=0, dtype=np.float32
            )
        return self.actor_obs_buffer.reshape(1, -1).astype(np.float32)

    def _get_depth_image_downsample_obs(self, depth_image):
        self.depth_image_buffer.append(self.preprocess_depth_image(depth_image))
        return self.depth_image_buffer.buffer[self.depth_obs_indices, ...]

    def _update_command(self, cmd_vel):
        cmd_vel = np.asarray(cmd_vel, dtype=np.float32)
        if self.cmd_is_joystick_ratio:
            self.cmd[0] = np.clip(cmd_vel[0] * self.range_velx[1], self.range_velx[0], self.range_velx[1])
            self.cmd[1] = np.clip(cmd_vel[1] * self.range_vely[1], self.range_vely[0], self.range_vely[1])
            self.cmd[2] = np.clip(cmd_vel[2] * self.range_velz[1], self.range_velz[0], self.range_velz[1])
        else:
            self.cmd[0] = np.clip(cmd_vel[0], self.range_velx[0], self.range_velx[1])
            self.cmd[1] = np.clip(cmd_vel[1], self.range_vely[0], self.range_vely[1])
            self.cmd[2] = np.clip(cmd_vel[2], self.range_velz[0], self.range_velz[1])

    def _configure_model_io(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.output_names = [out.name for out in outputs]
        self.action_output_index = 0
        for i, out in enumerate(outputs):
            out_name = out.name.lower()
            if any(token in out_name for token in ("action", "actions", "policy")):
                self.action_output_index = i

        self.model_input_mode = "history_only"
        self.obs_input_name = self.input_name
        self.depth_input_name = None
        if len(inputs) == 1:
            flat_dim = self._shape_dim(inputs[0].shape[1])
            if flat_dim == self.history_length * self.num_obs:
                return
            self.model_input_mode = "single_obs_depth"
            self.history_length = 1
            self.actor_proprio_dim = flat_dim - self.depth_history_len * self.depth_h * self.depth_w
            if self.actor_proprio_dim != self.num_obs:
                raise ValueError(
                    f"Single-input ONNX proprio dim mismatch: got {self.actor_proprio_dim}, "
                    f"expected {self.num_obs}"
                )
            return

        self.model_input_mode = "multi_input_history"
        self.obs_input_name = inputs[0].name
        self.depth_input_name = inputs[1].name
        self.history_length = int(self._shape_dim(inputs[0].shape[1]) / self.num_obs)
        self.depth_history_len = self._shape_dim(inputs[1].shape[1])
        self.depth_h = self._shape_dim(inputs[1].shape[2])
        self.depth_w = self._shape_dim(inputs[1].shape[3])

    def _warmup_policy(self):
        for _ in range(5):
            if self.model_input_mode == "history_only":
                obs_tensor = np.zeros((1, self.history_length * self.num_obs), dtype=np.float32)
                self.session.run(None, {self.input_name: obs_tensor})
            elif self.model_input_mode == "single_obs_depth":
                obs_tensor = np.zeros((1, self.num_obs + self.depth_history_len * self.depth_h * self.depth_w), dtype=np.float32)
                self.session.run(None, {self.input_name: obs_tensor})
            else:
                obs_tensor = np.zeros((1, self.history_length * self.num_obs), dtype=np.float32)
                depth_tensor = np.zeros((1, self.depth_history_len, self.depth_h, self.depth_w), dtype=np.float32)
                self.session.run(None, {
                    self.obs_input_name: obs_tensor,
                    self.depth_input_name: depth_tensor,
                })

    @staticmethod
    def _shape_dim(dim):
        if isinstance(dim, int):
            return dim
        if isinstance(dim, str):
            raise ValueError(f"Dynamic ONNX dimension is unsupported here: {dim}")
        return int(dim)

    @staticmethod
    def _resolve_policy_path(path):
        project_root = Path(__file__).resolve().parent.parent
        expanded = Path(path).expanduser()
        if expanded.is_absolute() and expanded.exists():
            return str(expanded)

        candidates = [
            Path.cwd() / expanded,
            project_root / expanded,
            project_root / "deploy_bx/deploy/policy/loco29_depth/model" / expanded,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(project_root / expanded)
