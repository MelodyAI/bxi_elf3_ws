"""Deployment adapter for external-reference ELF3 RGMT policies.

The ONNX graph is intentionally motion agnostic: it maps one flat RGMT
observation to one residual joint action.  This module owns reference-motion
loading/windowing, proprioceptive history, and the reference-residual action
conversion.  It therefore supports both an offline NPZ and a live 21-frame
reference window without changing or re-exporting the policy graph.

Quaternion inputs use WXYZ order.  Robot angular velocity must already be in
the robot base frame, matching Isaac Lab's ``base_ang_vel`` observation.
"""
# from bxi_example_py_elf3.models.rgmt import RgmtExternalReferencePolicy

# policy = RgmtExternalReferencePolicy(
#     motion_npz_path="/home/cheng/rl/whole_body_tracking/inputs/lafan1/dance1_subject2.npz",
#     model_onnx_path="/home/cheng/rl/whole_body_tracking/logs/rsl_rl/elf3_lafan1_extreme_stage1_fsqoff_b0/2026-07-28_11-44-00_extreme_stage1_resume20k_plus10k_gate1/exported/policy_actor.onnx",
#     reference_yaw_mode="initial",  # 实机推荐
# )

# # q/dq：硬件或 MuJoCo 29 关节顺序
# # quat：WXYZ
# # omega：机身坐标系角速度
# target_dof_pos = policy.inference_step(q, dq, quat, omega)

# # 你当前外层循环负责递增，所以不要同时设置 advance=True
# policy.timestep += 1

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort


RGMT_NUM_JOINTS = 29
RGMT_COMMAND_WINDOW_OFFSETS = np.arange(-10, 11, dtype=np.int64)
RGMT_COMMAND_WINDOW_SIZE = 21
RGMT_COMMAND_TOKEN_DIM = 38
RGMT_PROPRIO_TOKEN_DIM = 93
RGMT_PROPRIO_HISTORY_LENGTH = 10
RGMT_OBSERVATION_DIM = 1734

# A vector in the BXI/MuJoCo joint order indexed by this array becomes Isaac's
# articulation order.  The inverse array converts Isaac back to BXI/MuJoCo.
MUJOCO_TO_ISAAC_INDEX = np.asarray(
    [
        15, 22, 0, 16, 23, 1, 17, 24, 2, 18, 25, 3, 9, 19, 26,
        4, 10, 20, 27, 5, 11, 21, 28, 6, 12, 7, 13, 8, 14,
    ],
    dtype=np.int64,
)
ISAAC_TO_MUJOCO_INDEX = np.asarray(
    [
        2, 5, 8, 11, 15, 19, 23, 25, 27, 12, 16, 20, 24, 26, 28,
        0, 3, 6, 9, 13, 17, 21, 1, 4, 7, 10, 14, 18, 22,
    ],
    dtype=np.int64,
)

ELF3_KPS_MUJOCO = np.asarray(
    [
        108.448, 162.672, 176.421,
        176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
        176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
        54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
        54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
    ],
    dtype=np.float32,
)
ELF3_KDS_MUJOCO = np.asarray(
    [
        6.904, 10.356, 11.231,
        11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
        11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
        3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
        3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
    ],
    dtype=np.float32,
)


def _as_finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _parse_csv_floats(metadata: dict[str, str], key: str, length: int) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"ONNX metadata is missing required key {key!r}.")
    try:
        values = np.asarray(
            [float(item) for item in metadata[key].split(",") if item.strip()],
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError(f"ONNX metadata {key!r} is not a numeric CSV vector.") from error
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(
            f"ONNX metadata {key!r} must contain {length} finite values, got {values.shape}."
        )
    return values


def _parse_csv_ints(metadata: dict[str, str], key: str) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"ONNX metadata is missing required key {key!r}.")
    try:
        values = np.asarray(
            [int(float(item)) for item in metadata[key].split(",") if item.strip()],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ValueError(f"ONNX metadata {key!r} is not an integer CSV vector.") from error
    return values


def _normalize_quaternion_wxyz(name: str, value, expected_shape: tuple[int, ...]) -> np.ndarray:
    quaternion = _as_finite_array(name, value, expected_shape)
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{name} contains a zero-length quaternion.")
    return quaternion / norms


def quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float32).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def yaw_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = np.zeros_like(yaw)
    return np.stack((np.cos(0.5 * yaw), zeros, zeros, np.sin(0.5 * yaw)), axis=-1).astype(
        np.float32, copy=False
    )


def quaternion_to_rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("Cannot convert a zero-length quaternion to a rotation matrix.")
    q = quaternion / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3)).astype(np.float32, copy=False)


def quat_rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the WXYZ quaternion's local frame."""

    quaternion = np.asarray(quaternion, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    if quaternion.shape[:-1] != vector.shape[:-1] or quaternion.shape[-1:] != (4,) or vector.shape[-1:] != (3,):
        raise ValueError(
            "quaternion/vector shapes must be [...,4] and [...,3] with identical leading dimensions, "
            f"got {quaternion.shape} and {vector.shape}."
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("quaternion contains a zero-length value.")
    unit = quaternion / norm
    scalar = unit[..., :1]
    xyz = unit[..., 1:]
    cross = 2.0 * np.cross(xyz, vector)
    return (vector - scalar * cross + np.cross(xyz, cross)).astype(np.float32, copy=False)


def build_rgmt_command_window(
    reference_anchor_lin_vel_w,
    reference_anchor_ang_vel_w,
    reference_anchor_quat_w,
    reference_joint_pos,
) -> np.ndarray:
    """Build a clean 21x38 command window with the training-time field order."""

    lin_vel = _as_finite_array(
        "reference_anchor_lin_vel_w", reference_anchor_lin_vel_w, (RGMT_COMMAND_WINDOW_SIZE, 3)
    )
    ang_vel = _as_finite_array(
        "reference_anchor_ang_vel_w", reference_anchor_ang_vel_w, (RGMT_COMMAND_WINDOW_SIZE, 3)
    )
    quaternion = _normalize_quaternion_wxyz(
        "reference_anchor_quat_w", reference_anchor_quat_w, (RGMT_COMMAND_WINDOW_SIZE, 4)
    )
    joint_pos = _as_finite_array(
        "reference_joint_pos", reference_joint_pos, (RGMT_COMMAND_WINDOW_SIZE, RGMT_NUM_JOINTS)
    )
    gravity_w = np.broadcast_to(
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        (RGMT_COMMAND_WINDOW_SIZE, 3),
    )
    tokens = np.concatenate(
        (
            quat_rotate_inverse_wxyz(quaternion, lin_vel),
            quat_rotate_inverse_wxyz(quaternion, ang_vel),
            quat_rotate_inverse_wxyz(quaternion, gravity_w),
            joint_pos,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (RGMT_COMMAND_WINDOW_SIZE, RGMT_COMMAND_TOKEN_DIM):
        raise RuntimeError(f"Internal command layout error: got {tokens.shape}.")
    return tokens


class RgmtExternalReferencePolicy:
    """Run an ELF3 RGMT actor with an external NPZ or live reference source.

    ``q`` and ``dq`` default to BXI/MuJoCo order; reference joints default to
    Isaac/NPZ order; returned targets default to BXI/MuJoCo order.  These
    choices preserve the surrounding interfaces used by ``beyondmimic.py``.

    ``reference_yaw_mode`` controls only the six-dimensional anchor-orientation
    cue.  ``"initial"`` (recommended on hardware) aligns the first reference
    heading to the robot and keeps that transform fixed.  ``"none"`` gives
    exact simulation/world-frame semantics.  ``"continuous"`` preserves the
    old deployment class's locomotion behavior of realigning every step.
    """

    # Repeated inference on one frozen sensor sample creates a fictitious
    # action/state history.  The legacy deployment preheater should call reset
    # and let the first real control frame initialize all ten history slots.
    skip_legacy_preheat = True

    def __init__(
        self,
        motion_npz_path: str | None,
        model_onnx_path: str,
        start_frame: int = 0,
        fixed_pos: bool | None = None,
        *,
        reference_yaw_mode: str = "initial",
        anchor_body_index: int | None = None,
        robot_joint_order: str = "mujoco",
        reference_joint_order: str = "isaac",
        output_joint_order: str = "mujoco",
        calibration_offset_isaac=None,
        providers: Iterable[str] | None = None,
        intra_op_num_threads: int = 4,
    ) -> None:
        if fixed_pos is not None:
            reference_yaw_mode = "initial" if fixed_pos else "continuous"
        if reference_yaw_mode not in {"none", "initial", "continuous"}:
            raise ValueError(
                "reference_yaw_mode must be 'none', 'initial', or 'continuous', "
                f"got {reference_yaw_mode!r}."
            )
        for name, order in (
            ("robot_joint_order", robot_joint_order),
            ("reference_joint_order", reference_joint_order),
            ("output_joint_order", output_joint_order),
        ):
            if order not in {"isaac", "mujoco"}:
                raise ValueError(f"{name} must be 'isaac' or 'mujoco', got {order!r}.")
        if isinstance(intra_op_num_threads, bool) or intra_op_num_threads <= 0:
            raise ValueError("intra_op_num_threads must be a positive integer.")

        self.motion_npz_path = None if motion_npz_path is None else str(motion_npz_path)
        self.model_onnx_path = str(model_onnx_path)
        self.reference_yaw_mode = reference_yaw_mode
        self.fixed_pos = reference_yaw_mode == "initial"
        self.robot_joint_order = robot_joint_order
        self.reference_joint_order = reference_joint_order
        self.output_joint_order = output_joint_order
        self._explicit_anchor_body_index = anchor_body_index

        self.mujoco_to_isaac_idx = MUJOCO_TO_ISAAC_INDEX.tolist()
        self.isaac_to_mujoco_idx = ISAAC_TO_MUJOCO_INDEX.tolist()
        self.kps = ELF3_KPS_MUJOCO.copy()
        self.kds = ELF3_KDS_MUJOCO.copy()

        model_path = Path(self.model_onnx_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        self.metadata = {item.key: item.value for item in model.metadata_props}
        self._load_metadata_contract()
        if calibration_offset_isaac is not None:
            self.calibration_offset = _as_finite_array(
                "calibration_offset_isaac", calibration_offset_isaac, (RGMT_NUM_JOINTS,)
            ).copy()
        self.actual_default_joint_pos_isaac = (
            self.default_joint_pos_isaac + self.calibration_offset
        ).astype(np.float32, copy=False)
        self.default_dof_pos = self.actual_default_joint_pos_isaac[
            ISAAC_TO_MUJOCO_INDEX
        ].copy()
        del model

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(intra_op_num_threads)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ort.get_device() == "GPU"
                else ["CPUExecutionProvider"]
            )
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers),
            sess_options=options,
        )
        self._validate_onnx_io()

        self.motion = None
        self.motioninputpos = None
        self.motionquat = None
        self.motionpos = None
        self.start_frame = 0
        self.end_frame = -1
        self._timestep = 0
        self._reference_yaw_delta = None
        self._proprio_history = None
        self._reference_stream = deque(maxlen=RGMT_COMMAND_WINDOW_SIZE)
        self.raw_action = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action_buffer = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action = self.action_buffer.copy()
        self.obs = np.zeros(RGMT_OBSERVATION_DIM, dtype=np.float32)

        if self.motion_npz_path is not None:
            self.load_motion_npz(self.motion_npz_path, start_frame=start_frame)
        elif start_frame != 0:
            raise ValueError("start_frame requires an NPZ motion source.")

    @classmethod
    def for_live_reference(cls, model_onnx_path: str, **kwargs) -> "RgmtExternalReferencePolicy":
        """Construct a policy whose reference is supplied at inference time."""

        return cls(None, model_onnx_path, **kwargs)

    @property
    def timestep(self) -> int:
        return self._timestep

    @timestep.setter
    def timestep(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"timestep must be an integer, got {type(value).__name__}.")
        value = int(value)
        if value < 0:
            raise ValueError(f"timestep must be non-negative, got {value}.")
        if hasattr(self, "_timestep") and value < self._timestep:
            self._reset_policy_state(clear_reference_stream=False)
        self._timestep = value

    def _load_metadata_contract(self) -> None:
        metadata = self.metadata
        if metadata.get("export_format") != "external_reference_actor":
            raise ValueError(
                "This class requires an actor-only external-reference ONNX. "
                "Re-export with play.py --export_actor_onnx; policy.onnx embeds one motion."
            )
        if metadata.get("quaternion_convention") != "wxyz":
            raise ValueError("The deployment ABI requires quaternion_convention=wxyz.")
        if metadata.get("history_order") != "oldest_to_newest":
            raise ValueError("The deployment ABI requires history_order=oldest_to_newest.")
        if metadata.get("action_semantics") != "reference_joint_position_residual":
            raise ValueError("The ONNX action semantics are not reference-joint residuals.")
        if int(metadata.get("policy_observation_dim", -1)) != RGMT_OBSERVATION_DIM:
            raise ValueError("The ONNX policy observation dimension is not the RGMT 1734-D ABI.")
        if int(metadata.get("policy_action_dim", -1)) != RGMT_NUM_JOINTS:
            raise ValueError("The ONNX policy action dimension is not 29.")

        observation_names = metadata.get("observation_names", "").split(",")
        if observation_names != ["rgmt_command", "motion_anchor_ori_b", "rgmt_proprio"]:
            raise ValueError(f"Unexpected RGMT observation order: {observation_names}.")
        history_lengths = _parse_csv_ints(metadata, "observation_history_lengths")
        if not np.array_equal(history_lengths, np.asarray((1, 1, 10), dtype=np.int64)):
            raise ValueError(f"Unexpected observation history lengths: {history_lengths.tolist()}.")
        command_offsets = _parse_csv_ints(metadata, "command_window_offsets")
        if not np.array_equal(command_offsets, RGMT_COMMAND_WINDOW_OFFSETS):
            raise ValueError(f"Unexpected command offsets: {command_offsets.tolist()}.")

        self.joint_name = metadata["joint_names"].split(",")
        if len(self.joint_name) != RGMT_NUM_JOINTS:
            raise ValueError(f"Expected 29 joint names, got {len(self.joint_name)}.")
        self.default_joint_pos_isaac = _parse_csv_floats(
            metadata,
            "policy_default_joint_pos" if "policy_default_joint_pos" in metadata else "default_joint_pos",
            RGMT_NUM_JOINTS,
        )
        self.action_scale = _parse_csv_floats(
            metadata,
            "policy_action_scale" if "policy_action_scale" in metadata else "action_scale",
            RGMT_NUM_JOINTS,
        )
        self.calibration_offset = _parse_csv_floats(
            metadata, "reference_action_calibration_offset", RGMT_NUM_JOINTS
        )
        action_clip_text = metadata.get("policy_action_clip")
        if action_clip_text is None:
            raise KeyError("ONNX metadata is missing policy_action_clip.")
        if action_clip_text == "none":
            self.action_clip = None
        else:
            self.action_clip = float(action_clip_text)
            if not np.isfinite(self.action_clip) or self.action_clip <= 0.0:
                raise ValueError(f"policy_action_clip must be positive or 'none', got {action_clip_text!r}.")

        self.anchor_body_name = metadata.get("anchor_body_name", "torso_link")
        model_anchor_index = int(metadata.get("motion_anchor_body_index_full", -1))
        if model_anchor_index < 0:
            raise ValueError("ONNX metadata has no valid motion_anchor_body_index_full.")
        self.anchor_body_index = (
            model_anchor_index if self._explicit_anchor_body_index is None else self._explicit_anchor_body_index
        )
        if isinstance(self.anchor_body_index, bool) or not isinstance(self.anchor_body_index, int):
            raise TypeError("anchor_body_index must be an integer.")
        if self._explicit_anchor_body_index is not None and self.anchor_body_index != model_anchor_index:
            raise ValueError(
                f"Explicit anchor_body_index={self.anchor_body_index} conflicts with ONNX metadata "
                f"index {model_anchor_index}."
            )
        self.policy_fps = float(metadata["policy_fps"])
        self.policy_control_dt = float(metadata["policy_control_dt"])
        if not np.isfinite(self.policy_fps) or self.policy_fps <= 0.0:
            raise ValueError("policy_fps must be positive and finite.")

    def _validate_onnx_io(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs":
            raise ValueError(
                "Actor-only ONNX must have exactly one input named 'obs'; "
                f"got {[item.name for item in inputs]}."
            )
        if len(outputs) != 1 or outputs[0].name != "actions":
            raise ValueError(
                "Actor-only ONNX must have exactly one output named 'actions'; "
                f"got {[item.name for item in outputs]}."
            )
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if len(input_shape) != 2 or input_shape[1] != RGMT_OBSERVATION_DIM:
            raise ValueError(f"ONNX obs shape must be [batch,1734], got {input_shape}.")
        if len(output_shape) != 2 or output_shape[1] != RGMT_NUM_JOINTS:
            raise ValueError(f"ONNX actions shape must be [batch,29], got {output_shape}.")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.num_obs = RGMT_OBSERVATION_DIM
        self.num_actions = RGMT_NUM_JOINTS

    def _to_isaac(self, value: np.ndarray, order: str) -> np.ndarray:
        return value if order == "isaac" else value[..., MUJOCO_TO_ISAAC_INDEX]

    def _from_isaac(self, value: np.ndarray, order: str) -> np.ndarray:
        return value if order == "isaac" else value[..., ISAAC_TO_MUJOCO_INDEX]

    def load_motion_npz(self, motion_npz_path: str, *, start_frame: int = 0) -> None:
        """Load or replace the offline reference without touching the ONNX actor."""

        path = Path(motion_npz_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Motion NPZ does not exist: {path}")
        required = (
            "fps", "joint_pos", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
        )
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"Motion NPZ is missing required fields: {missing}.")
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size != 1:
                raise ValueError(f"Motion fps must contain one value, got shape {data['fps'].shape}.")
            motion_fps = float(fps_values[0])
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32).copy()
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float32).copy()
            body_quat = np.asarray(data["body_quat_w"], dtype=np.float32).copy()
            body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32).copy()
            body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32).copy()
            body_names = np.asarray(data["body_names"]).reshape(-1) if "body_names" in data else None

        if not np.isclose(motion_fps, self.policy_fps, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"Motion fps {motion_fps:g} does not match policy fps {self.policy_fps:g}; resample first."
            )
        if joint_pos.ndim != 2 or joint_pos.shape[1] != RGMT_NUM_JOINTS:
            raise ValueError(f"joint_pos must have shape [frames,29], got {joint_pos.shape}.")
        frame_count = joint_pos.shape[0]
        body_count = body_quat.shape[1] if body_quat.ndim == 3 else -1
        expected_shapes = {
            "body_pos_w": (frame_count, body_count, 3),
            "body_quat_w": (frame_count, body_count, 4),
            "body_lin_vel_w": (frame_count, body_count, 3),
            "body_ang_vel_w": (frame_count, body_count, 3),
        }
        arrays = {
            "joint_pos": joint_pos,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_lin_vel,
            "body_ang_vel_w": body_ang_vel,
        }
        for name, array in arrays.items():
            if name in expected_shapes and array.shape != expected_shapes[name]:
                raise ValueError(f"{name} must have shape {expected_shapes[name]}, got {array.shape}.")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinite values.")
        if frame_count < 1 or body_count < 1:
            raise ValueError("Motion NPZ must contain at least one frame and one body.")
        quaternion_norms = np.linalg.norm(body_quat, axis=-1)
        if np.any(quaternion_norms <= np.finfo(np.float32).eps):
            raise ValueError("body_quat_w contains a zero-length quaternion.")
        if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-3:
            raise ValueError("body_quat_w is not normalized within tolerance 1e-3.")

        if body_names is not None:
            decoded_names = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in body_names
            ]
            if len(decoded_names) != body_count:
                raise ValueError(
                    f"body_names has {len(decoded_names)} entries but body arrays have {body_count}."
                )
            if self.anchor_body_name not in decoded_names:
                raise ValueError(f"Motion body_names does not contain anchor {self.anchor_body_name!r}.")
            named_anchor_index = decoded_names.index(self.anchor_body_name)
            if named_anchor_index != self.anchor_body_index:
                raise ValueError(
                    f"Motion anchor {self.anchor_body_name!r} is body index {named_anchor_index}, but ONNX "
                    f"metadata expects {self.anchor_body_index}."
                )
        if self.anchor_body_index < 0 or self.anchor_body_index >= body_count:
            raise IndexError(
                f"anchor_body_index={self.anchor_body_index} is outside motion body range [0,{body_count - 1}]."
            )
        if isinstance(start_frame, bool) or not isinstance(start_frame, int):
            raise TypeError("start_frame must be an integer.")
        if start_frame < 0 or start_frame >= frame_count:
            raise IndexError(f"start_frame must be inside [0,{frame_count - 1}], got {start_frame}.")

        self.motion_npz_path = str(path)
        self.motioninputpos = joint_pos.copy()
        self.motionpos = body_pos
        self.motionquat = body_quat
        self.motion_anchor_quat_w = body_quat[:, self.anchor_body_index]
        self.motion_anchor_lin_vel_w = body_lin_vel[:, self.anchor_body_index]
        self.motion_anchor_ang_vel_w = body_ang_vel[:, self.anchor_body_index]
        self.motion_fps = motion_fps
        self.start_frame = int(start_frame)
        self.end_frame = frame_count - 1
        self._timestep = self.start_frame
        self._reset_policy_state(clear_reference_stream=True)

    def _reset_policy_state(self, *, clear_reference_stream: bool) -> None:
        self.raw_action = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action_buffer = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action = self.action_buffer.copy()
        self._proprio_history = None
        self._reference_yaw_delta = None
        if clear_reference_stream:
            self._reference_stream.clear()

    def reset(self, *, start_frame: int | None = None) -> None:
        """Reset action/history/yaw state and optionally choose a new NPZ frame."""

        if start_frame is None:
            start_frame = self.start_frame
        if self.end_frame >= 0 and not 0 <= start_frame <= self.end_frame:
            raise IndexError(f"start_frame must be inside [0,{self.end_frame}], got {start_frame}.")
        if self.end_frame < 0 and start_frame != 0:
            raise ValueError("A nonzero start_frame requires an NPZ motion.")
        self._timestep = int(start_frame)
        self._reset_policy_state(clear_reference_stream=True)

    def _npz_reference_window(self, timestep: int) -> tuple[np.ndarray, ...]:
        if self.motioninputpos is None:
            raise RuntimeError("No NPZ motion is loaded; use inference_step_with_reference_window instead.")
        center = min(max(int(timestep), 0), self.end_frame)
        frame_indices = np.clip(
            center + RGMT_COMMAND_WINDOW_OFFSETS,
            0,
            self.end_frame,
        )
        return (
            self.motioninputpos[frame_indices],
            self.motion_anchor_quat_w[frame_indices],
            self.motion_anchor_lin_vel_w[frame_indices],
            self.motion_anchor_ang_vel_w[frame_indices],
        )

    def _aligned_reference_quaternion(
        self, robot_quat_w: np.ndarray, reference_quat_w: np.ndarray
    ) -> np.ndarray:
        if self.reference_yaw_mode == "none":
            return reference_quat_w
        if self.reference_yaw_mode == "continuous" or self._reference_yaw_delta is None:
            robot_yaw = yaw_quaternion_wxyz(robot_quat_w)
            reference_yaw = yaw_quaternion_wxyz(reference_quat_w)
            self._reference_yaw_delta = quaternion_multiply_wxyz(
                robot_yaw, quaternion_conjugate_wxyz(reference_yaw)
            )
        aligned = quaternion_multiply_wxyz(self._reference_yaw_delta, reference_quat_w)
        return aligned / np.linalg.norm(aligned).clip(min=np.finfo(np.float32).eps)

    def _anchor_orientation_6d(
        self, robot_quat_w: np.ndarray, reference_quat_w: np.ndarray
    ) -> np.ndarray:
        aligned_reference = self._aligned_reference_quaternion(robot_quat_w, reference_quat_w)
        relative = quaternion_multiply_wxyz(
            quaternion_conjugate_wxyz(robot_quat_w), aligned_reference
        )
        relative /= np.linalg.norm(relative).clip(min=np.finfo(np.float32).eps)
        return quaternion_to_rotation_matrix_wxyz(relative)[:, :2].reshape(-1).astype(np.float32)

    def _push_proprio(self, token: np.ndarray) -> np.ndarray:
        if self._proprio_history is None:
            self._proprio_history = np.repeat(
                token.reshape(1, RGMT_PROPRIO_TOKEN_DIM),
                RGMT_PROPRIO_HISTORY_LENGTH,
                axis=0,
            )
        else:
            self._proprio_history[:-1] = self._proprio_history[1:]
            self._proprio_history[-1] = token
        return self._proprio_history

    def create_obs_input(
        self,
        q,
        dq,
        quat,
        omega,
        *,
        reference_joint_pos_window,
        reference_anchor_quat_window_w,
        reference_anchor_lin_vel_window_w,
        reference_anchor_ang_vel_window_w,
    ) -> np.ndarray:
        """Construct one actor input and advance the 10-token proprio history."""

        q_robot = _as_finite_array("q", q, (RGMT_NUM_JOINTS,))
        dq_robot = _as_finite_array("dq", dq, (RGMT_NUM_JOINTS,))
        robot_quat = _normalize_quaternion_wxyz("quat", quat, (4,))
        base_ang_vel = _as_finite_array("omega", omega, (3,))
        q_isaac = self._to_isaac(q_robot, self.robot_joint_order)
        dq_isaac = self._to_isaac(dq_robot, self.robot_joint_order)

        reference_joint_pos = _as_finite_array(
            "reference_joint_pos_window",
            reference_joint_pos_window,
            (RGMT_COMMAND_WINDOW_SIZE, RGMT_NUM_JOINTS),
        )
        reference_joint_pos = self._to_isaac(reference_joint_pos, self.reference_joint_order)
        reference_quat = _normalize_quaternion_wxyz(
            "reference_anchor_quat_window_w",
            reference_anchor_quat_window_w,
            (RGMT_COMMAND_WINDOW_SIZE, 4),
        )
        command = build_rgmt_command_window(
            reference_anchor_lin_vel_window_w,
            reference_anchor_ang_vel_window_w,
            reference_quat,
            reference_joint_pos,
        )
        anchor_orientation = self._anchor_orientation_6d(
            robot_quat, reference_quat[RGMT_COMMAND_WINDOW_SIZE // 2]
        )
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_quat,
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        )
        proprio = np.concatenate(
            (
                projected_gravity,
                base_ang_vel,
                q_isaac - self.actual_default_joint_pos_isaac,
                dq_isaac,
                self.action_buffer,
            )
        ).astype(np.float32, copy=False)
        if proprio.shape != (RGMT_PROPRIO_TOKEN_DIM,):
            raise RuntimeError(f"Internal proprio layout error: got {proprio.shape}.")
        history = self._push_proprio(proprio)
        self.obs = np.concatenate(
            (command.reshape(-1), anchor_orientation, history.reshape(-1))
        ).astype(np.float32, copy=False)
        if self.obs.shape != (RGMT_OBSERVATION_DIM,):
            raise RuntimeError(f"Internal observation layout error: got {self.obs.shape}.")
        return self.obs.reshape(1, RGMT_OBSERVATION_DIM)

    def _run_actor(self, obs: np.ndarray, reference_joint_pos_isaac: np.ndarray) -> np.ndarray:
        raw_action = self.session.run([self.output_name], {self.input_name: obs})[0]
        self.raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if self.raw_action.shape != (RGMT_NUM_JOINTS,) or not np.isfinite(self.raw_action).all():
            raise ValueError(f"ONNX produced an invalid action with shape {self.raw_action.shape}.")
        if self.action_clip is None:
            clipped_action = self.raw_action
        else:
            clipped_action = np.clip(self.raw_action, -self.action_clip, self.action_clip)
        self.action_buffer = clipped_action.astype(np.float32, copy=True)
        self.action = self.action_buffer.copy()
        target_isaac = (
            reference_joint_pos_isaac
            + self.calibration_offset
            + self.action_scale * self.action_buffer
        )
        return self._from_isaac(target_isaac, self.output_joint_order).astype(np.float32, copy=False)

    def inference_step(
        self,
        q,
        dq,
        quat,
        omega,
        base_lin_vel_b=None,
        robot_pos_w=None,
        advance: bool = False,
    ) -> np.ndarray:
        """Run one NPZ-backed control step; ONNX itself contains no NPZ data."""

        del base_lin_vel_b, robot_pos_w  # RGMT policy ABI does not observe either value.
        joint_pos, anchor_quat, anchor_lin_vel, anchor_ang_vel = self._npz_reference_window(
            self.timestep
        )
        obs = self.create_obs_input(
            q,
            dq,
            quat,
            omega,
            reference_joint_pos_window=joint_pos,
            reference_anchor_quat_window_w=anchor_quat,
            reference_anchor_lin_vel_window_w=anchor_lin_vel,
            reference_anchor_ang_vel_window_w=anchor_ang_vel,
        )
        reference_center_isaac = self._to_isaac(
            joint_pos[RGMT_COMMAND_WINDOW_SIZE // 2], self.reference_joint_order
        )
        target = self._run_actor(obs, reference_center_isaac)
        if advance:
            self._timestep = min(self._timestep + 1, self.end_frame)
        return target

    def inference_step_with_reference_window(
        self,
        q,
        dq,
        quat,
        omega,
        *,
        reference_joint_pos_window,
        reference_anchor_quat_window_w,
        reference_anchor_lin_vel_window_w,
        reference_anchor_ang_vel_window_w,
    ) -> np.ndarray:
        """Run one step from an explicit centered 21-frame reference window."""

        joint_pos = _as_finite_array(
            "reference_joint_pos_window",
            reference_joint_pos_window,
            (RGMT_COMMAND_WINDOW_SIZE, RGMT_NUM_JOINTS),
        )
        obs = self.create_obs_input(
            q,
            dq,
            quat,
            omega,
            reference_joint_pos_window=joint_pos,
            reference_anchor_quat_window_w=reference_anchor_quat_window_w,
            reference_anchor_lin_vel_window_w=reference_anchor_lin_vel_window_w,
            reference_anchor_ang_vel_window_w=reference_anchor_ang_vel_window_w,
        )
        reference_center_isaac = self._to_isaac(
            joint_pos[RGMT_COMMAND_WINDOW_SIZE // 2], self.reference_joint_order
        )
        return self._run_actor(obs, reference_center_isaac)

    def inference_step_streaming(
        self,
        q,
        dq,
        quat,
        omega,
        *,
        reference_joint_pos,
        reference_anchor_quat_w,
        reference_anchor_lin_vel_w,
        reference_anchor_ang_vel_w,
    ) -> np.ndarray | None:
        """Push one live reference frame and run after a 21-frame window exists.

        Without a future predictor, the window center is ten frames behind its
        newest frame.  At 50 Hz this intentionally creates about 200 ms of
        reference delay.  ``None`` is returned while the first 21 frames fill.
        """

        frame = (
            _as_finite_array("reference_joint_pos", reference_joint_pos, (RGMT_NUM_JOINTS,)).copy(),
            _normalize_quaternion_wxyz("reference_anchor_quat_w", reference_anchor_quat_w, (4,)).copy(),
            _as_finite_array("reference_anchor_lin_vel_w", reference_anchor_lin_vel_w, (3,)).copy(),
            _as_finite_array("reference_anchor_ang_vel_w", reference_anchor_ang_vel_w, (3,)).copy(),
        )
        self._reference_stream.append(frame)
        if len(self._reference_stream) < RGMT_COMMAND_WINDOW_SIZE:
            return None
        frames = tuple(self._reference_stream)
        return self.inference_step_with_reference_window(
            q,
            dq,
            quat,
            omega,
            reference_joint_pos_window=np.stack([item[0] for item in frames]),
            reference_anchor_quat_window_w=np.stack([item[1] for item in frames]),
            reference_anchor_lin_vel_window_w=np.stack([item[2] for item in frames]),
            reference_anchor_ang_vel_window_w=np.stack([item[3] for item in frames]),
        )


# Explicit compatibility name for call sites that use the old naming family.
DanceMotionPolicyGravityIsaaclabRGMT = RgmtExternalReferencePolicy


__all__ = [
    "DanceMotionPolicyGravityIsaaclabRGMT",
    "RgmtExternalReferencePolicy",
    "build_rgmt_command_window",
    "quat_rotate_inverse_wxyz",
]
