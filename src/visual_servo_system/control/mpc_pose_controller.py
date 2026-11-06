"""20 Hz 六维位姿 MPC。

MPC 用双积分模型优化六维加速度。它只输出受约束速度目标；求解失败、超时或数值异常
都会返回零速度的无效结果，绝不静默切换到另一套控制器。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import osqp
from scipy import sparse

from visual_servo_system.geometry.pose_transforms import so3_exp, so3_log, validate_transform
from visual_servo_system.models.data_types import MpcResult, PredictedTrajectory

if TYPE_CHECKING:
    from visual_servo_system.config.config_loader import MpcControllerSettings, SafetyLimits


@dataclass(frozen=True, slots=True)
class MpcPoseControllerConfig:
    dt_s: float = 0.05
    horizon_steps: int = 20
    solver_time_limit_s: float = 0.020
    max_reference_age_s: float = 0.100
    commissioning_linear_speed_mps: float = 0.050
    experiment_linear_speed_mps: float = 0.120
    commissioning_angular_speed_rad_s: float = np.deg2rad(15.0)
    experiment_angular_speed_rad_s: float = np.deg2rad(30.0)
    max_linear_acceleration_mps2: float = 0.25
    max_angular_acceleration_rad_s2: float = np.deg2rad(90.0)

    @classmethod
    def from_settings(
        cls, mpc: MpcControllerSettings, safety: SafetyLimits
    ) -> MpcPoseControllerConfig:
        """把严格 YAML 配置转换为控制器使用的 SI 单位。"""

        return cls(
            dt_s=float(mpc.dt_s),
            horizon_steps=int(mpc.horizon_steps),
            solver_time_limit_s=float(mpc.solver_time_limit_s),
            max_reference_age_s=min(2.0 * float(mpc.dt_s), float(safety.max_latency_s)),
            commissioning_linear_speed_mps=float(safety.commissioning_speed_mps),
            experiment_linear_speed_mps=float(safety.experiment_speed_mps),
            commissioning_angular_speed_rad_s=np.deg2rad(
                safety.commissioning_angular_speed_deg_s
            ),
            experiment_angular_speed_rad_s=np.deg2rad(
                safety.experiment_angular_speed_deg_s
            ),
            max_linear_acceleration_mps2=float(mpc.max_linear_acceleration_mps2),
            max_angular_acceleration_rad_s2=np.deg2rad(
                mpc.max_angular_acceleration_deg_s2
            ),
        )

    def __post_init__(self) -> None:
        positive = (
            self.dt_s,
            float(self.horizon_steps),
            self.max_reference_age_s,
            self.commissioning_linear_speed_mps,
            self.experiment_linear_speed_mps,
            self.commissioning_angular_speed_rad_s,
            self.experiment_angular_speed_rad_s,
            self.max_linear_acceleration_mps2,
            self.max_angular_acceleration_rad_s2,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("MPC dimensions and motion limits must be finite and positive")
        if not np.isfinite(self.solver_time_limit_s) or self.solver_time_limit_s < 0.0:
            raise ValueError("solver_time_limit_s must be finite and non-negative")
        if self.solver_time_limit_s > self.dt_s:
            raise ValueError("solver_time_limit_s must not exceed dt_s")
        if self.max_reference_age_s > 2.0 * self.dt_s:
            raise ValueError("max_reference_age_s must not exceed two MPC periods")


class MpcPoseController:
    """OSQP 热启动 MPC；稀疏矩阵结构只在构造时建立一次。"""

    def __init__(
        self,
        config: MpcPoseControllerConfig | None = None,
        *,
        solver_factory: Callable[[], Any] = osqp.OSQP,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or MpcPoseControllerConfig()
        self._clock = clock
        self._solver_factory = solver_factory
        self.solver_absolute_tolerance = 1e-7
        self.solver_relative_tolerance = 1e-7
        self.solve_count = 0
        self._build_prediction_matrices()
        self._solver = self._create_solver()

    def _create_solver(self) -> Any:
        """创建新 OSQP 代次，不复用上一运动阶段的热启动状态。"""

        solver = self._solver_factory()
        zero_state = np.zeros(12)
        q = self._linear_cost(zero_state)
        lower, upper = self._bounds(np.zeros(6), "commissioning")
        setup_settings: dict[str, Any] = {
            "verbose": False,
            "warm_starting": True,
            "polishing": False,
            "eps_abs": self.solver_absolute_tolerance,
            "eps_rel": self.solver_relative_tolerance,
        }
        if self.config.solver_time_limit_s > 0.0:
            setup_settings["time_limit"] = self.config.solver_time_limit_s
        solver.setup(
            P=self._quadratic_cost,
            q=q,
            A=self._constraint_matrix,
            l=lower,
            u=upper,
            **setup_settings,
        )
        return solver

    def reset(self) -> None:
        """为新机器人授权代次重建求解器并清零计数。"""

        self._solver = self._create_solver()
        self.solve_count = 0

    def _build_prediction_matrices(self) -> None:
        cfg = self.config
        state_size = 12
        control_size = 6
        transition = np.eye(state_size)
        transition[0:6, 6:12] = np.eye(6) * cfg.dt_s
        input_matrix = np.zeros((state_size, control_size))
        input_matrix[0:6] = np.eye(6) * (0.5 * cfg.dt_s**2)
        input_matrix[6:12] = np.eye(6) * cfg.dt_s

        sx = np.zeros((state_size * cfg.horizon_steps, state_size))
        su = np.zeros((state_size * cfg.horizon_steps, control_size * cfg.horizon_steps))
        for row in range(cfg.horizon_steps):
            sx[row * state_size : (row + 1) * state_size] = np.linalg.matrix_power(
                transition, row + 1
            )
            for column in range(row + 1):
                su[
                    row * state_size : (row + 1) * state_size,
                    column * control_size : (column + 1) * control_size,
                ] = np.linalg.matrix_power(transition, row - column) @ input_matrix

        state_weights = np.diag(
            [120.0] * 3 + [80.0] * 3 + [8.0] * 3 + [4.0] * 3
        )
        qbar = sparse.block_diag([state_weights] * cfg.horizon_steps, format="csc")
        rbar = sparse.eye(control_size * cfg.horizon_steps, format="csc") * 0.5
        hessian = su.T @ qbar @ su + rbar
        self._quadratic_cost = sparse.csc_matrix(2.0 * hessian)
        self._sx = sx
        self._su = su
        self._qbar = qbar

        identity = sparse.eye(control_size * cfg.horizon_steps, format="csc")
        cumulative = np.kron(np.tril(np.ones((cfg.horizon_steps, cfg.horizon_steps))), np.eye(6))
        cumulative *= cfg.dt_s
        self._constraint_matrix = sparse.vstack(
            (identity, sparse.csc_matrix(cumulative)), format="csc"
        )

    def _linear_cost(
        self, initial_state: np.ndarray, reference: np.ndarray | None = None
    ) -> np.ndarray:
        if reference is None:
            reference = np.zeros(12 * self.config.horizon_steps)
        prediction_error = self._sx @ initial_state - reference
        return np.asarray(2.0 * self._su.T @ self._qbar @ prediction_error).ravel()

    def _mode_norm_limits(self, mode: str) -> tuple[float, float]:
        if mode == "commissioning":
            return (
                self.config.commissioning_linear_speed_mps,
                self.config.commissioning_angular_speed_rad_s,
            )
        if mode == "experiment":
            return (
                self.config.experiment_linear_speed_mps,
                self.config.experiment_angular_speed_rad_s,
            )
        raise ValueError("mode must be 'commissioning' or 'experiment'")

    def _mode_box_limits(self, mode: str) -> np.ndarray:
        """QP 用保守内接盒，保证三轴任意组合都不越过二范数球。"""

        linear, angular = self._mode_norm_limits(mode)
        root_three = np.sqrt(3.0)
        return np.array([linear / root_three] * 3 + [angular / root_three] * 3)

    def _bounds(self, current_twist: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        root_three = np.sqrt(3.0)
        acceleration = np.array(
            [cfg.max_linear_acceleration_mps2 / root_three] * 3
            + [cfg.max_angular_acceleration_rad_s2 / root_three] * 3
        )
        acceleration = np.tile(acceleration, cfg.horizon_steps)
        speed = np.tile(self._mode_box_limits(mode), cfg.horizon_steps)
        current = np.tile(current_twist, cfg.horizon_steps)
        lower = np.concatenate((-acceleration, -speed - current))
        upper = np.concatenate((acceleration, speed - current))
        return lower, upper

    @staticmethod
    def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm == 0.0:
            return np.array(vector, copy=True)
        return np.asarray(vector) * (limit / norm)

    def _solution_is_safe(
        self, solution: np.ndarray, initial_state: np.ndarray, mode: str
    ) -> tuple[bool, str]:
        """OSQP 返回后按真实二范数逐步复核，不信任容差内的近似可行。"""

        expected_size = 6 * self.config.horizon_steps
        if solution.shape != (expected_size,) or not np.all(np.isfinite(solution)):
            return False, "osqp returned incomplete or non-finite solution"
        controls = solution.reshape(self.config.horizon_steps, 6)
        def tolerance(limit: float) -> float:
            return 5.0 * (
                self.solver_absolute_tolerance
                + self.solver_relative_tolerance * max(abs(limit), 1e-12)
            )

        if np.any(
            np.linalg.norm(controls[:, :3], axis=1)
            > self.config.max_linear_acceleration_mps2
            + tolerance(self.config.max_linear_acceleration_mps2)
        ):
            return False, "predicted linear acceleration constraint violated"
        if np.any(
            np.linalg.norm(controls[:, 3:], axis=1)
            > self.config.max_angular_acceleration_rad_s2
            + tolerance(self.config.max_angular_acceleration_rad_s2)
        ):
            return False, "predicted angular acceleration constraint violated"
        predicted_states = (
            self._sx @ initial_state + self._su @ solution
        ).reshape(self.config.horizon_steps, 12)
        if not np.all(np.isfinite(predicted_states)):
            return False, "predicted MPC state is non-finite"
        linear_limit, angular_limit = self._mode_norm_limits(mode)
        if np.any(
            np.linalg.norm(predicted_states[:, 6:9], axis=1)
            > linear_limit + tolerance(linear_limit)
        ):
            return False, "predicted linear speed constraint violated"
        if np.any(
            np.linalg.norm(predicted_states[:, 9:12], axis=1)
            > angular_limit + tolerance(angular_limit)
        ):
            return False, "predicted angular speed constraint violated"
        return True, "ok"

    @staticmethod
    def _sample_reference_at_absolute_offsets(
        trajectory: PredictedTrajectory, query_offsets_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """按轨迹原始时间锚采样，不把旧轨迹伪装成刚生成的新轨迹。

        查询点位于两个样本之间时，平移和速度线性插值，旋转沿 SO(3) 最短路径
        插值；超过预测尾端时保持最后一个预测状态，不对未知未来作速度外推。
        """

        source_times = trajectory.sample_times_s
        transforms = np.empty((query_offsets_s.size, 4, 4), dtype=np.float64)
        twists = np.empty((query_offsets_s.size, 6), dtype=np.float64)
        for output_index, query_s in enumerate(query_offsets_s):
            upper_index = int(np.searchsorted(source_times, query_s, side="left"))
            if upper_index == 0:
                transforms[output_index] = trajectory.transforms[0]
                twists[output_index] = trajectory.twists[0]
                continue
            if upper_index >= source_times.size:
                transforms[output_index] = trajectory.transforms[-1]
                twists[output_index] = trajectory.twists[-1]
                continue
            if np.isclose(source_times[upper_index], query_s, atol=1e-12, rtol=0.0):
                transforms[output_index] = trajectory.transforms[upper_index]
                twists[output_index] = trajectory.twists[upper_index]
                continue

            lower_index = upper_index - 1
            interval_s = source_times[upper_index] - source_times[lower_index]
            fraction = float((query_s - source_times[lower_index]) / interval_s)
            lower = trajectory.transforms[lower_index]
            upper = trajectory.transforms[upper_index]
            interpolated = np.eye(4)
            interpolated[:3, 3] = lower[:3, 3] + fraction * (
                upper[:3, 3] - lower[:3, 3]
            )
            relative_rotation = lower[:3, :3].T @ upper[:3, :3]
            interpolated[:3, :3] = lower[:3, :3] @ so3_exp(
                so3_log(relative_rotation) * fraction
            )
            transforms[output_index] = interpolated
            twists[output_index] = trajectory.twists[lower_index] + fraction * (
                trajectory.twists[upper_index] - trajectory.twists[lower_index]
            )
        return transforms, twists

    @staticmethod
    def _zero_result(
        timestamp_s: float,
        status: str,
        trajectory: PredictedTrajectory | None = None,
        solve_time_s: float = 0.0,
    ) -> MpcResult:
        return MpcResult(
            host_timestamp_s=float(timestamp_s),
            commanded_twist=np.zeros(6),
            solve_time_s=float(max(0.0, solve_time_s)),
            solver_status=status,
            valid=False,
            quality=0.0,
            predicted_trajectory=trajectory,
        )

    def solve(
        self,
        T_base_flange: np.ndarray,
        current_twist: np.ndarray,
        desired_flange_trajectory: PredictedTrajectory,
        host_timestamp_s: float,
        mode: str,
    ) -> MpcResult:
        """求解一次控制量并返回下一控制周期的速度目标。"""

        timestamp = float(host_timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("host_timestamp_s must be finite and non-negative")
        current_pose = validate_transform(T_base_flange)
        twist = np.asarray(current_twist, dtype=np.float64)
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            raise ValueError("current_twist must be a finite six-vector")
        self._mode_norm_limits(mode)  # 在调用求解器前拒绝拼写错误。
        reference_age_s = timestamp - desired_flange_trajectory.host_timestamp_s
        timestamp_tolerance_s = 1e-9
        if reference_age_s < -timestamp_tolerance_s:
            return self._zero_result(
                timestamp,
                "reference timestamp is in the future",
                desired_flange_trajectory,
            )
        if reference_age_s > self.config.max_reference_age_s + timestamp_tolerance_s:
            return self._zero_result(
                timestamp,
                "reference timestamp is stale",
                desired_flange_trajectory,
            )
        if not desired_flange_trajectory.valid:
            return self._zero_result(
                timestamp,
                f"reference invalid: {desired_flange_trajectory.reason}",
                desired_flange_trajectory,
            )
        if desired_flange_trajectory.transforms.shape[0] < self.config.horizon_steps:
            return self._zero_result(timestamp, "reference horizon is too short")
        expected_times = (
            np.arange(1, self.config.horizon_steps + 1, dtype=np.float64) * self.config.dt_s
        )
        if not np.allclose(
            desired_flange_trajectory.sample_times_s[: self.config.horizon_steps],
            expected_times,
            atol=1e-9,
            rtol=0.0,
        ):
            return self._zero_result(timestamp, "reference sample times do not match MPC dt")
        if self.config.solver_time_limit_s == 0.0:
            return self._zero_result(timestamp, "solver time limit is zero")

        query_offsets_s = expected_times + max(reference_age_s, 0.0)
        reference_transforms, reference_twists = (
            self._sample_reference_at_absolute_offsets(
                desired_flange_trajectory, query_offsets_s
            )
        )
        desired_pose = reference_transforms[0]
        initial_pose = np.concatenate(
            (
                current_pose[:3, 3] - desired_pose[:3, 3],
                so3_log(current_pose[:3, :3] @ desired_pose[:3, :3].T),
            )
        )
        initial_state = np.concatenate((initial_pose, twist))
        reference_states = np.zeros((self.config.horizon_steps, 12))
        for index in range(self.config.horizon_steps):
            reference_pose = reference_transforms[index]
            reference_states[index, 0:3] = reference_pose[:3, 3] - desired_pose[:3, 3]
            reference_states[index, 3:6] = so3_log(
                reference_pose[:3, :3] @ desired_pose[:3, :3].T
            )
            reference_states[index, 6:12] = reference_twists[index]
        linear_cost = self._linear_cost(initial_state, reference_states.ravel())
        lower, upper = self._bounds(twist, mode)

        started = self._clock()
        self.solve_count += 1
        try:
            self._solver.update(q=linear_cost, l=lower, u=upper)
            result = self._solver.solve(raise_error=False)
        except Exception as error:  # 厂商/求解器边界必须故障闭锁，具体原因写入日志字段。
            elapsed = float(max(0.0, self._clock() - started))
            return self._zero_result(
                timestamp,
                f"osqp exception: {type(error).__name__}: {error}",
                desired_flange_trajectory,
                elapsed,
            )
        elapsed = float(self._clock() - started)
        if elapsed > self.config.solver_time_limit_s:
            return MpcResult(
                timestamp,
                np.zeros(6),
                elapsed,
                "solver time limit exceeded",
                False,
                0.0,
                desired_flange_trajectory,
            )
        try:
            # 结果对象来自 C 扩展；属性访问、字符串化和数组转换都可能抛异常。
            status = str(result.info.status).strip().lower()
            solution_array = np.array(result.x, dtype=np.float64, copy=True)
        except Exception as error:
            return self._zero_result(
                timestamp,
                f"osqp result parse exception: {type(error).__name__}: {error}",
                desired_flange_trajectory,
                elapsed,
            )
        if status != "solved":
            return MpcResult(
                timestamp,
                np.zeros(6),
                elapsed,
                f"osqp {status}",
                False,
                0.0,
                desired_flange_trajectory,
            )
        safe_solution, unsafe_reason = self._solution_is_safe(
            solution_array, initial_state, mode
        )
        if not safe_solution:
            return MpcResult(
                timestamp,
                np.zeros(6),
                elapsed,
                unsafe_reason,
                False,
                0.0,
                desired_flange_trajectory,
            )

        acceleration = np.concatenate(
            (
                self._clip_norm(
                    solution_array[:3], self.config.max_linear_acceleration_mps2
                ),
                self._clip_norm(
                    solution_array[3:6], self.config.max_angular_acceleration_rad_s2
                ),
            )
        )
        commanded = twist + acceleration * self.config.dt_s
        linear_speed_limit, angular_speed_limit = self._mode_norm_limits(mode)
        commanded = np.concatenate(
            (
                self._clip_norm(commanded[:3], linear_speed_limit),
                self._clip_norm(commanded[3:], angular_speed_limit),
            )
        )
        return MpcResult(
            host_timestamp_s=timestamp,
            commanded_twist=commanded,
            solve_time_s=elapsed,
            solver_status=status,
            valid=True,
            quality=float(np.clip(desired_flange_trajectory.quality, 0.0, 1.0)),
            predicted_trajectory=desired_flange_trajectory,
        )
