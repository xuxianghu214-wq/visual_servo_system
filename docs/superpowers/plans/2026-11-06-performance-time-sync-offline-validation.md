# UR5 Visual Servo Performance and Time-Sync Implementation Plan

> **For Codex:** Implement each task test-first, keep robot motion disabled during all development tests, and preserve the calibrated YAML files.

**Goal:** Remove diagnostic overhead from real-time paths, correctly classify RTDE timestamps, add hardware-free replay/fault tests, and leave one portable Windows deployment folder.

**Architecture:** Keep the current camera, detector/PnP, EKF, MPC, safety state machine, and hardware adapters. Add a configurable bounded diagnostics pipeline, explicit controller-clock sample classification, and isolated offline replay adapters that reuse the production data types without constructing a robot control interface.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyYAML, pytest, ur_rtde 1.6.5, PowerShell deployment scripts.

---

### Task 1: Configurable low-disturbance diagnostics

**Files:** `config_files/diagnostics_settings.yaml`, `src/visual_servo_system/config/config_loader.py`, `src/visual_servo_system/diagnostics/performance_profiler.py`, `tests/test_config_loader.py`, `tests/test_performance_profiler.py`

1. Add failing tests for `OFF`, `LIGHT`, `FULL_ASYNC`, bounded overflow, non-blocking submission, batched close, and writer-error isolation.
2. Implement strict diagnostics configuration and a one-submission-per-cycle bounded profiler.
3. Run focused tests and the existing performance analysis compatibility tests.

### Task 2: Remove per-stage submissions from critical threads

**Files:** `src/visual_servo_system/camera/camera_gentl_capture.py`, `src/visual_servo_system/vision/marker_pose_detector.py`, `src/visual_servo_system/runtime/visual_servo_orchestrator.py`, `src/visual_servo_system/robot/ur_rtde_state_reader.py`, `src/visual_servo_system/runtime/production_factory.py`, related tests

1. Add tests proving a camera frame, vision frame, and robot sample produce at most one asynchronous diagnostic submission.
2. Collect stage timings locally and submit a fixed-shape batch after useful work completes.
3. Ensure `OFF` avoids detailed timing work and never creates a performance writer.

### Task 3: Correct RTDE clock and sample identity handling

**Files:** `src/visual_servo_system/runtime/device_clock_mapper.py`, `src/visual_servo_system/robot/ur_rtde_state_reader.py`, `src/visual_servo_system/models/data_types.py`, `tests/test_device_clock_mapper.py`, `tests/test_ur_rtde_adapters.py`

1. Add failing tests for accepted, duplicate, delayed, out-of-order, and true-reset samples.
2. Make duplicate/out-of-order controller timestamps non-publishing events; reset only on a configured material backward jump.
3. Add immutable controller timestamp/sample identity and anomaly counters while preserving existing constructors.
4. Prevent zero-wait catch-up bursts after missed receive deadlines.

### Task 4: Offline replay and fault injection

**Files:** `src/visual_servo_system/offline/*.py`, `scripts/run_offline_validation.py`, `tests/test_offline_replay.py`, `tests/test_fault_injection.py`

1. Test deterministic 30/77/125 Hz streams, gaps, drops, duplicates, reordering, delay/jitter, drift, and true reset.
2. Implement image-folder and simulated robot replay with explicit `simulated` provenance.
3. Implement no-motion fault scenarios and assert no control interface is constructed.

### Task 5: Benchmarking, reports, and deployment

**Files:** `scripts/benchmark_diagnostics_modes.py`, `scripts/analyze_performance_log.py`, `docs/离线性能优化与实机验证报告.md`, `docs/实验电脑部署与测试.md`, `README_先看这里.md`, deployment tests

1. Benchmark all three modes with mean/P50/P95/P99, FPS, submission time, drop count, CPU time, and peak memory.
2. Generate an explicitly offline report and hardware-validation checklist.
3. Verify all required installers/wheels/calibrations, run the complete test and lint suites, and verify a clean copied project can bootstrap without hard-coded developer paths.
4. Attempt the requested GitHub baseline backup from an external temporary clone; never create `.git` inside the portable project.
5. After all verification, remove obsolete sibling folders so the workspace root contains only `visual_servo_system`.
