"""Low-level control wrappers (copied/adapted from RoboNuke/Continuous_Force_RL).

These wrappers sit *between* the policy and an Isaac Lab Factory/Forge env. They
expand the action space with extra control channels (stiffness gains, force
targets, force/position selection) and override the env's ``_pre_physics_step`` /
``_apply_action`` to run operational-space (task-space) control, writing joint
effort + position targets to the articulation each physics step.

Modules:
  * :mod:`~wrappers.controllers.factory_control_utils` — shared task-space control
    math (pose PD wrench, force PID wrench, wrench→joint-torque via Jᵀ + null space).
  * :mod:`~wrappers.controllers.vic_pose_wrapper` — Variable Impedance Control:
    policy additionally commands translational stiffness (Kp) gains.
  * :mod:`~wrappers.controllers.hybrid_force_position_wrapper` — hybrid
    force/position control with a per-axis selection matrix gated by a fixed 6-D
    ``force_axes`` eligibility vector (from ``ControlCfg.force_axes``). Sources
    force-torque from the Forge env's own ``force_sensor_smooth``. Its pose control is
    bit-exact with the base env (targets + dead zone + Jᵀ + null space).
  * :mod:`~wrappers.controllers.hybrid_vic_wrapper` — unification: hybrid force/position
    control PLUS full 6x6 variable-impedance (the policy outputs stiffness K and damping D
    matrices). With ``force_axes`` all-zero it reduces to a full-matrix VIC.

Shared bit-exact control math (target generation + diagonal/matrix motion wrench + dead
zone) lives in :mod:`~wrappers.controllers.factory_control_utils`.
"""
