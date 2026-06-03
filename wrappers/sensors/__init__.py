"""Sensor wrappers.

Wrappers in this subpackage add derived sensing channels to a Forge/Factory env and
log them through the shared ``extras['to_log']`` framework (forwarded to TensorBoard
by :class:`~wrappers.scorers.reward_decomposition.RewardDecompositionWrapper`).

  * :mod:`~wrappers.sensors.contact_sensor_wrapper` — mounts an IsaacLab
    :class:`~isaaclab.sensors.ContactSensor` on the held asset (peg), filtered against
    the fixed asset (hole), and derives a 3-D *in-contact* boolean (one flag per
    task-space translation axis x/y/z) from the per-pair contact-force matrix. The
    sensor must be registered with the scene during env construction, so
    :func:`~wrappers.sensors.contact_sensor_wrapper.install_contact_sensor` is called
    by the runner BEFORE ``gym.make``; the runtime
    :class:`~wrappers.sensors.contact_sensor_wrapper.ContactSensorWrapper` then reads
    the live sensor each step.
"""
