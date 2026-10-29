# Orbit control scaffold

Set `ORBIT_CONTROL_ENABLED = True` in
[`scripts/simulation_template.py`](../scripts/simulation_template.py) to use
the optional event-aware propagation path. The existing one-call propagation
path remains active by default. In control mode, SatKit propagates one
`CONTROL_CHECK_INTERVAL_S` window at a time. The returned samples are checked
for node crossings (`z = 0`) and apsides (`r · v = 0`). The initial event
bracket is refined by repeated SatKit propagation to a 0.1 second time
tolerance, then the state is interpolated over that short interval. The policy
is queried there, an impulsive RTN delta-v is applied, and the next SatKit
window starts with the updated velocity.

## Integration checklist

- [ ] Implement the DQN observation encoding in `_policy_action` to match the
  model's training environment, including units, scaling, and event encoding.
- [ ] Load the trained model and pass it as `OrbitControlConfig.policy` from
  the script. The current `None` policy intentionally selects action zero.
- [ ] Replace `MANEUVER_ACTIONS_RTN_M_S` with the action set used during
  training, or replace the discrete mapping with the desired maneuver format.
- [ ] Confirm the policy output format and map its action to an index or a
  delta-v vector.
- [ ] Expose SatKit's dense interpolant from `propagator.py` if repeated
  propagation during event refinement is too expensive.
- [ ] Decide whether trajectory output should store both pre-maneuver and
  post-maneuver states at the same event time. The scaffold stores the
  post-maneuver state in the trajectory and retains both in `event_log`.
- [ ] Add policy/reward/episode handling if the DQN is being trained online;
  this scaffold only calls an already available prediction interface.

The control check interval bounds when crossings are searched. `RunConfig.dt`
sets the sampling grid used to find an initial bracket; the bracket is then
refined to 0.1 seconds. Reduce both intervals to catch narrow event windows
reliably.
Orbital node and apsis checks are skipped when the event is not bracketed by
the returned samples. Each check window stops at its first detected event and
restarts from that interpolated point after the action.
