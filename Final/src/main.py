"""Headless entry point: configure, initialize, propagate, and inspect one orbit."""

from propagator import propagate
from scenario import EARTH, RunConfig, validate_run_config
from state import elements_to_state

def main() -> None:
    """
    Coordination modules. No Orbital Equations in main.
    """

    # Construct SI initial state, propagate, and display sampled output.
    config = RunConfig()
    validate_run_config(config)
    initial_state = elements_to_state(
        config.orbit,
        mu=EARTH.mu,
    )
    states, elements = propagate(initial_state, config.dt_End, config)
    final_state = states[-1]
    final_elements = elements[-1]



    print(f"Samples: {len(states)}")
    print(f"Final elapsed time: {final_state.t:.3f} s")
    print(f"Final position [m]: {final_state.r}")
    print(f"Final velocity [m/s]: {final_state.v}")
    print(f"Final elements [?]: {final_elements}")


if __name__ == "__main__":
    main()
