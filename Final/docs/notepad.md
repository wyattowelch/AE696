### Questions about Observations
- how to normalize satellite vectors?
    - just doing 1 observation at each event, just add box_limit+/-1km as the max/min
- observation of an event type?
    - maybe just input the maneuver options if it doesnt rly need to know what type

    - "arming" actions by passing a state that says 
    - fielder
    - big negative reward <<--- Easiest option
- estimating time until next observation


**Observations:**
Position error from ideal, velocity error from ideal
$$
[\delta x, \delta y, \delta z, \delta vx, \delta vy, \delta vz]
$$

Box margin
$$
[min(box_{x}-x, -box_{x}-x), min(...y), min(...z)]
$$

Normalize the position to be +/- the optimal area, so the box, meaning if pos>1.0, then terminate bc its out of the box.

Time to decision ($t_{td}$) - time until next event. Just save the current propagation time then push forward to theV next and subtract the time difference... easy, right?? -_-
