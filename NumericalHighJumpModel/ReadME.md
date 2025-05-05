# Numerical model for high jump

This is **THE** classic video people use to demonstrate a crucial moment in high jump, the take-off.
[youtube video](https://youtube.com/clip/Ugkxb-wanlQlpa3wB1dw5aVSFf4tTin74ZTD?si=IhXdvZv-yrzSNcIa)
The idea behind a good take-off is the athlete should be as stiff as possible such that the horizontal momentum generated during the approach can be converted into vertical momentum without much energy loss.

While it is easy enough to state this as a fact, but there is no concrete numerical studies to precisely describe this motion. In this project, you will create numerical models for the take-off moment with increasing complexity to address the key questions shown below:

## Key questions

1. What is optimal lean angle?
2. How much pressure is the body/stick taking at a specific angle?
3. How does the speed and take-off angle correlates with the maximum height of a jummp.

There are many intermediate steps needed to be taken before one can address these key questions accurately.

## Tasks

1. Start with a 2D stick-ground model, compute the relationship between contact angle and the subsequent evolution of the center of mass of the stick.
2. Modify the ground stick model with a deformable stick to get the pressure over the stick as a function.
3. Use `mujoco` to model the system in 3D and incorporate the curve approach into the computation.
4. Introduce joints and actuators to the `stick`, i.e. turn it into a human body.


## References

1. [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html)
2. [Bouncing ball](https://en.wikipedia.org/wiki/Bouncing_ball)
