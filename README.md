# video-sequences-data-augmentation-tool

This tool generates groups of small local trajectories given a global trajectory. 
Local trajectories can be created in forward or backward direction.
It is possible to define constant or variable steps between frames.
Associated frame poses are also generated.
Local trajectories can have an overlap with respect to the previous trajectory.
Final local trajectories can be randomly shuffled.
Daatset is saved in a pandas datafile which contains the associated directory of the frames and 
pose in euler, quaterion or matrix representation.
Test script plots live the local trajectories generated and the associated global trajectory segment.

Any improvement to the tool is more than welcome.
