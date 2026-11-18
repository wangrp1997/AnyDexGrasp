"""Object-only scatter updates; captured robot/state buffer addresses persist."""
import warp as wp


@wp.kernel
def coordinates(state:wp.array2d(dtype=wp.float32),indices:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.float32),lane:int):
    i=wp.tid();state[lane,indices[i]]=values[i]


@wp.kernel
def forces(state:wp.array2d(dtype=wp.spatial_vector),indices:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.spatial_vector),lane:int):
    i=wp.tid();state[lane,indices[i]]=values[i]
