#!/usr/bin/env python
#

#
# A bouncing ball simulator.
import math
import os.path as osp
import time
from copy import deepcopy

import imageio

from tune.utils import *

DEBUG = False
# DEBUG = True
box_size = np.array([0.2, 0.2, 0.2])
obj_start_pos = [0, 0, 0]
obj_start_ori = [0, 0, 0, 1]
RADIUS = 0.5
friction = 0.2
mass = 1.0

hz = 60
length_in_seconds = 400/hz
num_timesteps = int(length_in_seconds*hz)

save_hz = 60
save_skip_frames = int(hz / save_hz)
num_datapoints = int(save_hz * length_in_seconds)

render_width = 100
render_height = 100

dataset_types = ['train', 'val']
TRAIN_COUNT = 800
VAL_COUNT = 100
sizes = [TRAIN_COUNT, VAL_COUNT]


class BallSim(object):
    """
    A single simulation with a bouncing ball. Customizable for a variety of datset situations.
    """

    def __init__(self, ball_radius=RADIUS, debug=DEBUG):
        self.ball_radius = ball_radius
        self.debug = debug

    def __enter__(self):
        setup_physics(gui=self.debug)
        p.setTimeStep(1 / hz)
        p.setRealTimeSimulation(False)

        self.view_matrix = p.computeViewMatrix(cameraEyePosition=[3.0, 2.8, 3.2], cameraTargetPosition=[0.0, 0.0, 1.0],
                                               cameraUpVector=[0.0, 0.0, 1.0])
        self.projection_matrix = p.computeProjectionMatrixFOV(40, 1.0, 0.01, 100)

        # spawn objects. Their physics parameters will be adjusted later.
        self.plane_id = p.loadURDF("plane.urdf")
        obj_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=self.ball_radius)
        self.obj_id = p.createMultiBody(baseMass=1.,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=obj_shape_id,
                                        baseVisualShapeIndex=obj_shape_id,
                                        basePosition=obj_start_pos,
                                        baseOrientation=obj_start_ori)

        return self

    def __exit__(self, _type, value, traceback):
        p.disconnect()
        
    def set_physics_params(self, restitution):
        p.setGravity(0, 0, -9.8)
        p.changeDynamics(self.plane_id, -1, restitution=1, lateralFriction=friction)
        p.changeDynamics(self.obj_id, -1, mass=mass, restitution=restitution, lateralFriction=friction)
        
    def run(self, zeta, obj_start_ang_vel=None, render=False,
            random_render=False, video_path=None):
        # print("Running BallSim with parameter vector {}".format(zeta))
        restitution, obj_start_pos = zeta
        if np.isscalar(obj_start_pos):
            obj_start_pos = [0, 0, obj_start_pos]
        if obj_start_ang_vel is None:
            obj_start_ang_vel = [0, 0, 0]
        start_position_list = np.empty([1])
        physics_list = np.empty([3])
        position_list = np.empty([num_datapoints, 3])
        orientation_list = np.empty([num_datapoints, 4])
        linear_velocity_list = np.empty([num_datapoints, 3])
        angular_velocity_list = np.empty([num_datapoints, 3])
        rgb_writer = None
        if render:
            rgb_list = np.empty([num_datapoints, render_width, render_height, 3], dtype=np.uint8)
            depth_list = np.empty([num_datapoints, render_width, render_height], dtype=np.uint8)
            seg_list = np.empty([num_datapoints, render_width, render_height], dtype=np.uint8)
            if video_path is not None:
                # save video file
                video_path = video_path
                makedirs("video")
                print("saving video to " + video_path)
                rgb_writer = imageio.get_writer(video_path, fps=hz, quality=10)
        else:
            rgb_list = depth_list = seg_list = None

        p.resetBasePositionAndOrientation(self.obj_id, obj_start_pos, obj_start_ori)
        p.resetBaseVelocity(self.obj_id, [0, 0, 0], obj_start_ang_vel)

        self.set_physics_params(restitution)

        # save variables that are single-item (aren't of length num_timesteps)
        start_position_list[0] = deepcopy(obj_start_pos[2])
        physics_list[:] = [restitution, friction, mass]

        if random_render:
            angle = np.random.random()*2*math.pi
            distance = np.random.random()*5+5
            x = math.cos(angle) * distance
            y = math.sin(angle) * distance
            new_camera_position = x, y, np.random.random()*3 + 5
            self.view_matrix = p.computeViewMatrix(cameraEyePosition=new_camera_position,
                                                   cameraTargetPosition=[0.0, 0.0, 2.0],
                                                   cameraUpVector=[0.0, 0.0, 1.0])
        # time.sleep(1 / hz)
        for t in range(num_timesteps):
            p.stepSimulation()
            if self.debug:
                time.sleep(1 / hz)

            if t % save_skip_frames == 0:
                # get position. orientation, and velocity, and store them
                box_pos, box_ori = p.getBasePositionAndOrientation(self.obj_id)
                lin_vel, ang_vel = p.getBaseVelocity(self.obj_id)
                position_list[t // save_skip_frames, :] = deepcopy(box_pos)
                orientation_list[t // save_skip_frames, :] = deepcopy(box_ori)
                linear_velocity_list[t // save_skip_frames, :] = deepcopy(lin_vel)
                angular_velocity_list[t // save_skip_frames, :] = deepcopy(ang_vel)

                if render:
                    # now render and store those
                    _, _, rgb, depth, seg = p.getCameraImage(render_width, render_height, viewMatrix=self.view_matrix,
                                                             projectionMatrix=self.projection_matrix,
                                                             shadow=False, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                    rgb_list[t // save_skip_frames, :, :] = rgb[:, :, :3]
                    depth_list[t // save_skip_frames, :, :] = depth[:, :]
                    seg_list[t // save_skip_frames, :, :] = seg[:, :]

                    if video_path:
                        rgb_writer.append_data(rgb[:, :, :3])

        output = start_position_list, physics_list, position_list, orientation_list, linear_velocity_list, \
            angular_velocity_list, rgb_list, depth_list, seg_list
        if render and video_path:
            rgb_writer.close()
        return output

    def generate_tune_dataset(self, dataset_name, plane_jitter, random_render, uniform_difference=False,
                              train_count=None, render=True, range_override=None):
        if train_count is None:
            train_count = TRAIN_COUNT
        sizes[0] = train_count
        overall_start_time = time.time()
        dataset_base_path = get_dataset_base_path(dataset_name)
        for dataset_type, dataset_size in zip(dataset_types, sizes):
            for run_id in range(dataset_size):
                # Make a directory for this data point
                makedirs(osp.join(dataset_base_path, dataset_type, str(run_id)))

                # set up physics

                obj_start_pos[2] = np.random.random() + 4

                # restitution, restitution2 = 0, 0

                if not uniform_difference:
                    restitution = np.random.uniform(0.3, 0.7)
                    restitution2 = np.random.uniform(0.3, 0.7)
                else:
                    restitution = np.random.uniform(0.5, 0.7)
                    restitution2 = restitution + np.random.uniform(-0.2, 0.2)
                if range_override is not None:
                    restitution = np.random.uniform(*range_override)
                    restitution2 = np.random.uniform(*range_override)

                obj_start_ang_vel = np.random.normal(0, plane_jitter, [3]).tolist()

                # save physics world so we have a reference in case we need it later
                p.saveBullet(osp.join(dataset_base_path, dataset_type, str(run_id), "world_state.bullet"))
                # start simulation
                start_time = time.time()

                start_position_list, physics_list1, position_list1, _, linear_velocity_list1, _, rgb_list1, depth_list1, seg_list1 = \
                    self.run([restitution, obj_start_pos], obj_start_ang_vel, render=render,
                             random_render=random_render)
                _, physics_list2, position_list2, _, linear_velocity_list2, _, rgb_list2, depth_list2, seg_list2 = \
                    self.run([restitution2, obj_start_pos], obj_start_ang_vel, render=render,
                             random_render=random_render)
                run_end_time = time.time()

                # p.resetBasePositionAndOrientation(self.plane_id, plane_orig_pos, plane_orig_ori)
                print("Runs completed: {} Run time: {} Total elapsed time: {}s".format(run_id, run_end_time - start_time, run_end_time - overall_start_time))

                def make_name(name):
                    return osp.join(dataset_base_path, dataset_type, str(run_id), name + ".npy")

                # print('save to' + make_name("start_position"))
                np.save(make_name("start_position"), start_position_list, allow_pickle=False)
                np.save(make_name("physics1"), physics_list1, allow_pickle=False)
                np.save(make_name("physics2"), physics_list2, allow_pickle=False)
                np.save(make_name("position1"), position_list1, allow_pickle=False)
                np.save(make_name("position2"), position_list2, allow_pickle=False)
                np.save(make_name("linear_velocity_list1"), linear_velocity_list1, allow_pickle=False)
                np.save(make_name("linear_velocity_list2"), linear_velocity_list2, allow_pickle=False)
                if render:
                    np.save(make_name("rgb1"), rgb_list1, allow_pickle=False)
                    np.save(make_name("seg1"), seg_list1, allow_pickle=False)
                    np.save(make_name("rgb2"), rgb_list2, allow_pickle=False)
                    np.save(make_name("seg2"), seg_list2, allow_pickle=False)
