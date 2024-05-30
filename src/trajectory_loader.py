from src.geometry import *
from src.data_loader import DRONE_RCS_CSV_TO_XARRAY
from src.noise_generator import add_noise_trajectory,generate_cov
from src.misc import draw_3d_lines_and_points_ref
from time import time
import xarray as xr

import numpy as np
import random


import matplotlib.pyplot as plt
from sktime.datatypes import convert
import os,glob,imageio
from tqdm import tqdm

def simulate_trajectories(init_position,time_step_size,vx,yaw_range,pitch_range,roll_range,TN,N_traj):
    """
    @param init_position: the positions in world coordinates of the target. Shape of Number of samples x Number of coordinates (3)
    @param time_step_size: float value for the resolution of euler approximation to kinematic model of drone
    @param vx: float value for the forward heading velocity (x-direction) to kinematic model of drone
    @param yaw_range: float value in radians for the standard deviation of jitter for yaw (along random walk of drone)
    @param pitch_range:  float value in radians for the standard deviation of jitter for pitch (along random walk of drone)
    @param roll_range:  float value in radians for the standard deviation of jitter for roll (along random walk of drone)
    @param TN: int value for the number of time steps we simulate the drone random walk for
    @param N_traj: int value for the number of samples to generate for the data
    @return: yaws,pitchs,rolls,translations numpy arrays of shape (Number of samples x Number of time steps) (rotations) and (Number of samples x Number of timme steps x Dim of coordinates) (translation)
    """
    # initialize empty arrays with for the yaw,pitch,roll andn translation
    # however, randomly initialize initial yaw between 0 and 2pi and initial target position
    # Shape of Number of samples x Number of time steps
    yaws = np.zeros((N_traj,TN))
    yaws[:,0] = np.random.uniform(0,2*np.pi,N_traj)
    pitchs = np.zeros((N_traj,TN))
    rolls = np.zeros((N_traj,TN))

    # Shape of Number of samples x Number of time steps x Dimension of coordinates (3)
    translations = np.zeros((N_traj,TN,3));
    translations[:,0,:] = init_position

    # iterate through each time step of all trajectories simulated
    for i in range(1,TN):
        # initialize the forward facing vector (unit vector in x-axis)
        # Number of coordates + 1 x 1
        forward_dir = np.zeros((4,1)); forward_dir[[0,-1],:] = 1;

        # create the rotation matrices from the yaws,pitchs, and rolls of previous step
        # Shape of Number of Samples x (Dim of coordinates (3) + 1) x (Dim of coordinates (3) + 1)
        yawt = yaw_matrix( yaws[:,i-1])
        pitcht = pitch_matrix(pitchs[:,i-1])
        rollt = roll_matrix(rolls[:,i-1])

        # rotate the target coordinate frame (the forward direction vector)
        # Number of samples x 1 x (Dim of coordinates (3) + 1)
        forward_dir = np.matmul(yawt @ pitcht @ rollt, forward_dir).transpose(0, 2, 1)

        # update the next time step translation, yaw, roll, and pitch (target coordinate frame) based on the previous pose
        # translate the target in the forward direction of the drone
        # forward direction is the current rotation from previous time step
        translations[:,i,:] = translations[:,i-1,:] + vx * time_step_size * forward_dir[:,0,:-1]

        # randomly jitter the yaw,roll, and pitch by samples from normal distribution with std of yaw_range
        # module 2pi because jitter is equivalent and want stability in numerical computation
        yaws[:,i] = (yaws[:,i-1] + np.random.randn(N_traj)*yaw_range) % (2*np.pi)
        rolls[:,i] = (rolls[:,i-1] + np.random.randn(N_traj)*roll_range)  % (2*np.pi)
        pitchs[:,i] = (pitchs[:,i-1] + np.random.randn(N_traj)*pitch_range)  % (2*np.pi)

    return yaws,pitchs,rolls,translations

def simulate_target_trajectory_azim_elev(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,TN,N_traj):

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                              roll_range, TN, N_traj)
    AZ_ts = np.zeros((N_traj, TN))
    EL_ts = np.zeros((N_traj, TN))

    radars = np.zeros((4, 1));
    radars[-1, :] = 1

    for t in range(TN):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_roll @ inv_pitch @ inv_yaw @ inv_trans, radars)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        elevation = elevation * 180 / np.pi;
        azimuth = azimuth * 180 / np.pi;

        # how to account for the negative elevations... what does a negative elevation even mean?

        # when we have negative azimuth, it means that it is flipped because symmetry... (wrapping effect)
        azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

        AZ_ts[:, t] = azimuth.ravel();
        EL_ts[:, t] = elevation.ravel();


    return AZ_ts,EL_ts,(yaws,pitchs,rolls,translations)


def RCS_TO_DATASET_Single_Trajectory(RCS_xarray_dictionary,time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,TN,num_points,verbose=False):
    """
    """
    # dictionary to convert label to digit label
    N_radars = 1

    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []

    yaws = []; pitchs = []; rolls = []; translations = [];


    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")

    drone_sample_count = {drone_key: num_points for drone_key in RCS_xarray_dictionary.keys()}
    start_time = time()
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts > 0])


        AZ_ts,EL_ts,(yaw,pitch,roll,translation) = simulate_target_trajectory_azim_elev(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, TN,
                                             N_traj=num_points)

        # iterate through each drone's RCS "stack"
        for drone, RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already...
            if drone_sample_count[drone] <= 0:
                continue

            # all the valid azimuth and elevationns ffor particular RCS_array
            azimuth_axes = RCS_array.coords["azimuth"].values
            elevation_axes = RCS_array.coords["elevation"].values
            frequency_axes = RCS_array.coords["f[GHz]"].values

            N_freqs = len(frequency_axes)

            # boundary of azimuth and elevations to check for valid sample generated
            valid_azimuths, valid_elevations = [np.min(azimuth_axes), np.max(azimuth_axes)], [np.min(elevation_axes),
                                                                                              np.max(elevation_axes)]

            # the random azimuth and elevation MUST BE WITHIN RANGE OF THE REAL DATA!
            # a sample is only valid if ALL the azimuths and elevations wrt each radar are VALID
            valid_sample_idx = ((AZ_ts <= valid_azimuths[1]) & (AZ_ts >= valid_azimuths[0]) & (
                        EL_ts <= valid_elevations[1]) & (EL_ts >= valid_elevations[0]))
            valid_sample_idx = valid_sample_idx.all(-1)
            azimuth_copy = AZ_ts[valid_sample_idx, :]
            elevation_copy = EL_ts[valid_sample_idx, :]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            try:
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            # number of trajectories x number of time steps x number of frequencies
            RCS_indexed = RCS_indexed.values.T.reshape(-1,TN,N_freqs*N_radars)

            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)

            yaws.append(yaw); pitchs.append(pitch); rolls.append(roll); translations.append(translation);

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

    RCSs = np.vstack(RCSs)
    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    ys = np.vstack(ys)


    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "ys":ys,
                "n_radars":1,
                "n_freq":N_freqs
            }

    end_time = time()
    if verbose:
        print("Dataset Creation Time: {:.3f}".format(end_time-start_time))

    return dataset






def simulate_target_trajectory_azim_elev_multi(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN,N_traj):
    """
    @param time_step_size: float value for the resolution of euler approximation to kinematic model of drone
    @param vx: float value for the forward heading velocity (x-direction) to kinematic model of drone
    @param yaw_range: float value in radians for the standard deviation of jitter for yaw (along random walk of drone)
    @param pitch_range:  float value in radians for the standard deviation of jitter for pitch (along random walk of drone)
    @param roll_range:  float value in radians for the standard deviation of jitter for roll (along random walk of drone)
    @param bounding_box: a numpy array [[x lb, x ub], [y lb, y ub], [z lb, z ub]] denoting the "box" which we sample uav locations
    @param radars: a numpy array [[x1, y1, z1], [x2, y2, z2], ..., [xJ, yJ , zJ]] denoting the radar locations
    @param TN: int value for the number of time steps we simulate the drone random walk for
    @param N_traj: int value for the number of samples to generate for the data
    @return: AZ_ts,EL_ts,(Rho_ts,yaws,pitchs,rolls,translations)
    """

    # simulate the initial world coordinate frame positions of the target for time step 0
    # Number of samples x Dims of coordinates (3)
    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    # simulate the yaws,pitch,rolls, and translations for random walk trajectories of a drone
    # Number of samples x number of  time steps
    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                            roll_range, TN, N_traj)

    # The number of radars in the experiment
    N_radars = radars.shape[0]

    # transform the radar world coordinates to homogenous world coordinates
    # Number of radars x Dims of coordinates + 1 (4)
    radars = np.column_stack((radars,np.ones((N_radars, 1))));

    # initialize the azimuth,elevation, and rho arrays
    # number of simulate points x number of time steps x number of radars
    AZ_ts = np.zeros((N_traj, TN,N_radars))
    EL_ts = np.zeros((N_traj, TN,N_radars))
    Rho_ts = np.zeros((N_traj, TN,N_radars))

    # iterate through each time step of the random walk trajectory
    for t in range(TN):

        # get time step t inverse yaw, pitch, roll, and translation matrix
        # number of simulation points x 4 x 4
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        # get the relative distance between the target and radar in the target coordinate frame
        # number of simulation points x dimension of coordinate (4)  x number of radars
        relative_distances = np.matmul(inv_roll @ inv_pitch @ inv_yaw @ inv_trans, radars.T)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        # convert the radar positions with respect to the target coordinate frame to spherical coordinates
        # number of simulation points x number of radars
        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        # convert the azimuth and elevation from radians to degrees
        elevation = elevation * 180 / np.pi;
        azimuth = azimuth * 180 / np.pi;

        # how to account for the negative elevations... what does a negative elevation even mean?

        # when we have negative azimuth, it means that it is flipped because symetry... (wrapping effect)
        # the target is symmetric along the z-plane (like a box)
        azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180)

        # collect the current time step azimuth,elevation,and rho to the time series data
        AZ_ts[:, t] = azimuth#.ravel();
        EL_ts[:, t] = elevation#.ravel();
        Rho_ts[:,t] = rho

    return AZ_ts,EL_ts,(Rho_ts,yaws,pitchs,rolls,translations)

def RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary,time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN,num_points,random_seed=123,verbose=False):
    """
    @param RCS_xarray_dictionary:
    @param time_step_size: float value for the resolution of euler approximation to kinematic model of drone
    @param vx: float value for the forward heading velocity (x-direction) to kinematic model of drone
    @param yaw_range: float value in radians for the standard deviation of jitter for yaw (along random walk of drone)
    @param pitch_range:  float value in radians for the standard deviation of jitter for pitch (along random walk of drone)
    @param roll_range:  float value in radians for the standard deviation of jitter for roll (along random walk of drone)
    @param bounding_box: a numpy array [[x lb, x ub], [y lb, y ub], [z lb, z ub]] denoting the "box" which we sample uav locations
    @param radars: a numpy array [[x1, y1, z1], [x2, y2, z2], ..., [xJ, yJ , zJ]] denoting the radar locations
    @param TN: int value for the number of time steps we simulate the drone random walk for
    @param num_points: int value for the number of trajectories to generate for the data
    @param random_seed: random seed to set for data reproducibility
    @param verbose: True for debugging print statements
    @return: dataset = { "RCS":RCSs,"azimuth":azimuths,"elevation":elevations,"rho":rhos,"ys":ys, "n_radars":N_radars,"n_freq":N_freqs, "yaws":yaws, "pitchs":pitchs,"rolls":rolls, "translations":translations,"TN": TN,"time_step_size":time_step_size,"n_classes": len(RCS_xarray_dictionary.keys())}
    """
    # dictionary to convert label to digit label
    # Get the number of radars for the experiment
    N_radars = radars.shape[0]

    # initialize empty lists to collect the RCS, class labels,azimuth, elevations, and rho
    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []
    rhos = []

    # initialize empty lists to collect the yaw,pitch,roll and translations
    yaws = []; pitchs = []; rolls = []; translations = [];

    # the number of samples to minimally sample every data generation iteration (per drone)
    batch_size = 250

    # set the random seed for data reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO TRAJECTORY X,y DATASET")

    # initialize dictionary to keep track of the required number of samples left until we reach the goal dataset size
    # {drone class number : number of remaining samples required}
    drone_sample_count = {drone_key: num_points for drone_key in RCS_xarray_dictionary.keys()}
    start_time = time()

    # while there remains a drone class that still requires samples
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts > 0])

        # generate the azimuth and elevations of the target with respect to radar line of sight.
        # number of trajectories x number of time steps x number of radars
        AZ_ts,EL_ts,(Rho_ts,yaw,pitch,roll,translation) = simulate_target_trajectory_azim_elev_multi(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, radars,TN,
                                             N_traj=max(batch_size,num_points))

        # iterate through each drone's RCS "stack"
        for drone, RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already... if it does skip it
            if drone_sample_count[drone] <= 0:
                continue

            # all the valid azimuth and elevationns ffor particular RCS_array
            azimuth_axes = RCS_array.coords["azimuth"].values
            elevation_axes = RCS_array.coords["elevation"].values
            frequency_axes = RCS_array.coords["f[GHz]"].values

            # the number of unique frequencies in the xarray stack
            N_freqs = len(frequency_axes)

            # boundary of azimuth and elevations to check for valid sample generated
            # valid_azimuths, valid_elevations = [np.min(azimuth_axes), np.max(azimuth_axes)], [np.min(elevation_axes),
            #                                                                                   np.max(elevation_axes)]
            valid_azimuths, valid_elevations = [0, 180], [-90, 90]

            # the random azimuth and elevation MUST BE WITHIN RANGE OF THE REAL DATA!
            # a sample is only valid if ALL the azimuths and elevations wrt each radar are VALID

            # Number of trajectories x Number of Time steps x Number of Radars
            valid_sample_idx = ((AZ_ts <= valid_azimuths[1]) & (AZ_ts >= valid_azimuths[0]) & (
                        EL_ts <= valid_elevations[1]) & (EL_ts >= valid_elevations[0]) & np.expand_dims((translation[:,:,-1] > 0),-1))

            # Check if the generated azimuths and elevations (per sample) are valid for all radars at every time step.
            # Number of trajectories
            valid_sample_idx = valid_sample_idx.all(axis=(-2,-1))

            # if there are no valid samples (trajectories), then continue to next drone
            if valid_sample_idx.sum() == 0:
                continue

            # subset only the valid azimuths and elevations to create RCS data
            # keep only the required number of samples remaining based on num_points
            # Number of trajectories x Number of time steps x Number of Radars
            azimuth_copy = AZ_ts[valid_sample_idx,:,:][:num_points]
            elevation_copy = EL_ts[valid_sample_idx,:,:][:num_points]

            # keep only the required number of samples remaining based on num_points
            # Number of trajectories x Number of time steps x Number of Radars
            rho_copy = Rho_ts[valid_sample_idx,:][:num_points]

            # keep only the required number of samples remaining based on num_points
            # Number of trajectories x Number of time steps
            yaw_copy = yaw[valid_sample_idx][:num_points]
            pitch_copy = pitch[valid_sample_idx][:num_points]
            roll_copy = roll[valid_sample_idx][:num_points]

            # Number of trajectories x Number of time steps x Dimensino of coordinates (3)
            # keep only the required number of samples remaining based on num_points
            translation_copy = translation[valid_sample_idx][:num_points]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            # 1x1 array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            # number of frequencies x (number of valid samples * time steps * number of radars)
            try:
                # number of frequencies x (number of valid samples x time steps x number of radars)
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            # # number of trajectories x number of time steps x number of frequencies
            # RCS_indexed = RCS_indexed.values.T.reshape(-1,TN,N_freqs*N_radars)
            #
            # number of trajectories x number of time steps x (number of radars * number of frequencies)
            RCS_indexed = RCS_indexed.values.reshape(N_freqs, *azimuth_copy.shape).transpose(1, 2, 3, 0).reshape(-1,TN,N_radars*N_freqs)
            # example in RCS_indexed[0][2] for 3 freq, 2 radar, 4 time steps, and 5 samples
            # array(['s1 f1 t3 r1', 's1 f2 t3 r1', 's1 f3 t3 r1', 's1 f1 t3 r2',
            #        's1 f2 t3 r2', 's1 f3 t3 r2'])

            # append to RCS,label, azimuth, and elevation, rhs, yaws, pitch, rolls, translations list (add the samples to the current running dataset
            RCSs.append(RCS_indexed)
            ys.append(np.ones((min(valid_sample_idx.sum(),num_points),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)
            rhos.append(rho_copy)
            yaws.append(yaw_copy); pitchs.append(pitch_copy); rolls.append(roll_copy); translations.append(translation_copy);


            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - min(valid_sample_idx.sum(),num_points))

        if verbose:
            print(drone_sample_count)

    # concetate all the data samples along the rows such that
    # RCSs is Number of trajectories x Number of time steps x  (Number of radars * Number of Frequencies))
    RCSs = np.vstack(RCSs)

    # Number of trajectories x Number of time steps x Number of radars
    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    rhos = np.vstack(rhos)

    # Number of trajectories x 1
    ys = np.vstack(ys)

    # Number of trajectories x Number of time steps
    yaws = np.vstack(yaws)
    pitchs = np.vstack(pitchs)
    rolls = np.vstack(rolls)

    # Number of trajectories x Number of time steps x Dim of coordinates (3)
    translations = np.vstack(translations)

    # create a dataset dictionary to keep all generated data organized
    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "rho":rhos,
                "ys":ys,
                "n_radars":N_radars,
                "n_freq":N_freqs,
                "yaws":yaws,
                "pitchs":pitchs,
                "rolls":rolls,
                "translations":translations,
                "TN": TN,
                "time_step_size":time_step_size,
                "n_classes": len(RCS_xarray_dictionary.keys())
            }

    end_time = time()
    # if verbose:
    print("Trajectory Dataset Creation Time: {:.3f}s".format(end_time-start_time))

    return dataset

def simulate_target_gif(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN,plotting_args={"arrow_length": 15, "arrow_linewidth": 2}):
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    N_traj = 1
    photo_dump = os.path.join("..","results","tmp_photo")
    remove_photo_dump = True
    os.makedirs(photo_dump,exist_ok=True)

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                            roll_range, TN, N_traj)
    N_radars = radars.shape[0]
    radars = np.column_stack((radars,np.ones((N_radars, 1))));

    # number of simulate points x number of time steps x number of radars
    frames = []
    for t in tqdm(range(TN)):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_roll @ inv_pitch @ inv_yaw @ inv_trans, radars.T)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        draw_3d_lines_and_points_ref(range_, rho, azimuth, elevation, translations[:,t], yaws[:,t], pitchs[:,t], rolls[:,t], radars[:,:3],
                                     coordinate_system="spherical", ax=ax,plotting_args=plotting_args)

        filename = os.path.join(photo_dump,f"frame_{t}.png")

        # Optionally, you can add grid lines for better visualization
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add some labels (optional)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.savefig(filename)
        frames.append(imageio.imread(filename))
        [line.remove() for line in ax.lines[-N_radars:]]

    plt.close()

    # Save frames as a GIF
    gif_filename = os.path.join("..","results","target_movement.gif")
    imageio.mimsave(gif_filename, frames, duration=0.5)  # Adjust duration as needed
    print(f"GIF saved as '{gif_filename}'")

    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump,"frame_*")):
            os.remove(filename)

def target_with_predictions_gif(dataset,predictions,radars,plotting_args={"arrow_length": 15, "arrow_linewidth": 2}):
    fig = plt.figure(figsize=(28,7))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / dataset["n_classes"]) for i in range(dataset["n_classes"])]

    photo_dump = os.path.join("..","results","tmp_photo")
    remove_photo_dump = False
    os.makedirs(photo_dump,exist_ok=True)

    N_radars = radars.shape[0]
    radars = np.column_stack((radars, np.ones((N_radars, 1))));

    yaws,pitchs,rolls,translations = dataset["yaws"][[0]],dataset["pitchs"][[0]],dataset["rolls"][[0]],dataset["translations"][[0]]
    predictions = predictions[0]
    TN = dataset["TN"]

    forward_coordinates = np.eye(2)

    # number of simulate points x number of time steps x number of radars
    frames = []
    for t in tqdm(range(TN)):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_roll @ inv_pitch @ inv_yaw @ inv_trans, radars.T)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        draw_3d_lines_and_points_ref(range_, rho, azimuth, elevation, translations[:,t], yaws[:,t], pitchs[:,t], rolls[:,t], radars[:,:3],
                                     coordinate_system="spherical", ax=ax1,plotting_args=plotting_args)

        filename = os.path.join(photo_dump,f"frame_{t}.png")

        # Optionally, you can add grid lines for better visualization
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add some labels (optional)
        ax1.set_xlabel("X-axis [m]",fontsize=20,labelpad=5,weight="bold")
        ax1.set_ylabel("Y-axis [m]",fontsize=20,labelpad=5,weight="bold")
        ax1.set_zlabel("Z-axis [m]",fontsize=20,weight="bold")
        label1 = ax1.plot([],[],marker="o",color="r",linestyle="",label="Radar",markersize=10)[0]
        label2 = ax1.plot([],[],marker="o",color="b",linestyle="",label="Target",markersize=10)[0]
        label3 = ax1.plot([],[],linestyle="-",color="r",label="RLOS",markersize=10)[0]
        ax1.legend(handles=[label1,label2,label3],fontsize=30,prop={"weight":"bold"})
        ax1.tick_params(axis='both', which='major', labelsize=15,pad=-1)
        ax1.tick_params(axis='both', which='minor', labelsize=15,pad=-1)

        # title_ = str(np.round(azimuth*180/np.pi).ravel().tolist())
        # title_ = title_ + "\n" + str(np.round(elevation*180/np.pi).ravel().tolist())
        # ax1.set_title(title_)

        for i in range(dataset["n_classes"]):
            ax2.plot(predictions[:t+1,i],linewidth=3,color=colors[i])

        ax2.set_xlim([0,TN])
        ax2.set_ylim([0,1.05])
        ax2.set_ylabel("Class Prediction Probability",fontsize=20,weight="bold")
        ax2.set_xlabel(f"Time ({dataset['time_step_size']} resolution [s])",fontsize=20,weight="bold")
        ax2.set_title(f"True Label {int(dataset['ys'][0].item())}",fontsize=20,weight="bold")
        ax2.legend([f"Drone {i}" for i in np.arange(dataset["n_classes"])],fontsize=30,prop={"weight":"bold"})
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='minor', labelsize=15)

        ax3.plot(translations[:,:t+1,-1].ravel(),'yo-')
        ax3.set_xlabel(f"Time ({dataset['time_step_size']} resolution [s])",fontsize=20,weight="bold")
        ax3.set_ylabel("Z-axis [m]",fontsize=20,weight="bold")
        ax3.set_title("Elevation [m]",fontsize=20,weight="bold")
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax3.tick_params(axis='both', which='minor', labelsize=15)

        rotation_matrix = np.array([[np.cos(yaws[:,t]).item(),-np.sin(yaws[:,t]).item()],
                                    [np.sin(yaws[:,t]).item(),np.cos(yaws[:,t]).item()]])

        tgt_frame = rotation_matrix @ forward_coordinates
        xaxis = tgt_frame[:, 0].ravel()
        yaxis = tgt_frame[:, 1].ravel()

        ax4.plot(translations[:,:t+1,0].ravel(),translations[:,:t+1,1].ravel(),'bo-')
        ax4.plot(radars[:,0],radars[:,1],'ro')

        quiver_artist1 = ax4.quiver(translations[:,t,0].ravel(), translations[:,t,1].ravel(), xaxis[0], xaxis[1], color='g', linewidth=0.5,label="Target X-axis")
        quiver_artist2 = ax4.quiver(translations[:,t,0].ravel(), translations[:,t,1].ravel(), yaxis[0], yaxis[1], color='m', linewidth=0.5,label="Target Y-axis")

        ax4.set_xlabel("X-axis [m]",fontsize=20,weight="bold")
        ax4.set_ylabel("Y-axis [m]",fontsize=20,weight="bold")
        ax4.set_title("X-Y Overview",fontsize=20,weight="bold")
        label1 = ax4.plot([],[],marker="o",color="r",linestyle="",label="Radar")[0]
        label2 = ax4.plot([],[],marker="o",color="b",linestyle="",label="Target")[0]
        ax4.legend(handles=[label1,label2,quiver_artist1,quiver_artist2],fontsize=30,prop={"weight":"bold"})
        ax4.tick_params(axis='both', which='major', labelsize=15)
        ax4.tick_params(axis='both', which='minor', labelsize=15)
        plt.tight_layout(w_pad=3)

        plt.savefig(filename)
        quiver_artist2.remove()
        quiver_artist1.remove()
        frames.append(imageio.imread(filename))
        ax2.legend_ = None
        [line.remove() for line in ax1.lines[-(N_radars+3):]]


    plt.close()

    # Save frames as a GIF
    gif_filename = os.path.join("..","results","target_accuracy_movement.gif")
    imageio.mimsave(gif_filename, frames, duration=0.5)  # Adjust duration as needed
    print(f"GIF saved as '{gif_filename}'")

    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump,"frame_*")):
            os.remove(filename)

def dataset_to_sktime(dataset):

    dataset["RCS"] =  convert(np.swapaxes(dataset["RCS"],1,2),from_type="numpy3D",to_type="pd-multiindex")
    dataset["elevation"] = convert(np.expand_dims(dataset["elevation"],1),from_type="numpy3D",to_type="pd-multiindex")
    dataset["azimuth"] = convert(np.expand_dims(dataset["azimuth"], 1), from_type="numpy3D",to_type="pd-multiindex")
    dataset["ys"] = dataset["ys"].ravel()

def main():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from src.misc import radar_grid
    import xarray as xr
    mpl.use('Qt5Agg')

    TN = 50
    N_traj = 10
    time_step_size = 0.25
    vx = 50
    yaw_range , pitch_range , roll_range = 0,0,0
    xlim = [-50, 50];  ylim = [-50, 50]; zlim = [150, 300]
    bounding_box = np.array([xlim,ylim,zlim])
    num_radars = 4
    SNR = 5
    exponentiate = True

    plotting_args = {"arrow_length": 5, "arrow_linewidth": 2}

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws,pitchs,rolls,translations = simulate_trajectories(init_position,time_step_size, vx, yaw_range, pitch_range, roll_range, TN, N_traj)

    # Creating 4x2 subplots
    fig = plt.figure(figsize=(16, 16))
    n_axes = 1

    radars = radar_grid(n_radars=num_radars,xlim=xlim,ylim=ylim)

    for i in range(n_axes):
        ax = fig.add_subplot(1,n_axes,i+1, projection = '3d')  # you can adjust the size as per your requirement


        yaw = yaw_matrix(yaws[i])
        pitch = pitch_matrix(pitchs[i])
        roll = roll_matrix(rolls[i])
        trans = translation_matrix(translations[i])

        for j in range(TN):
            plot_target_frames(ax,trans[[j]],yaw[[j]],pitch[[j]],roll[[j]],length=plotting_args["arrow_length"],linewidth=plotting_args["arrow_linewidth"])

        ax.plot(radars[0,0],radars[0,1],'ro')
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        ax.set_zlabel("Z")
        ax.set_title(f"Traj {i}")
        ax.axis('equal')
        ax.view_init(10, -5,0)

    plt.show()

    DRONE_RCS_FOLDER =  "../Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER,visualize=False,exponentiate=exponentiate)

    dataset = RCS_TO_DATASET_Single_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                               time_step_size=time_step_size, vx=vx,
                                               yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,bounding_box=bounding_box,
                                               TN=TN,num_points=N_traj,
                                               verbose=True)


    print(dataset)

    covs_single = generate_cov(TraceConstraint=1, d=15, N=10,
                               blocks=1, color="color",
                               noise_method="random")

    # dataset["RCS"] = add_noise_trajectory(dataset["RCS"], 10, covs_single[0])

    dataset_to_sktime(dataset)

    #### ============== SIMULATE MULTIPLE RADARS ================ #####

    # simulate radar positions within some bounding box of limits
    # radars = np.column_stack((
    #     np.random.uniform(xlim[0], xlim[1], num_radars),
    #     np.random.uniform(ylim[0], ylim[1], num_radars),
    #     np.zeros((num_radars,))
    # ))

    # set the random seed for data reproducibility
    np.random.seed(123)
    random.seed(123)

    AZs,ELs,(Rho_ts,yaws,pitchs,rolls,translations)= simulate_target_trajectory_azim_elev_multi(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box,
                                               radars, TN, N_traj)

    dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                               time_step_size=time_step_size, vx=vx,
                                               yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,bounding_box=bounding_box,
                                               TN=TN,radars=radars,
                                               num_points=N_traj,
                                               verbose=True)



    # plot a mapping of azimuth and elevation to RCS
    sample_idx = 4
    frames = []
    for t in tqdm(range(TN)):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].plot(dataset_multi["RCS"][sample_idx,:t+1,0].ravel(),'b-')

        axes[0].plot(drone_rcs_dictionary[dataset_multi["ys"][sample_idx].item()].interp(azimuth=xr.DataArray(dataset_multi["azimuth"][sample_idx,:t+1,0],dims="points1"),
                                                                                         elevation=xr.DataArray(dataset_multi["elevation"][sample_idx,:t+1,0],dims="points1")).loc[26].values,',r--')
        axes[0].set_title("RCS")
        axes[0].set_xlabel("Time Step")

        axes[1].plot(dataset_multi["azimuth"][sample_idx,:t+1,0].ravel())
        axes[1].plot(dataset_multi["elevation"][sample_idx,:t+1,0].ravel())
        axes[1].legend(["Azimuth","Elevation"])
        axes[1].set_title("Target Orientation")
        axes[1].set_xlabel("Time Step")

        temp_data = 10 * np.log10(drone_rcs_dictionary[dataset_multi["ys"][sample_idx].item()].loc[26])
        vmax = np.max(temp_data)*1.5
        vmin = np.min(temp_data)*0.5
        map = xr.plot.imshow(temp_data,vmin=vmin,vmax=vmax,ax=axes[2],cmap="jet")
        axes[2].plot(dataset_multi["elevation"][sample_idx,:t+1,0].ravel(),dataset_multi["azimuth"][sample_idx,:t+1,0].ravel(),linewidth=5,color="k")
        axes[2].plot(dataset_multi["elevation"][sample_idx,0,0].ravel(),dataset_multi["azimuth"][sample_idx,0,0].ravel(),markersize=20,color="purple",marker="o")
        axes[2].plot(dataset_multi["elevation"][sample_idx,t,0].ravel(),dataset_multi["azimuth"][sample_idx,t,0].ravel(),markersize=20,color="purple",marker="*")




        ax = fig.add_subplot(1,4,4, projection = '3d')  # you can adjust the size as per your requirement

        yaw = yaw_matrix(dataset_multi["yaws"][sample_idx])
        pitch = pitch_matrix(dataset_multi["pitchs"][sample_idx])
        roll = roll_matrix(dataset_multi["rolls"][sample_idx])
        trans = translation_matrix(dataset_multi["translations"][sample_idx])

        for j in range(t+1):
            plot_target_frames(ax,trans[[j]],yaw[[j]],pitch[[j]],roll[[j]],length=plotting_args["arrow_length"],linewidth=plotting_args["arrow_linewidth"])

        ax.plot(radars[0,0],radars[0,1],'ro')
        ax.axis('equal')
        ax.view_init(10, -5,0)


        fig.savefig(os.path.join("tmp_images",f"example_trajectory{t}.jpg"))

        frames.append(imageio.imread(os.path.join("tmp_images",f"example_trajectory{t}.jpg")))
        # axes[0].cla()
        # axes[1].cla()
        # axes[2].cla()
        # del map
        # axes[3].cla()
        plt.close()

    # Save frames as a GIF
    gif_filename = os.path.join("..","results","example_trajectory.gif")
    imageio.mimsave(gif_filename, frames, loop=5,duration=0.5)  # Adjust duration as needed



    print("MULTI")
    add_noise_trajectory(dataset_multi["RCS"],SNR=SNR,cov=covs_single[0])
    simulate_target_gif(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, radars, TN,plotting_args=plotting_args)


    # fig = plt.figure(figsize=(16, 16))
    #
    # ax = fig.add_subplot(1,1,1, projection = '3d')  # you can adjust the size as per your requirement
    #
    # yaw = yaw_matrix(dataset_multi["yaws"][0])
    # pitch = pitch_matrix(dataset_multi["pitchs"][0])
    # roll = roll_matrix(dataset_multi["rolls"][0])
    # trans = translation_matrix(dataset_multi["translations"][0])
    #
    # print(yaw)
    #
    # for j in range(TN):
    #     plot_target_frames(ax,trans[[j]],yaw[[j]],pitch[[j]],roll[[j]],length=plotting_args["arrow_length"],linewidth=plotting_args["arrow_linewidth"])
    #
    # ax.plot(radars[0,0],radars[0,1],'ro')
    #
    # plt.show()

    # plt.show()
if __name__ == "__main__":
    main()