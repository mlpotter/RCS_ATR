import numpy as np


# TODO: VECTORIZE YAW PITCH ROLL
# https://msl.cs.uiuc.edu/planning/node102.html
#
def inverse_roll_matrix(rolls):
    inverse_roll_rot = [np.expand_dims(np.array([[1,0,0,0],[0,np.cos(rolli),np.sin(rolli),0],[0,-np.sin(rolli),np.cos(rolli),0],[0,0,0,1]]),0) for rolli in rolls]

    return np.concatenate(inverse_roll_rot,0)

def inverse_pitch_matrix(pitchs):
    inverse_pitch_rot = [np.expand_dims(np.array([[np.cos(pitchi),0,-np.sin(pitchi),0],[0,1,0,0],[np.sin(pitchi),0,np.cos(pitchi),0],[0,0,0,1]]),0) for pitchi in pitchs]

    return np.concatenate(inverse_pitch_rot,0)

def inverse_yaw_matrix(yaws):
    inverse_yaw_rot = [np.expand_dims(np.array([[np.cos(yawi),np.sin(yawi),0,0],[-np.sin(yawi),np.cos(yawi),0,0],[0,0,1,0],[0,0,0,1]]),0) for yawi in yaws]

    return np.concatenate(inverse_yaw_rot,0)

def inverse_translation_matrix(translations):
    inverse_translations = [np.expand_dims(np.array([[1,0,0,-transi[0]],[0,1,0,-transi[1]],[0,0,1,-transi[2]],[0,0,0,1]]),0) for transi in translations]

    return np.concatenate(inverse_translations,0)

def roll_matrix(rolls):
    roll_rot = [np.expand_dims(np.array([[1,0,0,0],[0,np.cos(rolli),-np.sin(rolli),0],[0,np.sin(rolli),np.cos(rolli),0],[0,0,0,1]]),0) for rolli in rolls]

    return np.concatenate(roll_rot,0)

def pitch_matrix(pitchs):
    pitch_rot = [np.expand_dims(np.array([[np.cos(pitchi),0,np.sin(pitchi),0],[0,1,0,0],[-np.sin(pitchi),0,np.cos(pitchi),0],[0,0,0,1]]),0) for pitchi in pitchs]

    return np.concatenate(pitch_rot,0)

def yaw_matrix(yaws):
    yaw_rot = [np.expand_dims(np.array([[np.cos(yawi),-np.sin(yawi),0,0],[np.sin(yawi),np.cos(yawi),0,0],[0,0,1,0],[0,0,0,1]]),0) for yawi in yaws]

    return np.concatenate(yaw_rot,0)

def translation_matrix(translations):
    translations = [np.expand_dims(np.array([[1,0,0,transi[0]],[0,1,0,transi[1]],[0,0,1,transi[2]],[0,0,0,1]]),0) for transi in translations]

    return np.concatenate(translations,0)

def cartesian2aer(x,y,z):
    range_ = np.sqrt(x**2+y**2)

    # get the distance between target and radar
    rho = np.sqrt(x**2+y**2+z**2)


    # azimuth = tan(azimuth) = y/x
    azimuth = np.arctan2(y, x)


    # elevation = tan(elevation) = y/z
    # negative delta z to be consistennt with RCS DRONE paper dataset (drone Z-axis is DOWNWARD)
    elevation = np.arctan2(y,-z)

    return range_,rho,azimuth,elevation

def aer2cartesian(range_,rho,azimuth,elevation):
    # get the relative vector with respect to target coordinate frame
    # number of points x number of radars
    deltax = np.cos(azimuth) * range_
    deltay = np.sin(azimuth) * range_

    # this will error out if you have a ZERO elevation. Or otherwise stated, you are EXACTLY along the same x-axis line as target.
    # negative to be consistent with AER COORDINATES
    deltaz = -deltay / (np.tan(elevation))

    return deltax,deltay,deltaz

# TODO: REALLY CHECK IF THESE C2S and S2C are correct for Radar
def cartesian2spherical(x,y,z):
    # get the distance between target and radar
    rho = np.sqrt(x**2 + y**2 + z**2)

    # azimuth = tan(azimuth) = y/x
    azimuth = np.arctan2(y, x)

    # negative to assume Z-axis is looking from down
    # (-1)**(x<0) to denote if object is BEHIND, or BELOW
    # print("Elevation PRE: ",np.arccos(-z / rho) * 180/np.pi )
    elevation = (-1)**(x<0).astype(float) * np.arccos(-z / rho)
    # print("Elevation POST: ",elevation)
    # return spherical coordinates
    return None,rho,azimuth,elevation

def spherical2cartersian(range_,rho,azimuth,elevation):
    deltax = rho * np.sin((-1)**(elevation<0) * elevation) * np.cos(azimuth)
    deltay = rho * np.sin((-1)**(elevation<0) * elevation) * np.sin(azimuth)
    deltaz = -rho * np.cos(elevation)

    return deltax,deltay,deltaz

def calculate_3d_angles_ref(translations,yaws,pitchs,rolls, radars,coordinate_system="spherical"):
    """Calculate angles for each point with all four corners."""

    radars = np.hstack((radars, np.ones((radars.shape[0], 1))))

    # compute inverse transforms from radar coordinate frame vector to target coordinate frame vector
    inv_yaw = inverse_yaw_matrix(yaws)
    inv_pitch = inverse_pitch_matrix(pitchs)
    inv_roll = inverse_roll_matrix(rolls)
    inv_trans = inverse_translation_matrix(translations)

    # number of simulate points x dimension of coordinate (4) x number of radars
    relative_distances = np.matmul(inv_yaw@inv_pitch@inv_roll@inv_trans, radars.T)

    # number of simulate points  x number of radars x dimension of coordinate (4)
    relative_distances = relative_distances.transpose(0,2,1)

    if coordinate_system == "aer":
        range_, rho, azimuth, elevation = cartesian2aer(relative_distances[:,:,0],relative_distances[:,:,1],relative_distances[:,:,2])
    elif coordinate_system == "spherical":
        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:,:,0],relative_distances[:,:,1],relative_distances[:,:,2])

    # return spherical coordinates
    return range_,rho,azimuth,elevation

def plot_target_frames(ax,trans,roll,pitch,yaw,length=2):
    coordinates = np.eye(3)
    coordinates = np.hstack((coordinates,np.ones((coordinates.shape[0],1))))
    radar_positions = np.matmul(roll @ pitch @ yaw, coordinates.T).transpose(0,2,1)
    for i in range(trans.shape[0]):
        tx, ty, tz,_ = trans[i,:,-1]
        # green is the x axis
        ax.quiver(tx,ty,tz,radar_positions[i,0,0],radar_positions[i,0,1],radar_positions[i,0,2],color="g",length=length,linewidth=5)
        # magenta is the y axis
        ax.quiver(tx,ty,tz,radar_positions[i,1,0],radar_positions[i,1,1],radar_positions[i,1,2],color="m",length=length,linewidth=5)
        # yellow is the z axis
        # ax.quiver(tx,ty,tz,radar_positions[i,2,0],radar_positions[i,2,1],radar_positions[i,2,2],color="y")
        ax.quiver(tx,ty,tz,-radar_positions[i,2,0],-radar_positions[i,2,1],-radar_positions[i,2,2],color="y",length=length,linewidth=5)

def main():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Qt5Agg')

    yaws =   np.array([0,np.pi/3,0,0,-np.pi/3,0])
    pitchs = np.array([0,0,np.pi/3,0,0,-np.pi/3])
    rolls =  np.array([0,0,0,np.pi/3,0,0])
    translations = np.array([[0,0,0],
                             [0,0,0],
                             [0,0,0],
                             [0,0,0],
                             [0,0,0],
                             [5,5,4]])


    yaw = yaw_matrix(yaws)
    pitch = pitch_matrix(pitchs)
    roll = roll_matrix(rolls)
    trans = translation_matrix(translations)


    # Creating 4x2 subplots
    fig = plt.figure(figsize=(31, 16))
    n_axes = translations.shape[0]
    rad2deg = 180/np.pi

    for i in range(n_axes):
        ax = fig.add_subplot(1,n_axes,i+1, projection = '3d')  # you can adjust the size as per your requirement

        plot_target_frames(ax,trans[[i]],roll[[i]],pitch[[i]],yaw[[i]])
        title = "Roll={:.2f}, Pitch={:.2f}, Yaw={:.2f}".format(rolls[i]*rad2deg,pitchs[i]*rad2deg,yaws[i]*rad2deg)
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        ax.set_zlabel("Z")

        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        ax.set_zlim([-10,10])

        ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    main()