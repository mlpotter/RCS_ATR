import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
from src.geometry import *

# TODO: CLEAN CODE... ERASE WHAT IS NOT NEEDED AND ADD MORE COMMENTS
def draw_3d_lines_and_points_ref(range_,rho,azimuth,elevation,translations,yaws,pitchs,rolls,radars,coordinate_system="spherical",ax=None,
                                 plotting_args={"arrow_length": 15, "arrow_linewidth": 2}):
    # Set up the plot
    # xmax,ymax,_ = radars.max(0)
    # xmin,ymin,_ = radars.min(0)
    zmax = translations[:,2].max()
    #
    #
    #
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin,ymax)
    ax.set_zlim(0,zmax)
    # ax.set_aspect('equal', adjustable='box')  # to keep the aspect ratio square

    radars = np.hstack((radars, np.ones((radars.shape[0], 1))))


    num_points,num_radars = rho.shape[:2]

    # Define the corners


    for i,radar in enumerate(radars):
        ax.plot(radar[0], radar[1], radar[2], 'yo')
        ax.text(radar[0], radar[1], radar[2],f'{i}')


    # # back from elevation of RCS DRONE DATA to SPHERICACL
    # elevation =  -(elevation - np.pi)

    # get the relative vector with respect to target coordinate frame
    # number of points x number of radars
    if coordinate_system == "aer":
        deltax, deltay, deltaz = aer2cartesian(range_, rho, azimuth, elevation)
    elif coordinate_system == "spherical":
        deltax, deltay, deltaz = spherical2cartersian(range_, rho, azimuth, elevation)

    # number of points x number of radars x coordinates
    # the relative distance between the target and radar in wrt radar coordinate frame
    delta = np.concatenate([np.expand_dims(delta, -1) for delta in [deltax, deltay, deltaz]], -1)
    delta = np.concatenate((delta,np.ones((delta.shape[0],delta.shape[1],1))),-1)

    # number of points x number of radars x coordinates
    # the yaw, pitch, roll, and translation to get the relative coordinate frame of target (wrt radar)
    yaw = yaw_matrix(yaws)
    pitch = pitch_matrix(pitchs)
    roll = roll_matrix(rolls)
    trans = translation_matrix(translations)
    # plot_target_frames(ax,trans,yaw,pitch,roll,length=plotting_args["arrow_length"],linewidth=plotting_args["arrow_linewidth"])

    for i,transi in enumerate(translations):
        ax.plot(transi[0], transi[1], transi[2], 'bo',markersize=10)
        # ax.text(transi[0], transi[1], transi[2],f'{i}',fontsize=20)

    # recover the radar position in the radar coordinate frame from the target coordinate frame information
    # and relative vector between target and radar
    for i in np.arange(num_points):
        transi,rolli,pitchi,yawi = trans[i] , roll[i] , pitch[i] , yaw[i]

        # number of radars x number of dimensions
        # sanity check to see if I recover the radar positions from perspective of target...
        radar_positions = np.matmul(transi @ yawi @ pitchi @ rolli, delta[i].T).T

        ax.plot(radar_positions[:,0], radar_positions[:,1], radar_positions[:,2], 'ro',markersize=10)


    for i,target_position in enumerate(translations):
        for j,radar_position in enumerate(radar_positions):
            ax.plot([radar_position[0], target_position[0]], [radar_position[1],  target_position[1]],[radar_position[2],  target_position[2]], 'r-',linewidth=5)  # 'r-' means red color, solid line


    # Optionally, you can add grid lines for better visualization
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add some labels (optional)
    ax.set_xlabel("X-axis",weight="bold",labelpad=15)
    ax.set_ylabel("Y-axis",weight="bold",labelpad=15)
    ax.set_zlabel("Z-axis",weight="bold",labelpad=15)

def radar_grid(n_radars,xlim,ylim):
    xs,ys = np.linspace(xlim[0],xlim[1],int(np.sqrt(n_radars))),np.linspace(ylim[0],ylim[1],int(np.sqrt(n_radars)))
    X,Y = numpy.meshgrid(xs,ys)

    return np.column_stack((X.ravel(),Y.ravel(),np.zeros((n_radars,))))

def main():
    import matplotlib as mpl
    mpl.use('Qt5Agg')

    num_points = 1
    rad2deg = 180/np.pi
    plotting_args =  {"arrow_length": 2, "arrow_linewidth": 8}
    # #

    # radars = np.array([(xlim[0], ylim[0], 0),
    #                    (xlim[0], ylim[1], 0),
    #                    (xlim[1], ylim[0], 0),
    #                    (xlim[1], xlim[1], 0)])
    # radars = np.array([[0.,10.,0],
    #                    [10.,10.,0],
    #                    [-10.,-10.,0],
    #                    [-2,-2,0],
    #                    [3,-4,0]])
    radars = radar_grid(4,[0,10],[0,10])

    # xlim =[0,20]
    # ylim=[0,20]
    # n_radars=20
    #
    # radars = np.column_stack((
    #     np.random.uniform(xlim[0], xlim[1], n_radars),
    #     np.random.uniform(ylim[0], ylim[1], n_radars),
    #     np.zeros((n_radars,))
    # ))

    xmin,ymin,_ = radars.min(0);
    xmax,ymax,_ = radars.max(0);
    zmin,zmax = [50,150]


    yaws = np.random.uniform(-np.pi/2,np.pi/2,num_points)
    pitchs = np.random.uniform(-np.pi/2,np.pi/2,num_points)
    rolls = np.random.uniform(-np.pi/2,np.pi/2,num_points)

    translations = np.column_stack((
        np.random.uniform(xmin,xmax, num_points),
        np.random.uniform(ymin, ymax, num_points),
        np.random.uniform(zmin, zmax, num_points)
    ))


    # yaws = np.ones((1,)) * np.pi/7
    # pitchs = np.ones((1,)) * np.pi/5
    # rolls = np.ones((1,)) * np.pi/6
    # translations = np.array([[5,7,5]])

    range_,rho,azimuth,elevation = calculate_3d_angles_ref(translations, yaws, pitchs, rolls, radars)

    if range_ is not None:
        print("range_: ", np.round(range_,2))
    print("rho: ", np.round(rho,2))
    print("Azimuth: ", np.round(azimuth*rad2deg,2))
    print("Elevation: ",np.round(elevation*rad2deg,2))
    print()
    # ax = plt.figure().add_subplot(projection='3d')
    # draw_3d_lines_and_points_ref(range_,rho,azimuth,elevation,translations, yaws, pitchs, rolls, radars,ax=ax,plotting_args=plotting_args)
    # ax.set_title("Random Points and Angles to Corners")
    # plt.show()

    ## sanity tests, will code better later

    # yaws_tests = [np.array([0.]),np.array([0.]),np.array([0.]),np.array([np.pi/3]),np.array([0.]),np.array([0]),np.array([np.pi/10])]
    # pitchs_tests = [np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]),np.array([np.pi]),np.array([np.pi/7])]
    # rolls_tests = [np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]),np.array([0]),np.array([0]),np.array([np.pi/5])]
    # translations_tests = [np.array([[5.0,5.,5.]]),np.array([[10.,10.,5.]]),np.array([[10.,10.,-5.]]),np.array([[5.,5.,5.]]),np.array([[0,5,5]]),np.array([[5,5,10]]),np.array([[5,5,5]])]
    # translations_tests = [x+1e-8 for x in translations_tests]

    yaws_tests = [np.array([0.]),np.array([np.pi/3]),np.array([0]),np.array([np.pi/4])]
    pitchs_tests = [np.array([0.]),np.array([0.]),np.array([np.pi]),np.array([0])]
    rolls_tests = [np.array([0]),np.array([0.]),np.array([0]),np.array([np.pi/4])]
    translations_tests = [np.array([[5.0,5.,5.]]),np.array([[0,5,5]]),np.array([[3,3,10]]),np.array([[5,5,5]])]
    translations_tests = [x+1e-8 for x in translations_tests]

    fig = plt.figure(figsize=(15,15))
    SMALL_SIZE = 40
    plt.rc('xtick', labelsize=SMALL_SIZE * 0.8)
    plt.rc('ytick', labelsize=SMALL_SIZE * 0.8)
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=SMALL_SIZE)

    for i,(yaws,pitchs,rolls,translations) in enumerate(zip(yaws_tests,pitchs_tests,rolls_tests,translations_tests)):
        inverse_yaw_matrix(yaws)
        inverse_pitch_matrix(pitchs)
        inverse_roll_matrix(rolls)
        inverse_translation_matrix(translations)

        range_,rho,azimuth,elevation = calculate_3d_angles_ref(translations, yaws, pitchs, rolls, radars)
        print("Yaw {} Pitch {} Roll {}".format(yaws*180/np.pi,pitchs*180/np.pi,rolls*180/np.pi))
        if range_ is not None:
            print("range_: ", np.round(range_,2))
        print("rho: ", np.round(rho,2))
        print("Azimuth: ", np.round(azimuth*rad2deg,2))
        print("Elevation: ",np.round(elevation*rad2deg,2))
        print()

        title = f"(yaw,pitch,roll) = {(np.round(yaws.item()*180/np.pi,2),np.round(pitchs.item()*180/np.pi,2),np.round(rolls.item()*180/np.pi,2))}\n rho: {np.round(rho,2)} \n Azimuth: {np.round(azimuth*rad2deg,2)[0]} \n Elevation {np.round(elevation*rad2deg,2)[0]}"
        ax = fig.add_subplot(2,2,i+1,projection='3d')
        draw_3d_lines_and_points_ref(range_,rho,azimuth,elevation,translations, yaws, pitchs, rolls,radars,ax=ax,plotting_args=plotting_args)
        ax.set_zlim([0,10])
        ax.set_title(title,fontsize=SMALL_SIZE*0.6,weight="bold")

    plt.tight_layout(h_pad=2,w_pad=0.3)
    plt.savefig(os.path.join("..","results","images","rollpitchyaw.pdf"),dpi=1600,bbox_inches='tight')
    plt.show()


    # ax = plt.figure().add_subplot(projection='3d')
    # Show the plot
    # plt.show()
    # ax.set_title("Random Points and Angles to Corners")



# Run the main function
if __name__ == "__main__":
    main()
