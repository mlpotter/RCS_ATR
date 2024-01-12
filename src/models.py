
import numpy as np


class distributed_classifier(object):
    def __init__(self,clf,use_geometry=False):
        self.clf = clf
        self.use_geometry = use_geometry
    def predict(self,dataset,fusion_method="average"):
        n_radars = dataset["n_radars"]
        n_freq = dataset["n_freq"]

        RCS = dataset["RCS"]
        azimuth = dataset["azimuth"]
        elevation = dataset["elevation"]
        n_classes = len(np.unique(dataset["ys"]))

        predictions = np.zeros((RCS.shape[0],n_classes,n_radars))

        for i in range(n_radars):

            X = RCS[:,(i*n_freq):((i+1)*n_freq)]
            if self.use_geometry:
                azimuth_i = azimuth[:, [i]]
                elevation_i = elevation[:, [i]]
                X = np.hstack((X,azimuth_i,elevation_i))

            y_pred = self.clf.predict_proba(X)

            predictions[:,:,i] = y_pred

        if fusion_method == "average":
            predictions = predictions.mean(-1).argmax(1,keepdims=True)

        elif fusion_method == "max":
            predictions = predictions.max(-1).argmax(1,keepdims=True)

        elif fusion_method == "fusion":
            predictions = np.log(predictions+1e-16).sum(-1).argmax(1,keepdims=True)
        else:
            raise Exception("Not a valid fusion method of multiple radars")

        return predictions

class distributed_recursive_classifier(object):
    def __init__(self,clf,dataset=None,use_geometry=False):
        self.clf = clf
        self.use_geometry = use_geometry

        if dataset is None:
            self.pc = None
        else:
            values,counts = np.unique(dataset["ys"].ravel(),return_counts=True)
            self.pc = counts/np.sum(counts)

        self.n_classes = len(self.pc)
        self.eps = 0.4
        self.A = np.ones((self.n_classes,self.n_classes))* self.eps#np.eye(self.n_classes)
        np.fill_diagonal(self.A,1-(self.n_classes-1)*self.eps)
    def predict_instant(self,dataset,t=0,fusion_method="average"):
        n_radars = dataset["n_radars"]
        n_freq = dataset["n_freq"]

        RCS = dataset["RCS"][:,t,:]
        azimuth = dataset["azimuth"][:,t,:]
        elevation = dataset["elevation"][:,t,:]
        n_classes = len(np.unique(dataset["ys"]))

        predictions = np.zeros((RCS.shape[0],n_classes,n_radars))

        for i in range(n_radars):

            X = RCS[:,(i*n_freq):((i+1)*n_freq)]
            if self.use_geometry:
                azimuth_i = azimuth[:, [i]]
                elevation_i = elevation[:, [i]]
                X = np.hstack((X,azimuth_i,elevation_i))

            y_pred = self.clf.predict_proba(X)

            predictions[:,:,i] = y_pred

        if fusion_method == "average":
            predictions = predictions.mean(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "max":
            predictions = predictions.max(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "fusion":
            predictions = np.log(predictions+1e-16).sum(-1)#.argmax(1,keepdims=True)
        else:
            raise Exception("Not a valid fusion method of multiple radars")

        return predictions

    def predict(self,dataset,fusion_method="fusion"):

        N_traj, N_time, d = dataset["RCS"].shape

        # initialize p_cprev_given_zpast as prior for every trajectory
        p_cprev_given_zpast = np.tile(self.pc,(dataset["RCS"].shape[0],1))



        for t in np.arange(N_time):
            # Initialize running recursive probability holder matrix
            p_ccurr_given_zall = np.zeros_like(p_ccurr_given_zcurr)

            # Number of Trajectories x Number of Classes
            p_ccurr_given_zcurr = self.predict_instant(dataset,t=t,fusion_method="average")

            for c in np.arange(self.n_classes):

                # calculat the whole summation sign
                summation_term = 0
                for c_prev in np.arange(self.n_classes):
                    p_ccurr_given_cprev_num = self.A[c_prev,c]

                    # calculate the denominator of the summation
                    summation_term_den = 0
                    for c_curr in np.arange(self.n_classes):

                        p_ccurr_given_cprev_den = self.A[c_prev,c_curr]

                        # Number of Trajectores
                        summation_term_den += (p_ccurr_given_zcurr[:,c_curr]/self.pc[c_curr] * p_ccurr_given_cprev_den)

                    summation_term += p_cprev_given_zpast[:,c_prev]*(p_ccurr_given_cprev_num/summation_term_den)

                p_ccurr_given_zall[:,c] = p_ccurr_given_zcurr[:,c] / self.pc[c] * summation_term

        return p_ccurr_given_zall




def main():
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.metrics import classification_report

    from src.noise_generator import add_noise, add_noise_block,add_noise_trajectory
    from src.data_loader import DRONE_RCS_CSV_TO_XARRAY, RCS_TO_DATASET, RCS_TO_DATASET_Single_Point, dataset_to_tensor, \
        dataset_train_test_split

    from src.trajectory_loader import RCS_TO_DATASET_Trajectory
    from src.noise_generator import generate_cov



    from lazypredict.Supervised import LazyClassifier, CLASSIFIERS

    import random


    CLASSIFIERS = dict(CLASSIFIERS)

    np.random.seed(123)
    random.seed(123)

    classifiers_names = ["XGBClassifier","KNeighborsClassifier","LogisticRegression"]

    classifiers = [CLASSIFIERS[name] for name in classifiers_names]  # + [MLPClassifier]

    DRONE_RCS_FOLDER = "..\Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER)
    drone_names = list(drone_rcs_dictionary.keys())
    n_freq = len(drone_rcs_dictionary[drone_names[0]].coords["f[GHz]"])

    xlim = [0, 20];
    ylim = [0, 20];
    zlim = [50, 200];
    bounding_box = np.array([xlim, ylim, zlim])
    yaw_lim = [-np.pi / 5, np.pi / 5];
    pitch_lim = [-np.pi / 5, np.pi / 5]
    roll_lim = [-np.pi / 5, np.pi / 5]

    n_radars = 20

    radars = np.column_stack((
        np.random.uniform(xlim[0], xlim[1],n_radars),
        np.random.uniform(ylim[0], ylim[1], n_radars),
        np.zeros((n_radars,))
    ))

    use_geometry = False

    noise_color="color"
    noise_method="random"
    SNR_constraint = 1000

    covs_single = generate_cov(TraceConstraint=1, d=n_freq, N=1,
                               blocks=n_radars, color=noise_color,
                               noise_method=noise_method)

    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=10000,method="random")

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])

    dataset_train, dataset_test = dataset_train_test_split(dataset_single)



    X_train, y_train = dataset_to_tensor(dataset_train, use_geometry)
    X_test, y_test = dataset_to_tensor(dataset_test, use_geometry)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, classifiers=classifiers)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(models.to_string())

    dataset_multi = RCS_TO_DATASET(drone_rcs_dictionary, radars, yaw_lim, pitch_lim, roll_lim, bounding_box,
                                   num_points=100)

    dataset_multi["RCS"] = add_noise_block(dataset_multi["RCS"], SNR_constraint, covs_single[0],
                                           n_radars)


    for classifiers_name in classifiers_names:

        distributed_clf = distributed_classifier(clf.models[classifiers_name],use_geometry)

        print(f"Distributed {classifiers_name} CLASSIFICATION")
        y_pred_avg = distributed_clf.predict(dataset_multi,fusion_method="average")
        print(classification_report(dataset_multi["ys"],y_pred_avg))





    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=10000,method="random")

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])

    dataset_train, dataset_test = dataset_train_test_split(dataset_single)

    #================= TEST DISTRIBUTED RADAR CLASSIFIER =================#
    TN = 50
    N_traj = 300
    time_step_size = 0.1
    vx = 50
    yaw_range = pitch_range = roll_range = np.pi/15
    xlim = [-50, 50];  ylim = [-50, 50]; zlim = [50, 150]
    bounding_box = np.array([xlim,ylim,zlim])

    dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                              time_step_size=time_step_size, vx=vx,
                                              yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,
                                              bounding_box=bounding_box,
                                              TN=TN, radars=radars,
                                              num_points=N_traj,
                                              verbose=False)

    dataset_multi["RCS"] = add_noise_trajectory(dataset_multi["RCS"],SNR_constraint,covs_single[0],n_radars)

    dataset_train, dataset_test = dataset_train_test_split(dataset_multi)
    drc =  distributed_recursive_classifier(clf.models[classifiers_name],dataset_train,use_geometry=False)
    drc.predict(dataset_test)


if __name__ == "__main__":
    main()




