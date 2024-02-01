import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import ortho_group,random_correlation
from scipy.linalg import block_diag,sqrtm
import scipy.stats as ss
import random

def confidence_ellipse(ax, cov, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = 0

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = 0

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

def plot_covariance_ellipses(covs,title=""):
    d = covs.shape[-1]
    fig, ax_nstds = plt.subplots(nrows=5,ncols=5,figsize=(30, 30))
    for i,ax_nstd in enumerate(ax_nstds.ravel()):
        samples = np.random.multivariate_normal(np.zeros(d,),covs[i],size=(100,))
        ax_nstd.scatter(x=samples[:,0],y=samples[:,1],c='red')

        confidence_ellipse(ax_nstd,covs[i], n_std=1,
                           label=r'$1\sigma$', edgecolor='firebrick')
        confidence_ellipse(ax_nstd,covs[i], n_std=2,
                           label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
        confidence_ellipse(ax_nstd,covs[i],n_std=3,
                           label=r'$3\sigma$', edgecolor='blue', linestyle=':')

        # ax_nstd.autoscale_view()
        ax_nstd.set_aspect('equal', adjustable='box')

    plt.suptitle(title,fontsize=30)
    plt.tight_layout()
    plt.show()

def plot_covariance_matrices(covs,title=""):
    d = covs.shape[-1]
    fig, ax_nstds = plt.subplots(nrows=5,ncols=5,figsize=(30, 30))
    for i,ax_nstd in enumerate(ax_nstds.ravel()):
        ax_nstd.imshow(covs[i])

        ax_nstd.set_aspect('equal', adjustable='box')

    plt.suptitle(title,fontsize=30)
    plt.tight_layout()
    plt.show()

def generate_white_cov(TraceConstraint,d,N,blocks=1,method="random"):

    if np.isscalar(TraceConstraint):
        TraceConstraint = np.ones((N,))*TraceConstraint

    if np.any(TraceConstraint <= 0):
        raise Exception("You have a negative trace constraint")

    if method == "none":
        return [None] * N

    elif method == "constant":
        cov = np.tile(np.expand_dims(np.eye(d),0),(N,1,1))

    elif method == "random":
        diagonals = np.random.rand(N,d)
        cov = np.zeros((N, d, d))
        index = np.arange(d)
        cov[:, index, index] = diagonals

    elif method == "block":
        V = np.random.multivariate_normal(np.zeros(d,),cov=np.eye(d),size=(N,blocks,d))

        cov_blocks = np.matmul(V.transpose(0,1,3,2),V)

        cov = np.zeros((N,blocks*d,blocks*d))
        for i,block_matrices in enumerate(cov_blocks):
            cov[i] = block_diag(*block_matrices)


    scale = np.expand_dims(TraceConstraint / np.trace(cov,axis1=1,axis2=2),[1,2])

    cov = cov * scale

    return cov


def generate_colored_cov(TraceConstraint,d,N,method="random"):
    # use householder matrix (HOLD THE DOOR) ... HOLD DOOR ... HODOR

    if np.isscalar(TraceConstraint):
        TraceConstraint = np.ones((N,))*TraceConstraint

    if np.any(TraceConstraint <= 0):
        raise Exception("You have a negative trace constraint")

    if method == "none":
        return [None] * N

    elif method == "random":
        vs = np.random.multivariate_normal(np.zeros(d,),cov=np.eye(d),size=(N,d))

        covs = np.matmul(vs.transpose(0,2,1),vs)

    scale = np.expand_dims(TraceConstraint / np.trace(covs,axis1=1,axis2=2),[1,2])

    covs = covs * scale

    return covs

def generate_cov(TraceConstraint,d,N,blocks=1,color="white",noise_method="random",verbose=False):
    if verbose:
        print("GENERATING COVARIANCE MATRICES")

    if (blocks > 1) and (color=="white") and (noise_method=="block"):
        assert (d % blocks == 0) and (d > blocks), "The number of blocks and dimension of features are not correct!"
        d = int(d/blocks)

    if color == "white":
        covs = generate_white_cov(TraceConstraint, d, N,blocks=blocks,method=noise_method)

    elif color == "color":
        covs = generate_colored_cov(TraceConstraint, d, N,method=noise_method)

    else:
        covs = [None]*N

    return covs

def add_noise(X,SNR,cov):
    """
    @param X: The RCS numpy array. Number of samples x (Number of radars * Number of frequencies)
    @param SNR: The signal to noise ratio in dB
    @param cov: A colored covariance matrix of shape Number of frequencies x Number of frequencies
    @return: The RCS numpy array + Gaussian Noise
    """
    # set the random seed for noise generation reproducibility. Note that although the random seed is constant,
    # the passed in covariance matrix is not.
    random.seed(123)
    np.random.seed(123)

    # Get the correct dimensions
    N,d = X.shape

    if cov is None:
        return X

    # compute the required scaling factor to the covariance martrix to get the desired SNR
    # Number of samples x ,
    TraceConstraint = (X**2).sum(1) / (10 ** (SNR/10))

    # Number of samples x 1 x 1
    TraceConstraint = np.expand_dims(TraceConstraint,axis=[1,2])

    # Get the sqrt of the unit matrix such that M@M.T = C, but scale down to have unit trace
    # 1 x Number of Frequencies x Number of Frequencies
    L = np.expand_dims(sqrtm(cov/np.trace(cov)),0)

    # generate gaussian noise
    # noise is M sqrt(Trace Constraint) @ w_k
    # Number of sample x Number of Frequencies
    noise = np.random.multivariate_normal(np.zeros((d,)), cov=np.eye(d), size=(N,))
    # Number of sample x Number of Frequencies x 1
    noise = np.expand_dims(noise,axis=[2])

    # Scale the standard normal gaussian by the sqrt of the covariance matrix scaled
    # np.matmul(L,noise) is Number of samples x Number of Frequencies x 1
    # final noise is Number of samples x Number of Frequencies
    noise = (np.matmul(L,noise) * np.sqrt(TraceConstraint)).squeeze(-1)

    # add the gaussian noise to the RCS values
    X = X + noise

    return X

def add_jitter(X,width,lb,ub):
    """
    @param X: The azimuth or elevation numpy array in degrees (any shape)
    @param width: The uniform lower and upper bounds are [-width/2,width/2]
    @param lb: The lower bound to clip the azimuth or elevation (in degrees)
    @param ub: The upper bound to clipy the azimuth or elevation (in degrees)
    @return:
    """

    # set the random seed for noise generation reproducibility
    random.seed(123)
    np.random.seed(123)

    # sample uniformly at random azimuth or elevation degree jitter
    jitter = np.random.uniform(-width/2,width/2,size=X.shape)

    # clip the azimuth or elevation degree to the prespecified lower and upper bound
    X = np.clip(X + jitter,lb,ub)

    return X


# def add_noise_trajectory(X, SNR, cov):
#     N, T,d = X.shape
#
#     if cov is None:
#         return X
#
#     TraceConstraint = (X ** 2).sum(-1) / (10 ** (SNR / 10))
#     TraceConstraint = np.expand_dims(TraceConstraint, axis=[-1, -2])
#
#     # Get the sqrt of the matrix such that M@M.T = C, but scale down to have unit trace
#     L = np.expand_dims(sqrtm(cov / np.trace(cov)), [0,1])
#
#     # noise is M sqrt(Trace Constraint) @ w_k
#     noise = np.random.multivariate_normal(np.zeros((d,)), cov=np.eye(d), size=(N,T))
#     noise = np.expand_dims(noise, axis=[-1])
#     noise = (np.matmul(L, noise) * np.sqrt(TraceConstraint)).squeeze(-1)
#
#     X = X + noise
#
#     return X

def add_noise_trajectory(X,SNR,cov):
    """
    @param X: The RCS numpy array for trajectories. Number of Trajectories x Number of Time Steps x (Number of radars * Number of frequencies)
    @param SNR: The signal to noise ratio in dB
    @param cov: A colored covariance matrix of shape Number of frequencies x Number of frequencies
    @return: The RCS numpy array + Gaussian Noise
    """
    # X is Number of Trajectores x Number of Time Steps x (Number of radars * Number of frequencies)

    # get the nescessary shapes
    N,TN,d = X.shape

    if cov is None:
        return X

    # the number of unique freqencies
    f = cov.shape[-1]

    # get the number of radars
    n_radars = X.shape[-1]//cov.shape[-1]

    # # get the number of frequencies. This is sort of redundant, do not need to pass in n_radars
    # f = d // n_radars

    # set the random seed for reproducibility. Note that the covariance matrix will lead to different noises generated for MC trials.
    random.seed(123)
    np.random.seed(123)

    # Number of trajectories x Number of time steps x (Number of radars * Number of Frequencies)
    noise = np.zeros_like(X)

    # iterate through the radars
    for i in range(n_radars):
        # compute the required scaling factor to the covariance martrix to get the desired SNR
        # Number of trajectories x Number of tiem steps
        TraceConstraint = (X[:,:,i*f : (i+1)*f]**2).sum(-1) / (10 ** (SNR / 10))

        # Number of trajectories x Number of time steps x 1 x 1
        TraceConstraint = np.expand_dims(TraceConstraint, axis=[-2, -1])

        # the sqrt of the unit covariance matrix M such that M@M.T = C
        # 1 x 1 x Number of frequencies x Number of frequencies
        L = np.expand_dims(sqrtm(cov / np.trace(cov)), [0,1])


        # Sample standord normal gaussian noise
        # Number of trajectories x Number of Time steps x Number of Frequencies
        noise_temp = np.random.multivariate_normal(np.zeros((f,)), cov=np.eye(f), size=(N,TN))

        # Number of trajectories x Number of Time steps x Number of Frequencies x 1
        noise_temp = np.expand_dims(noise_temp, axis=[-1])

        # Scale the standard normal gaussian by the sqrt of the covariance matrix scaled
        # np.matmul(L,noise) is Number of samples x Number of time steps x  Number of Frequencies x 1
        noise_temp = (np.matmul(L, noise_temp) * np.sqrt(TraceConstraint)).squeeze(-1)

        # assign the noise to the correct radar
        noise[:, :, i*f : (i+1)*f] = noise_temp

    # add the noise of shape Number of trajectories x number of time samples x (Number of radars * frequencies)
    X = X + noise

    return X

def add_rice_noise(X,SNR,K=2):
    # X is Number of Trajectores x Number of Time Stpes x (Number of radars * Number of frequencies)
    # if X.ndim == 3:
    #     N,TN,d = X.shape
    # if X.ndim == 2:
    #     N,d = X.shape
    # parameters of the distribution....
    # zeta = np.sqrt(0.5 * X / (10 ** (SNR / 10)));
    #
    # A = np.sqrt(K / (K + 1) * X);
    #
    # s = np.sqrt(1 / 2 * (A ** 2) / (K) + zeta ** 2);

    random.seed(123)
    np.random.seed(123)

    A = np.sqrt((K*10**(SNR/10)*X) / ((K+1)*(10**(SNR/10)+1)));
    zeta = np.sqrt((A**2 * (K+1)) / (2*K *10**(SNR/10)));
    s = np.sqrt(1/2*(A**2)/(K) + zeta**2);

    rv = ss.ncx2(df=np.ones_like(A)*2, nc=(A ** 2) / (s ** 2), scale=s ** 2)

    # print("Noise Power: ", (2 * zeta ** 2).mean())
    # print("Signal Power: ", (A ** 2 / K + A ** 2).mean())
    # print("SNR: ",10*np.log10(((A ** 2 / K + A ** 2))/(2 * zeta ** 2)).mean())

    # noise = rv.rvs()
    # print(noise.max())
    # np.corcoeff(noise.ravel(),X.ravel())

    return rv.rvs()


def add_noise_block(X,SNR,cov,n_radars):
    N, d = X.shape
    f = d // n_radars
    if cov is None:
        return X

    noise = np.zeros((N,d))

    random.seed(123)
    np.random.seed(123)

    for i in range(n_radars):
        TraceConstraint = (X[:,i*f : (i+1)*f]**2).sum(1) / (10 ** (SNR / 10))
        TraceConstraint = np.expand_dims(TraceConstraint, axis=[1, 2])

        L = np.expand_dims(sqrtm(cov / np.trace(cov)), 0)


        noise_temp = np.random.multivariate_normal(np.zeros((f,)), cov=np.eye(f), size=(N,))
        noise_temp = np.expand_dims(noise_temp, axis=[2])
        noise_temp = (np.matmul(L, noise_temp) * np.sqrt(TraceConstraint)).squeeze(-1)

        noise[:, i*f : (i+1)*f] = noise_temp


    X = X + noise

    return X


def main():
    # test white noise generation
    TraceConstraint = 3.4
    d = 2
    # white_noise(sigma,d,N)
    N = 100
    blocks=2

    white_covs = generate_white_cov(TraceConstraint, d, N,method="random")

    for white_cov in white_covs:
        eigvals = np.linalg.eigvals(white_cov)
        assert (eigvals.sum() <= TraceConstraint+1e-8) & (eigvals.sum() >= TraceConstraint-1e-8), "The Trace(cov) != TraceConstraint!"
        assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"

    plot_covariance_matrices(white_covs,"Random White")

    if d == 2:
        plot_covariance_ellipses(white_covs,"Random White")

    white_covs = generate_white_cov(TraceConstraint, d, N,method="constant")

    for white_cov in white_covs:
        eigvals = np.linalg.eigvals(white_cov)
        assert (eigvals.sum() <= TraceConstraint+1e-8) & (eigvals.sum() >= TraceConstraint-1e-8), "The Trace(cov) != TraceConstraint!"
        assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"

    plot_covariance_matrices(white_covs,"Constant White")

    if d == 2:
        plot_covariance_ellipses(white_covs,"Constant White")


    white_covs = generate_white_cov(TraceConstraint, d, N,blocks=blocks,method="block")

    for white_cov in white_covs:
        eigvals = np.linalg.eigvals(white_cov)
        assert (eigvals.sum() <= TraceConstraint+1e-8) & (eigvals.sum() >= TraceConstraint-1e-8), "The Trace(cov) != TraceConstraint!"
        assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"


    plot_covariance_matrices(white_covs,"Block White")

    if (d == 2) and (blocks==1):
        plot_covariance_ellipses(white_covs,"Block White")

    colored_covs = generate_colored_cov(TraceConstraint, d, N,method="random")

    for colored_cov in colored_covs:
        eigvals = np.linalg.eigvals(colored_cov)

        assert (eigvals.sum() <= TraceConstraint+1e-8) & (eigvals.sum() >= TraceConstraint-1e-8), "The Trace(cov) != TraceConstraint!"
        assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"

    plot_covariance_matrices(colored_covs,"Random Colored")

    if d == 2:
        plot_covariance_ellipses(colored_covs,"Random Colored")


    # ==================== TEST ADD NOISE ======================= #
    n_freq = 15
    n_radars = 3
    N = 100
    d = n_radars*n_freq
    X = np.random.randn(N,d) + 1
    SNR = -10


    # colored_cov = generate_colored_cov(TraceConstraint=1, d=d, N=1,method="random").squeeze()
    #
    # X_noise,noise_cov = add_noise(X,SNR,colored_cov)
    # SNR_check = 10 * np.log10(X.sum(1) / np.trace(noise_cov,axis1=1,axis2=2))
    #
    # for SNR_checki,noise_covi in zip(SNR_check,noise_cov):
    #     eigvals = np.linalg.eigvals(noise_covi)
    #
    #     assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"
    #     assert (np.abs(SNR_checki-SNR) <= 1e-8) & (np.abs(SNR_checki-SNR) >= -1e-8), "The SNR IS NOT CORRECT!"
    #
    #
    # white_cov = generate_white_cov(TraceConstraint=1, d=d, N=1,method="random").squeeze()
    #
    # X_noise,noise_cov = add_noise(X,SNR,white_cov)
    # SNR_check = 10 * np.log10(X.sum(1) / np.trace(noise_cov,axis1=1,axis2=2))
    #
    # for SNR_checki,noise_covi in zip(SNR_check,noise_cov):
    #     eigvals = np.linalg.eigvals(noise_covi)
    #
    #     assert np.all(eigvals >= (-1e-9)), "The eigenvalues must all be greater than 0!"
    #     assert (np.abs(SNR_checki-SNR) <= 1e-8) & (np.abs(SNR_checki-SNR) >= -1e-8), "The SNR IS NOT CORRECT!"


if __name__ == "__main__":
    main()