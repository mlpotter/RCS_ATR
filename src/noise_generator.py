import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import ortho_group,random_correlation
from scipy.linalg import block_diag,sqrtm

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

    N,d = X.shape

    if cov is None:
        return X

    TraceConstraint = (X**2).sum(1) / (10 ** (SNR/10))
    TraceConstraint = np.expand_dims(TraceConstraint,axis=[1,2])

    # Get the sqrt of the matrix such that M@M.T = C, but scale down to have unit trace
    L = np.expand_dims(sqrtm(cov/np.trace(cov)),0)

    # noise is M sqrt(Trace Constraint) @ w_k
    noise = np.random.multivariate_normal(np.zeros((d,)), cov=np.eye(d), size=(N,))
    noise = np.expand_dims(noise,axis=[2])
    noise = (np.matmul(L,noise) * np.sqrt(TraceConstraint)).squeeze(-1)

    X = X + noise

    return X

def add_jitter(X,width,lb,ub):


    N,d = X.shape

    jitter = np.random.uniform(-width/2,width/2,size=(N,1))


    X = np.clip(X + jitter,lb,ub)

    return X


def add_noise_trajectory(X, SNR, cov):
    N, T,d = X.shape

    if cov is None:
        return X

    TraceConstraint = (X ** 2).sum(-1) / (10 ** (SNR / 10))
    TraceConstraint = np.expand_dims(TraceConstraint, axis=[-1, -2])

    # Get the sqrt of the matrix such that M@M.T = C, but scale down to have unit trace
    L = np.expand_dims(sqrtm(cov / np.trace(cov)), [0,1])

    # noise is M sqrt(Trace Constraint) @ w_k
    noise = np.random.multivariate_normal(np.zeros((d,)), cov=np.eye(d), size=(N,T))
    noise = np.expand_dims(noise, axis=[-1])
    noise = (np.matmul(L, noise) * np.sqrt(TraceConstraint)).squeeze(-1)

    X = X + noise

    return X

def add_noise_block(X,SNR,cov,n_radars):
    N, d = X.shape
    f = d // n_radars
    if cov is None:
        return X

    noise = np.zeros((N,d))

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