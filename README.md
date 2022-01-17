# UNet_XraySPI
Companion repository for Bellisario A., Maia F. R. N. C. , and Ekeberg T. (2021). "Noise reduction and mask removal neural network for X-ray single particle imaging".  Journal of applied crystallography.

Our work examined deep learning as a tool to denoise and demask diffraction patterns. The goal is to map Poisson sampled and masked 2D diffraction intensities into continuous Fourier amplitudes. We trained a convolutional neural network on 9900 diffraction patterns, each simulated from an independent PDB file, and studied the model's performance under different signal strengths and mask sizes. 

### Data & Network weights

The complete list of PDB IDs used for simulations can be found in ~/data/dataset_training.csv and  ~/data/dataset_test.csv.

In ~/data/testing/ we uploaded twenty patterns simulated at different signal strengths and mask size, as described in the paper. In particular:

1. ~/data/testing/groundtruth/ contains the file 'groundtruth.h5' with noiseless and maskless simulations
2. ~/data/testing/masked/ contains files 'masked_Xpx.h5' where X is the mask width in number of pixels
3. ~/data/testing/noisy/ contains files 'noisy_X.h5' where X is the intensity factor used during Poisson sampling
4. ~/data/testing/noisy&masked/ contains files 'noisy_X_Ypx.h5' where X is the intensity factor used during Poisson sampling and Y is the mask width in number of pixels
5. ~/data/testing/metadata.h5 reports PDB IDs and orientation of the structure used during the diffraction intensities simulations

To avoid surpassing the Git LFS data quota, weights can be found at this external link: https://bit.ly/3qy3Q6Y
Files are organized in directories according to the training and testing conditions, reporting the intensity factor and the mask size.
For any problem please contact alfredo.bellisario@icm.uu.se


#### Python packages recommendations:

We suggest to use Tensorflow 2.0.0
