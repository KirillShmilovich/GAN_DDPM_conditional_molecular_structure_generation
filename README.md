# A Comparison of Generative Modelling Approaches for Conditional Molecular Structure Generation

### By: Mike Jones and Kirill Shmilovich

In this work we compare GANs and DDPMs for the task of conditional molecular structure generation.

The code the GAN and DDPM models are available in the associated `GAN` and `DDPM` directories.

CONTRIBUTIONS:

-- Kirill: GAN

-- Mike: DPPM

## Comparisson of generated molecular trajectories

left=REAL, center=GAN, right=DDPM

94-atom WLALL pentapeptide, conditioning on 4-dim TICA embedding using all heavt atom backbone distances
![ ](imgs/pep_aa_compare.gif)

20-atom WLALL pentapeptide backbone, conditioning on 4-dim TICA embedding using all pairwise backbone distances
![ ](imgs/pep_bb_compare.gif)

8-atom alanine dipeptide, conditioning on 2-dim backbone $\phi$, $\psi$ angles:
![adp_compare-2022-12-10_16 00 34](https://user-images.githubusercontent.com/40403472/206876984-b55f8022-8a6b-4ef5-8ed3-f7d8151a9ca5.gif)

## GAN training progress

Structures generated at intermediate training epochs for a fixed conditioning variable input:
![ ](imgs/gan_progress.gif)

## DDPM diffusion process

Atom positions throughout the 1000-step diffusion process arriving at the final predicted structure.
![ ](imgs/diffusion_progress.gif)

## Generated structures under different random noises
Using a fixed conditioning defined by the reference frame (transparent)

GAN
![ ](imgs/vary_noise_frame-0_gan.gif)

DDPM
![ ](imgs/vary_noise_frame-0_ddpm.gif)

## Comparisson of Conditioning Free Energy Surfaces
For the test set we compare the free energy surfraces in the conditioning variables. These free energy surfances are quantities proportional to the density of states, where we should expect our reconstructed molecular structures to adhere to the real conditioning variables used as input to the models at inference time. 

![image](https://user-images.githubusercontent.com/40403472/206928265-fbf71c12-dd33-4dd8-9c88-112e4da3f4c6.png)
