# A Comparison of Generative Modelling Approaches for Conditional Molecular Structure GenerationLearning

## By: Mike Jones and Kirill Shmilovich

In this work we compare GANs and DDPMs for the task of conditional molecular structure generation.

### Comparisson of generated molecular trajectories

left=REAL, center=GAN, right=DDPM

94-atom WLALL pentapeptide, conditioning on 4-dim TICA embedding using all heavt atom backbone distances
![ ](imgs/pep_aa_compare.gif)

20-atom WLALL pentapeptide backbone, conditioning on 4-dim TICA embedding using all pairwise backbone distances
![ ](imgs/pep_bb_compare.gif)

8-atom alanine dipeptide, conditioning on 2-dim backbone $\phi$, $\psi$ angles:
![adp_compare-2022-12-10_16 00 34](https://user-images.githubusercontent.com/40403472/206876984-b55f8022-8a6b-4ef5-8ed3-f7d8151a9ca5.gif)
