# GRACE_Tiles
GRACE gravity field inversion package

GRACE_Tiles is a library of routines that can be used to create regularized, gridded, 
global fields of surface mass variability from unconstrained GRACE spherical harmonic 
coefficients.  These scripts will take GRACE Stokes coefficients and process them in 
a manner similar to 'mascon' approaches, allowing a user to regularize the gravity 
field solutions using constraints that are defined in the spatial domain (as opposed 
to in the spectral domain).  Users can easily change the regularization parameters and 
tile shapes.
