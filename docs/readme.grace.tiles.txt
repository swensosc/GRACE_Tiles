---------  GRACE Tile Inversion Scripts README  -----------------

The grace.tiles.py python script, which uses a library of routines 
called lib_tile.py, creates regularized, gridded, global fields of 
surface mass variability from the unconstrained GRACE spherical 
harmonic coefficients and the corresponding covariance matrices.   

A number of steps are required convert the spherical harmonic data 
into gridded, global surface mass fields:

1) set up coordinate system and create land/ocean mask
The default coordinate system is a 1 degree by 1 degree 
latitude/longitude grid, but other resolutions can be specified.
A land/ocean mask is used to discriminate between land and ocean
tiles.  This mask can be further modified by the user if desired.

2) define tiles, using separate land/ocean tile sizes
Tiles in this context are geographical regions for which 
spatially averaged surface mass variations will be 
calculated.  By default, tiles are equal angular (rectangular 
in the coordinate domain), but can be specified separately 
for land and ocean.  The tile sizes are set with the variables 
'land_tile_size' and 'ocean_tile_size'.

3) read in GRACE spherical harmonic coefficients
The unconstrained GRACE spherical harmonic coefficients are read 
in, then modified by adding a degree one estimate, replacing 
the C20 estimate with the SLR-based estimate provided by CSR, 
and removing an estimate of the mass changes due to glacial 
isostatic adjustment.

4) create a tile transfer matrix, which is the matrix that 
transforms spherical harmonic coefficients into the tile 
representation.  If the transfer matrix is multiplied by a 
spherical harmonic set, it will result in the tile-averages 
of the unconstrained data.  In other words, it will be equivalent 
to converting the spherical harmonic coefficients to the coordinate 
domain and calculating the spatial average over each tile explicitly.  
This results in a noisy solution.  However, by regularizing (i.e. 
applying constraints) this solution in a least squares framework, 
the effects of noise can be reduced; this is the purpose of 
this script.

5) read in model data (used to constrain tile amplitude)
One way of regularizing a least squares inversion is to damp the 
solution, which means that solution parameters that diverge 
greatly from the a priori solution are penalized.  In this context, 
a model can be used to estimate the relative amplitude of 
tiles as a function of location.  Thus a tile over a desert 
region (in which one might expect only small amplitude surface 
mass anomalies) might be penalized more greatly than a tile 
in a region known to have large amplitude mass variations.  
By default, the expected mass variations could come from the 
GRACE data themselves, after being suitably filtered.  Alternatively, 
one could use a geophysical model, such as an Earth System model, 
to estimate the global mass variations.  There are two parameters 
that the user can set to further adjust the model estimate: 
'minimum_rms', which provides a lower bound on the estimated surface 
mass variations, and 'rms_scale_factor', which can be used to 
uniformly scale the mass estimate.

6) create damping matrix 
In addition to simply damping the solution, one can define higher 
order metrics of solution good-behavior.  Spatial correlations can 
be used to regularize the least squares inversion by penalizing 
solutions for tiles that vary independently of their neighbors.  
By default, the script applies a matrix that calculates the gaussian 
weighted average of the surface mass variations surrounding a 
particular tile.  The gaussian average is controlled by specifying 
its half-width in degrees.  The half-width can be specified 
independently for land and ocean using the parameters 'hw_land' and 
'hw_ocean'.  The land/ocean mask is further used to reduce correlations 
between land and ocean tiles.  

7) calculate GRACE-derived covariance matrices
To damp the original, unconstrained spherical harmonic coefficients, 
the original covariance matrix is required.  (See Swenson and Wahr, 
'Estimating signal loss in regularized GRACE gravity field solutions', 
Geophysical Journal International, 2011 for a brief summary of 
how this methodology is applied in the spherical harmonic domain.)  
The scripts reads in the covariance matrix and transforms it to the 
tile domain, which enables the regularization to be defined in the 
tile domain, as described above.

8) convert Stokes coefficients to tiles in mass units  
Finally, surface mass variations in the tile domain are converted 
to coordinate space (lat/lon).  The GRACE covariance matrix does not 
include information on degree one variations, so these variations 
are added back to the solution at this stage.   
