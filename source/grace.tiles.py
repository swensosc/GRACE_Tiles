#! /usr/bin/env python
import sys
import os
import string
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as netcdf4

from scipy import linalg

# Set base directory environment variable
dir_root = os.path.dirname(os.path.realpath(__file__))
os.environ['TILE_ROOT'] = dir_root
env_check=os.getenv('TILE_ROOT', 'Not Set')
print 'dir_root: ',env_check

# append path before importing libraries
#libdir = dir_root + '/lib'
#sys.path.append(libdir)

sys.path.append(dir_root)

import lib_tile as tlib

# manually reload module to account for changes during development
reload(tlib)

''' -------------------   Begin Instructions  -----------------
These python scripts use the original, unconstrained GRACE gravity
spherical harmonic coefficients to construct a global, gridded 
dataset of surface mass change.  A high resolution lat/lon 
coordinate grid is used to define a set of coarser resolution 
tiles, for which average surface mass change values are estimated.
To reduce the effects of errors in the GRACE data, constraints are 
placed on the solution via a least squares inversion.  Two basic 
constraints are used: 1) damping the amplitude of the variability 
of each tile toward an a priori estimate and 2) damping the value 
of each tile toward the spatial average of its surroundings.  

The script is divided into sections, and the output of each 
section can be saved to speed up the script by allowing 
some intermediate data to be read in rather than calculated 
each time.  

The steps in the script are as follows:
1) set up coordinate system and create land mask
2) define tiles, using separate land/ocean tile sizes
3) read in GRACE spherical harmonic coefficients
4) create tile transfer matrix, that relates tiles to 
   spherical harmonic representation
5) read in model data (used to constrain tile amplitude)
6) create damping matrix 
7) calculate GRACE-derived covariance matrices
8) convert Stokes coefficients to tiles in mass units  

The main parameters are the tiles sizes and the correlation 
lengths [degrees], which are set below (land_tile_size, 
ocean_tile_size, hw_land, hw_ocean).  The a priori 
variance estimate can also be adjusted using the (minimum_rms, 
rms_scale_factor) parameters.

-------------------   End Instructions  ----------------------- '''

#@0
#-----------------------------------------------------
#  set control parameters
#-----------------------------------------------------

land_tile_size   = 3    #[degrees]
ocean_tile_size  = 8    #[degrees]

hw_land          = 4.0  #[degrees]
hw_ocean         = 12.0 #[degrees]

minimum_rms      = 30.0 #[mm]
rms_scale_factor = 1.5  #[ ]

data_dir = './'
data_dir = '/project/tss/swensosc/grace_data/covmat/tiles/'
ancillary_data_dir= '/project/tss/swensosc/grace_data/covmat/blocks/tar_dir/ancillary_data/'

# read/write flags, if False then perform calculation
# step 4
transfer_matrix_directory = data_dir + 'tmat_tile_dir/'
read_tile_transfer_matrix  = True

# step 5
model_covariance_matrix_directory = data_dir + 'model_covmat_dir/'
read_model_covariance_matrix  = True

# step 6
alpha_prime_matrix_directory = data_dir + 'alpha_dir/'
read_alpha_prime_matrix  = True

# step 7
grace_covariance_matrix_directory = data_dir + 'grace_covmat_dir/'
read_grace_covariance_matrix  = True

#--  choose covariance matrix to use
covnum = 1
cov_dir = data_dir + 'cov_dir/'
if covnum == 1:
    covlab='rl05'
    covmatfile='RL05_July_2013_cov'
    covmatfile='/project/tss/swensosc/grace_data/covmat/blocks/tar_dir/cov_dir/RL05_July_2013_cov'

# step 8
tile_tws_directory = data_dir + 'tws_dir/'
write_tile_tws  = True

# other required data files
tellus_dir = data_dir + 'tellus_dir/'
tellus_dir = '/project/tss/swensosc/grace_data/tellus/web/'
tellus_grace_file = tellus_dir + 'GRCTellus.CSR.200204_201509.LND.RL05.DSTvSCS1409.nc'
tellus_scaling_file = tellus_dir + 'CLM4.SCALE_FACTOR.DS.G300KM.RL05.DSTvSCS1401.nc'

topofile = ancillary_data_dir + 'DEM.nc'
#land_mask_file = ancillary_data_dir + 'land_mask.nc'
land_mask_file = data_dir + 'land_mask.nc'

grace_data_dir='csr/rl05/'
grace_data_dir='/project/tss/swensosc/grace_data/csr/rl05/'
geocenter_file=ancillary_data_dir + 'gad_gsm.rl05.txt'
c20_file=ancillary_data_dir + 'TN-07_C20_SLR.txt'
gia_file=ancillary_data_dir + 'pgr_stokes_geruo.a.txt'

lovenumfile=ancillary_data_dir + 'loadlove'

#@1
#-----------------------------------------------------
#  set up coordinate system and create land mask
#-----------------------------------------------------
dlatlon = 1.0
bim     = np.int32(360/dlatlon)
bjm     = np.int32(180/dlatlon)
blon    = -180.0+(np.arange(bim)+0.5)*dlatlon
blat    = -90.0+(np.arange(bjm)+0.5)*dlatlon

phi  = (np.pi/180.)*blon
th   = (np.pi/180.)*(90.0 - blat)
dphi = np.abs(phi[0]-phi[1])
dth  = np.abs(th[0]-th[1])

#for i in range(bim):
#    print i, bim, blon[i]
#for j in range(bjm):
#    print j, bjm, blat[j]

lmask = tlib.read_land_mask(blon,blat,land_mask_file, neglon=True) 
omask = 1.0 - lmask

#--  save initial mask, then remove isolated groups of points
omask_initial = np.copy(omask)
omask         = tlib.mask_remove_points(omask)

#--  convert red sea and caspian sea to land points
convert_red_and_caspian_seas_to_land = 1
if convert_red_and_caspian_seas_to_land == 1:
    ind=np.where(np.bitwise_and((blon >= 44), (blon <= 56)))[0]
    ind2=np.where(np.bitwise_and((blat >= 37),(blat <= 52)))[0]
    for i in range(ind.size):
        omask[ind2,ind[i]]=0
    ind=np.where(np.bitwise_and((blon >= 33), (blon <= 43)))[0]
    ind2=np.where(np.bitwise_and((blat >= 14),(blat <= 29)))[0]
    for i in range(ind.size):
        omask[ind2,ind[i]]=0
    # update lmask to be consistent with omask
    lmask = 1.0 - omask

print 'step @1 complete'

if 1 == 2: # plot for debugging purposes
    x0=0.1
    w0=0.95-x0
    y0=0.1
    h0=0.95-y0
    
    fig1=plt.figure()
    ax1=fig1.add_axes([x0,y0,w0,h0])
    lnum=10
    ilevel=np.arange(lnum,dtype=float)/np.float(lnum-1)
    level=2.*ilevel
    con=plt.contourf(blon,blat,lmask,levels=level)
    #con=plt.contourf(blon,blat,omask,levels=level)
    #con=plt.contourf(blon,blat,elev,levels=level)
    #con=plt.contourf(blon,blat,tmp,levels=level)
    plt.ylabel('Lat')
    plt.xlabel('Lon')
    plt.title('Data')
    plt.show()
    stop
print '\nstep @1 complete: coordinates and masks defined\n'

#@2
#------------------------------------------------------
#  define tiles, using separate land/ocean tile sizes
#  the tile_mask will give a different identification
#  number to each tile
#------------------------------------------------------

print 'land/ocean tile sizes: ', land_tile_size, ocean_tile_size, '\n'

#--  divide land into tiles  ----------------------
land_tile_mask = tlib.tile_mask(blon,blat,land_tile_size)
land_tile_mask = lmask * land_tile_mask

if 1 == 2: # plot for debugging purposes
    x0=0.1
    w0=0.95-x0
    y0=0.1
    h0=0.95-y0
    
    fig1=plt.figure()
    ax1=fig1.add_axes([x0,y0,w0,h0])
    lnum=10
    ilevel=np.arange(lnum,dtype=float)/np.float(lnum-1)
    #level=5e2*ilevel
    level=8e3*ilevel
    level -= np.max(level)
    print np.min(land_tile_mask), np.max(land_tile_mask)
    con=plt.contourf(blon,blat,land_tile_mask,levels=level)
    plt.ylabel('Lat')
    plt.xlabel('Lon')
    plt.title('Data')
    plt.show()


land_tile_indices = np.asarray(np.unique(land_tile_mask[land_tile_mask < 0]),
                                dtype=np.int32)

#--  divide ocean into tiles  ----------------------
ocean_tile_mask = tlib.tile_mask(blon,blat,ocean_tile_size)
ocean_tile_mask += np.min(land_tile_mask)
ocean_tile_mask = omask * ocean_tile_mask

ocean_tile_indices = np.asarray(np.unique(ocean_tile_mask[ocean_tile_mask < 0]), 
                                 dtype=np.int32)

full_tile_mask   = ocean_tile_mask + land_tile_mask
full_tile_indices = np.append(land_tile_indices, ocean_tile_indices)

print 'number of land tiles / index range'
print land_tile_indices.size, np.min(land_tile_indices),np.max(land_tile_indices)
print 'number of ocean tiles / index range'
print ocean_tile_indices.size, np.min(ocean_tile_indices),np.max(ocean_tile_indices)
print 'total number of tiles / index range'
print full_tile_indices.size, np.min(full_tile_indices),np.max(full_tile_indices)

if 1 == 2: # plot for debugging purposes
    x0=0.1
    w0=0.95-x0
    y0=0.1
    h0=0.95-y0
    
    fig1=plt.figure()
    ax1=fig1.add_axes([x0,y0,w0,h0])

    plt.plot(land_tile_indices)
    plt.ylabel('indices')
    plt.xlabel('number')
    plt.title('land/ocean/full')
    plt.show()

if 1 == 2: # plot for debugging purposes
    x0=0.1
    w0=0.95-x0
    y0=0.1
    h0=0.95-y0
    
    fig1=plt.figure()
    ax1=fig1.add_axes([x0,y0,w0,h0])
    lnum=10
    ilevel=-1.0*np.flipud(np.arange(lnum,dtype=float)/np.float(lnum-1))
    level=8.5e3*ilevel
    level=1e3*ilevel - 7500.
    print np.min(land_tile_mask), np.max(land_tile_mask)
    print np.min(ocean_tile_mask), np.max(ocean_tile_mask)
    print np.min(full_tile_mask), np.max(full_tile_mask)
    #con=plt.contourf(blon,blat,full_tile_mask,levels=level)
    con=plt.contourf(blon,blat,ocean_tile_mask,levels=level)
    plt.ylabel('Lat')
    plt.xlabel('Lon')
    plt.title('Data')
    plt.show()

print '\nstep @2 complete: tiles defined\n'

#@3
#-----------------------------------------------------
#  read in GRACE spherical harmonic coefficients
#-----------------------------------------------------

x=tlib.read_grace_data_rl05(grace_data_dir=grace_data_dir,
                            geocenter_file=geocenter_file,
                            c20_file=c20_file, gia_file=gia_file)

gtime=x['time']
clm=x['clm']
slm=x['slm']
tm=gtime.size

lm=clm.shape[0] - 1
tm=gtime.size

#--  save degree one coefficients for later use
geoclm=clm[0:1,0:1,:]
geoslm=slm[0:1,0:1,:]

#--  remove temporal mean over period of data record
meanclm=np.sum(clm,2)/np.float64(tm)
meanslm=np.sum(slm,2)/np.float64(tm)
for t in range(tm):
   clm[:,:,t]-=meanclm
   slm[:,:,t]-=meanslm

print '\nstep @3 complete: spherical harmonic coefficients read in\n'

#@4
#-----------------------------------------------------
#  create tile transfer matrix (tmat_tile)
#  this is the matrix whose vectors are the SH coefficients 
#  describing each tile
#  tmat_tile corresponds to eqn 14 in the derivation document
#-----------------------------------------------------
ltotc = (np.power(lm,2)+3*lm-4)/2           #for l=2,lm# m=0,l (i.e. CLM)
ltots = (np.power(lm,2)+lm-2)/2             #for l=2,lm# m=1,l (i.e. SLM)
ltot  = ltotc+ltots

itot=full_tile_indices.size
print 'itot, ltot: ',itot,ltot
tmat_tile=np.zeros((ltot,itot),dtype=np.float64)
gclm=np.zeros((lm+1,lm+1),dtype=np.float64)
gslm=np.zeros((lm+1,lm+1),dtype=np.float64)
 
if read_tile_transfer_matrix:

    infile = transfer_matrix_directory+'btm_lnd.'+str(land_tile_size) \
             +'_ocn.'+str(ocean_tile_size)+'.nc'

    print 'reading tile transfer matrix from: ', infile

    f1 =  netcdf4.Dataset(infile, 'r', format='NETCDF4')
    tmat_tile  = np.copy(f1.variables['tile_transfer_matrix'][:,:])
    f1.close

else:
    print 'creating tile transfer matrix'

    for i0 in range(itot):
        print i0,itot
        ind=np.where(full_tile_mask == full_tile_indices[i0])[0]
        if ind.size <= 0:
            print 'index: ', i0
            sys.exit('error, no points in full_mask for this index')

        mask=np.float64(np.where(full_tile_mask == full_tile_indices[i0],1,0))
        tlib.gen_mask_sh_coefficients(mask,phi,th,gclm,gslm)
        area=gclm[0,0]

        gclm=gclm/area
        gslm=gslm/area

        tmat_tile[:,i0]=tlib.vectorize_sh_coefs(gclm,gslm)

        if 1 == 2: # plot for debugging purposes
            x0=0.1
            w0=0.95-x0
            y0=0.1
            h0=0.95-y0
            
            fig1=plt.figure()
            ax1=fig1.add_axes([x0,y0,w0,h0])
            lnum=10
            ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
            if 1 == 2:
                level=np.max(gclm)*ilevel
                con=plt.contourf(np.transpose(gclm),levels=level)
            else:
                level=np.max(gslm)*ilevel
                con=plt.contourf(np.transpose(gslm),levels=level)
            plt.ylabel('l')
            plt.xlabel('m')
            plt.title('clm/slm')
            plt.show()

    if not read_tile_transfer_matrix:
        outfile=transfer_matrix_directory+'btm_lnd.'+str(land_tile_size) \
                 +'_ocn.'+str(ocean_tile_size)+'.nc'

        w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
        command='date "+%y%m%d"'
        x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
        x=x2.communicate()
        timetag = x[0].strip()
        w.creation_date = timetag 
        w.title = 'Tile Transfer Matrix'
        w.createDimension('ltot',np.int32(ltot))
        w.createDimension('itot',np.int32(itot))
        w.createDimension('lon',np.int32(bim))
        w.createDimension('lat',np.int32(bjm))
        w.createDimension('lbi',np.int32(land_tile_indices.size))
        w.createDimension('obi',np.int32(ocean_tile_indices.size))
        w.createDimension('fbi',np.int32(full_tile_indices.size))
    
        wtmat_tile = w.createVariable('tile_transfer_matrix','f8',('ltot','itot'))
        wlbsize = w.createVariable('land_tile_size','i4')
        wobsize = w.createVariable('ocean_tile_size','i4')
        wlon = w.createVariable('lon','f8',('lon'))
        wlat = w.createVariable('lat','f8',('lat'))
        wlmask = w.createVariable('land_mask', 'f8',('lat','lon'))
        womask = w.createVariable('ocean_mask', 'f8',('lat','lon'))
        wlbm = w.createVariable('land_tile_mask', 'f8',('lat','lon'))
        wobm = w.createVariable('ocean_tile_mask','f8',('lat','lon'))
        wfbm = w.createVariable('full_tile_mask', 'f8',('lat','lon'))
        wlbi = w.createVariable('land_tile_indices', 'f8',('lbi'))
        wobi = w.createVariable('ocean_tile_indices','f8',('obi'))
        wfbi = w.createVariable('full_tile_indices', 'f8',('fbi'))

        # write to file  --------------------------------------------
        wtmat_tile[:,:] = tmat_tile
        wlbsize[:]   = land_tile_size
        wobsize[:]   = ocean_tile_size
        wlon[:]   = blon
        wlat[:]   = blat
        wlbm[:,:] = land_tile_mask
        wobm[:,:] = ocean_tile_mask
        wfbm[:,:] = full_tile_mask
        wlmask[:,:] = lmask
        womask[:,:] = omask
        wlbi[:]   = land_tile_indices
        wobi[:]   = ocean_tile_indices
        wfbi[:]   = full_tile_indices

        w.close()

        print 'tile transfer matrix written to file: ', outfile

print '\nstep @4 complete: tile transfer matrix\n'

#@5
#-----------------------------------------------------
#  read in model data and convert to tile representation
#  calculate variance for use in damping matrix
#  here, GRACE is used as the model
#-----------------------------------------------------

if read_model_covariance_matrix:

    infile = model_covariance_matrix_directory + 'tellus.covariance_lnd.'+str(land_tile_size) +'_ocn.'+str(ocean_tile_size)+'.nc'
    print 'reading model covariance matrix from: ', infile

    f1 =  netcdf4.Dataset(infile, 'r', format='NETCDF4')
    mcov  = np.copy(f1.variables['model_covariance_matrix'][:,:])
    f1.close

    mrms = np.sqrt(np.diag(mcov)) #extract diagonal

else:
    print 'creating model covariance matrix'

    #--  read in tellus grace tws data
    f1 =  netcdf4.Dataset(tellus_grace_file, 'r', format='NETCDF4')
    tellus_tws  = np.copy(f1.variables['lwe_thickness'][:,:,:])
    tellus_time = np.copy(f1.variables['time'][:])
    tellus_lon  = np.copy(f1.variables['lon'][:])
    tellus_lat  = np.copy(f1.variables['lat'][:])
    im2=tellus_lon.size
    jm2=tellus_lat.size
    hm2=tellus_time.size
    f1.close
    
    #--  Remove mean storage terms  ----------------------------------
    for i in range(0,im2):
        for j in range(0,jm2):
            tellus_tws[:,j,i]-=np.mean(tellus_tws[:,j,i])

    #--  Convert from cm to mm equivalent water thickness
    tellus_tws = 10.0 * tellus_tws

    #--  read in tellus gain factors
    f1 =  netcdf4.Dataset(tellus_scaling_file, 'r', format='NETCDF4')
    tellus_gain_factor  = np.copy(f1.variables['SCALE_FACTOR'][:,:])
    f1.close
    
    print 'tellus data read in'

    ind=np.where(tellus_gain_factor > 3e4)
    if ind[0].size > 0:
        tellus_gain_factor[ind]=1.

    if 1 == 2: # plot for debugging purposes
        x0=0.1
        w0=0.95-x0
        y0=0.1
        h0=0.95-y0
        
        fig1=plt.figure()
        ax1=fig1.add_axes([x0,y0,w0,h0])
        lnum=10
        ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
        if 1==2:
            level=np.max(tellus_gain_factor)*ilevel
            con=plt.contourf(tellus_gain_factor,levels=level)
        else:
            level=np.max(tellus_tws[80,:,:])*ilevel
            con=plt.contourf(tellus_tws[80,:,:],levels=level)
        plt.ylabel(' y')
        plt.xlabel(' x')
        plt.title('gain factor')
        plt.show()
        
        stop

    th2  =np.pi/180.0*(90.0-tellus_lat)
    phi2 =np.pi/180.0*tellus_lon
    plm_th = tlib.gen_plm(lm,np.cos(th2))
    iclm =np.zeros((lm+1,lm+1),dtype=np.float64)
    islm =np.zeros((lm+1,lm+1),dtype=np.float64)
   
    mtiles=np.zeros((itot,hm2),dtype=np.float64)
    for t in range(0,hm2):
        tmp = tellus_gain_factor * tellus_tws[t,:,:]
        tlib.gen_mask_sh_coefficients(tmp,phi2,th2,iclm,islm,plm=plm_th)
        vclm=tlib.vectorize_sh_coefs(iclm,islm)
        mtiles[:,t]=np.dot(np.transpose(tmat_tile),vclm)

    print 'tellus data converted to tiles'

    m_coefs=np.zeros((6),dtype=np.float64)
    mrms=np.zeros((itot),dtype=np.float64)
    ocn_min=np.max(ocean_tile_indices)
    for i0 in range(0,itot):
        mrms[i0]=np.sqrt(np.sum(np.power(mtiles[i0,:],2))/np.float64(hm2))
        if full_tile_indices[i0] <= ocn_min:
            mrms[i0]=minimum_rms
        else:
            # mrms[i0]=max([minimum_rms,mrms[i0]])
            # amplify rms by some factor
            mrms[i0]=np.max([minimum_rms,rms_scale_factor*mrms[i0]])

    print 'model rms calculated'

    mcov=np.zeros((itot,itot),dtype=np.float64)
    for i0 in range(0,itot):
      for j0 in range(0,itot):
          mcov[i0,j0]=np.sum(mtiles[i0,:]*mtiles[j0,:])/np.float64(hm2)

    eps=1.e-12
    ind=np.where(np.bitwise_and((mcov < eps),(mcov >= 0.)))
    if ind[0].size > 0:
        mcov[ind]=eps
    ind=np.where(np.bitwise_and((mcov > -eps),(mcov <= 0.)))
    if ind[0].size > 0:
        mcov[ind]=-eps

    print 'model covariance calculated'


    if 1 == 2: # plot for debugging purposes
        x0=0.1
        w0=0.95-x0
        y0=0.1
        h0=0.95-y0
        
        fig1=plt.figure()
        ax1=fig1.add_axes([x0,y0,w0,h0])
        lnum=10
        ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
        if 1==1:
            level=np.max(mcov)*ilevel
            print level
            con=plt.contourf(mcov,levels=level)
        else:
            pass
            #level=np.max(tellus_tws[80,:,:])*ilevel
            #con=plt.contourf(tellus_tws[80,:,:],levels=level)
        plt.ylabel(' y')
        plt.xlabel(' x')
        plt.title('model covariance')
        plt.show()
        
        stop

    if not read_model_covariance_matrix:
        outfile = model_covariance_matrix_directory +'tellus.covariance_lnd.'+str(land_tile_size)+'_ocn.'+str(ocean_tile_size)+'.nc'

        w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
        command='date "+%y%m%d"'
        x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
        x=x2.communicate()
        timetag = x[0].strip()
        w.creation_date = timetag 
        w.title = 'Model Covariance Matrix'
        w.createDimension('itot',np.int32(itot))
    
        wmcov = w.createVariable('model_covariance_matrix','f8',('itot','itot'))

        # write to file  --------------------------------------------
        wmcov[:,:] = mcov
        w.close()

        print 'model covariance written to file: ', outfile

print '\nstep @5 complete: model covariance\n'

#@6
#-----------------------------------------------------
#  create damping matrix (alpha)
#  alpha_prime describes geographical correlations  
#  beta describes model variability
#-----------------------------------------------------

alpha=np.zeros((itot,itot),dtype=np.float64)
ocn_min=np.max(ocean_tile_indices)

if read_alpha_prime_matrix:
      ocn_min=np.max(ocean_tile_indices)
      infile=alpha_prime_matrix_directory+'lnd.'+str(land_tile_size)+'.g'+str(hw_land)+'_ocn.'+str(ocean_tile_size)+'.g'+str(hw_ocean)+'_alpha_prime.nc'

      alpha_prime=np.zeros((itot,itot),dtype=np.float64)

      f1 =  netcdf4.Dataset(infile, 'r', format='NETCDF4')
      alpha_prime  = np.copy(f1.variables['alpha_prime_matrix'][:,:])
      f1.close
      print 'alpha_prime read in'
else:
      print 'creating alpha matrix'  
      alpha=np.zeros((itot,itot),dtype=np.float64)
      alpha_prime=np.zeros((itot,itot),dtype=np.float64)
      ocn_min=np.max(ocean_tile_indices)
      
      alpha_prime=tlib.create_alpha_prime(blon,blat,full_tile_indices,
                                          ocn_min, full_tile_mask, 
                                          lmask, omask, 
                                          hw_land=hw_land,hw_ocn=hw_ocean)
      if not read_alpha_prime_matrix:
          outfile=alpha_prime_matrix_directory+'lnd.'+str(land_tile_size)+'.g'+str(hw_land)+'_ocn.'+str(ocean_tile_size)+'.g'+str(hw_ocean)+'_alpha_prime.nc'

          w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
          command='date "+%y%m%d"'
          x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
          x=x2.communicate()
          timetag = x[0].strip()
          w.creation_date = timetag 
          w.title = 'Alpha Prime Matrix'
          w.createDimension('itot',np.int32(itot))
          
          walpha = w.createVariable('alpha_prime_matrix','f8',('itot','itot'))
          
          # write to file  --------------------------------------------
          walpha[:,:] = alpha_prime
          w.close()

          print 'alpha prime matrix written to file: ', outfile

if 1 == 2: # plot for debugging purposes
    x0=0.1
    w0=0.95-x0
    y0=0.1
    h0=0.95-y0
    
    fig1=plt.figure()
    ax1=fig1.add_axes([x0,y0,w0,h0])
    lnum=10
    ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
    if 1==1:
        level=np.max(alpha_prime)*ilevel
        print level
        con=plt.contourf(alpha_prime,levels=level)
    else:
        pass
        #level=np.max(tellus_tws[80,:,:])*ilevel
        #con=plt.contourf(tellus_tws[80,:,:],levels=level)
    plt.ylabel(' y')
    plt.xlabel(' x')
    plt.title('model covariance')
    plt.show()
    
    stop


#  Create alpha matrix from alpha_prime matrix
if 1==1:
    # use model-derived variance to damp solution
    beta=np.dot(np.diag(1./mrms),alpha_prime)
    alpha=np.dot(np.transpose(beta),beta)
else:
    # apply global scale factor to alpha
    alpha=np.dot(np.transpose(alpha_prime),alpha_prime)
    alpha_scale_factor = 1.e-4
    alpha=alpha_scale_factor*alpha

print 'beta and alpha matrices created'

print '\nstep @6 complete: alpha matrix\n'

#@7
#-----------------------------------------------------
#  Calculate GRACE-derived covariance matrices
#  
#  BtB corresponds to eqn 21 in the derivation document
#  D corresponds to eqn 22 in the derivation document
#  
#-----------------------------------------------------

#--  Read in grace covariance matrix  ---------------------------------
if read_grace_covariance_matrix:
    infile=grace_covariance_matrix_directory+'lnd_'+str(land_tile_size)+'_ocn_'+str(ocean_tile_size)+'_grace_tile_cov_'+covlab+'.nc'

    HtH=np.zeros((ltot,ltot))
    DtHtH=np.zeros((itot,ltot))
    DtHtHD=np.zeros((itot,itot))

    f1 = netcdf4.Dataset(infile, 'r', format='NETCDF4')
    HtH     = np.copy(f1.variables['HtH'][:,:])
    DtHtH   = np.copy(f1.variables['DtHtH'][:,:])
    DtHtHD  = np.copy(f1.variables['DtHtHD'][:,:])
    f1.close
    print 'grace tile covariance matrices (HtH, DtHtH, DtHtHD) read in'

else:
    '''
    Creation and inversion of the BtB matrix are sensitive to the 
    precision of the calculation.  Ideally, the accumulation 
    of the various matrix elements should be done at a higher 
    precision than the matrix itself.  Below, the tmat_tile matrix
    is converted to single precision before calculating BtB with 
    double precision, and BtB is toggled between single and double 
    precision before being passed to the matrix inversion routine 
    for the same reason.  If the output of the script appears 
    suspect, the user should first check this step, and ensure 
    that BtB is calculated correctly and that the inversion if 
    successful, i.e. that BtB * [BtB]^-1 equals the identity.  
    Ultimately, it would be beneficial to have double precision 
    variables, with quadruple precision calculations.  
    '''

    tmp = np.copy(np.float32(tmat_tile))
    BtB = np.float32(linalg.blas.dgemm(alpha=1.0, a=tmp, b=tmp, trans_a=True))
    BtB = np.float64(BtB)

    #--  invert BtB -------------------------
    print 'inverting BtB matrix'
    x=tlib.invert_matrix(BtB)
    iBtB=x['inverse']
    
    D=np.dot(tmat_tile,iBtB)
            
    #--  read in covariance matrix
    if covnum == 1:
        x=tlib.read_2d_covmat_csr_rl05(covmatfile,reorder_matrix=True,
                                       full_matrix=True,lovenumfile=lovenumfile)
    cov=x['covariance']*1.e12
    degorder=x['degorder']
    print 'covariance matrix read in'
    
    #--  reduce (co)variances of lowest degrees (ad hoc)  -------------
    if 1 == 1:
        print 'Reducing low degree covariances!'
        sf=0.25
        for l in range(2,5):
            for m in range(0,l+1):
                ind=np.where(np.bitwise_and((degorder[:,0] == l),(degorder[:,1] == m)))[0]
                if ind.size == 1:
                    cov[ind[0],:]=cov[ind[0],:]*sf
                    cov[:,ind[0]]=cov[:,ind[0]]*sf
                if ind.size > 1:
                    for i in range(0,ind.size):
                        cov[ind[i],:]=cov[ind[i],:]*sf
                        cov[:,ind[i]]=cov[:,ind[i]]*sf
            
    #--  invert covariance matrix to get (H^T)H  -------------------------
    print 'inverting covariance matrix'
    x=tlib.invert_matrix(cov)
    HtH=x['inverse']
    print 'inversion status: ',x['status']
   
    #--  multiply HTH by D 
    DtHtH=np.dot(np.transpose(D),HtH)
    
    #  multiply DtHtH by D
    DtHtHD=np.dot(DtHtH,D)
    
    if not read_grace_covariance_matrix:
        outfile=grace_covariance_matrix_directory+'lnd_'+str(land_tile_size)+'_ocn_'+str(ocean_tile_size)+'_grace_tile_cov_'+covlab+'.nc'
        
        print outfile
        w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
        command='date "+%y%m%d"'
        x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
        x=x2.communicate()
        timetag = x[0].strip()
        w.creation_date = timetag 
        w.title = 'GRACE Tile Covariance Matrices'
        w.createDimension('itot',np.int32(itot))
        w.createDimension('ltot',np.int32(ltot))
        
        whth    = w.createVariable('HtH','f8',('ltot','ltot'))
        wdthth  = w.createVariable('DtHtH','f8',('itot','ltot'))
        wdththd = w.createVariable('DtHtHD','f8',('itot','itot'))
        
        # write to file  --------------------------------------------
        whth[:,:]    = HtH
        wdthth[:,:]  = DtHtH
        wdththd[:,:] = DtHtHD
        w.close()
        
        print 'grace tile covariance matrices written to file'

#--  invert DtHthD
print 'inverting DtHtHD'

x=tlib.invert_matrix((DtHtHD+alpha))
cov2=x['inverse']
print 'inversion status: ',x['status']

print '\nstep @7 complete: covariance matrix inverted\n'

#@8
#-----------------------------------------------------
#  
#  Convert Stokes coefficients to tiles in mass units  
#  
#-----------------------------------------------------

#--  create full time series  ------------------------
storage=np.zeros((tm,bjm,bim),dtype=np.float32)
kl=tlib.get_love(lovenumfile,lm)
for t in range(0,tm):
    iclm=np.copy(clm[:,:,t])
    islm=np.copy(slm[:,:,t])
    tlib.convert_stokes_to_mass(iclm,islm,kl)
    
    #--  compute RHS by multiplying BTHTH and x   
    vclm=tlib.vectorize_sh_coefs(iclm,islm)

    if 1 == 2: # plot for debugging purposes
        x0=0.1
        w0=0.95-x0
        y0=0.1
        h0=0.95-y0
        
        fig1=plt.figure()
        ax1=fig1.add_axes([x0,y0,w0,h0])

        plt.plot(vclm)

        plt.ylabel(' y')
        plt.xlabel(' x')
        plt.title('vclm')
        plt.show()
        
        stop

    DtHty=np.dot(DtHtH,vclm)

    #--  multiply BtHty by new inverse covariance matrix to get tiles
    vtiles=np.dot(cov2,DtHty)
    
    # test basic tile averaging
    if 1==2:
        vtiles = np.dot(np.transpose(tmat_tile),vclm)

    bgrid=np.zeros((bjm,bim),dtype=np.float32)
    for i0 in range(0,itot):
        ind=np.where(full_tile_mask == full_tile_indices[i0])
        bgrid[ind]=vtiles[i0]

    #--  add geocenter terms      
    iclm2=np.copy(geoclm[:,:,t])
    islm2=np.copy(geoslm[:,:,t])
    tlib.convert_stokes_to_mass,iclm2,islm2,kl
    degree_one=tlib.calc_sig(iclm2,islm2,phi,th)
    bgrid+=np.transpose(degree_one)

    storage[t,:,:]=bgrid

if write_tile_tws:
    outfile=tile_tws_directory+'tws_lnd_'+str(land_tile_size)+'_ocn_'+str(ocean_tile_size)+'_tiles.nc'

    print outfile
    w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
    command='date "+%y%m%d"'
    x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
    x=x2.communicate()
    timetag = x[0].strip()
    w.creation_date = timetag 
    w.title = 'GRACE Tile Total Water Storage'
    w.createDimension('lon',np.int32(bim))
    w.createDimension('lat',np.int32(bjm))
    w.createDimension('time',np.int32(tm))
          
    wtime = w.createVariable('time','f8',('time'))
    wlon  = w.createVariable('lon','f8',('lon'))
    wlat  = w.createVariable('lat','f8',('lat'))
    wtws  = w.createVariable('tws','f8',('time','lat','lon'))
          
    # write to file  --------------------------------------------
    wtime[:]  = gtime 
    wlon[:]   = blon
    wlat[:]   = blat
    wtws[:,:,:] = storage
    w.close()

    print 'tile tws written to file'

print '\nstep @8 complete: tile tws\n'
