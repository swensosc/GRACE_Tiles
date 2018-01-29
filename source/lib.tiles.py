#! /usr/bin/env python
import sys
import os
import string
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.special import sph_harm
from scipy import linalg
import netCDF4 as netcdf4

""" -------------------------------------------------------

Functions used to calculate tile averages of grace data   

------------------------------------------------------- """

#@1 
#--  read land/ocean mask  --------------------------------------
def read_land_mask(lon,lat,maskfile,neglon=False):

    f1 =  netcdf4.Dataset(maskfile, 'r', format='NETCDF4')
    lon1  = np.copy(f1.variables['longitude'][:])
    lat1  = np.copy(f1.variables['latitude'][:])
    mask1 = np.copy(f1.variables['mask'][:,:])
    f1.close

    if neglon:
        lon=np.where(lon < 0.0,lon+360.0,lon)
        lon=np.roll(lon,lon.size/2)

    #x=interpolate.RectBivariateSpline(np.flipud(lat1),lon1,np.flipud(mask1))
    x=interpolate.RectBivariateSpline(lat1,lon1,mask1)
    tmp=x(lat,lon)
    lthresh = 0.0
    lthresh = 0.8
    mask=np.where(tmp > lthresh,1,0)

    if neglon:
        mask=np.roll(mask,-lon.size/2,axis=1)

    if 1 == 2: # plot for debugging purposes
        fig1=plt.figure()
        ax1=fig1.add_axes()
        lnum=20
        ilevel=np.arange(lnum,dtype=float)/np.float(lnum-1)
        level=1.25*ilevel
        print level
        #con=plt.contourf(lon1,lat1,mask1,levels=level)
        #con=plt.contourf(lon,lat,tmp,levels=level)
        con=plt.contourf(lon,lat,mask,levels=level)
        plt.ylabel('Lat')
        plt.xlabel('Lon')
        plt.title('Data')
        plt.show()
        stop
    return mask

#@2
#--  Calculate a discrete laplacian from four cardinal points  ---
def four_point_laplacian(mask):
    jm=mask.shape[0]
    im=mask.shape[1]

    laplacian = -4.0 * np.copy(mask)
    laplacian += (mask * np.roll(mask,1,axis=1) 
                  + mask * np.roll(mask,-1,axis=1))
    temp = np.roll(mask,1,axis=0)
    temp[0,:] = mask[1,:]
    laplacian += mask * temp
    temp = np.roll(mask,-1,axis=0)
    temp[jm-1,:] = mask[jm-2,:]
    laplacian += mask * temp

    return np.abs(laplacian)

#@3
#--  remove isolated points in a mask  ---------------------------
def mask_remove_points(imask,nm=20,shift=False):
    mask=np.copy(imask)
    jm=mask.shape[0]
    im=mask.shape[1]

    if shift:
        mask=np.roll(mask,0,im/2)

    #--  nm specifies size of regions to remove
    for i in range(nm):
        mask0 = np.where(four_point_laplacian(mask) >= 3,1,0)
        if np.sum(mask0) == 0:
            break
        mask-=mask0

    if shift:
        mask=np.roll(mask,0,-im/2)

    return mask

#@4
#--  Define rectangular tiles on input coordinate grid ------
#--  all points in a tile are given the same index

def tile_mask(blon,blat,bdel):
    bim  = blon.size
    bjm  = blat.size
    bim2 = 360/bdel
    bjm2 = 180/bdel

    blon2=-180.0+(np.arange(bim2)+0.5)*bdel
    blat2=-90.0+(np.arange(bjm2)+0.5)*bdel
    mask=np.zeros((bjm,bim))
    ocnt=np.int32(-1)
    '''
    print np.min(blon),np.max(blon)
    print np.min(blon2),np.max(blon2)
    print np.min(blat),np.max(blat)
    print np.min(blat2),np.max(blat2)
    '''

    for i in range(bim2):
        if i < bim2-1:
            ind2=np.where(np.bitwise_and((blon >= blon2[i]),(blon < blon2[i+1])))[0]
        else:
            ind2=np.where(np.bitwise_or((blon >= blon2[i]),(blon < blon2[0])))[0]

        for j in range(bjm2-1):
            ind=np.where(np.bitwise_and((blat >= blat2[j]),(blat < blat2[j+1])))[0]
            if (ind.size > 0 and ind2.size > 0):
                for k in range(ind.size):
                    mask[ind[k],ind2]=ocnt
                ocnt-=1

        j=0
        ind=np.where(blat < blat2[j])[0]
        if (ind.size > 0 and ind2.size > 0):
            for k in range(ind.size):
                mask[ind[k],ind2]=ocnt
            ocnt-=1

        j=bjm2-1
        ind=np.where(blat >= blat2[j])[0]
        if (ind.size > 0 and ind2.size > 0):
            for k in range(ind.size):
                mask[ind[k],ind2]=ocnt
            ocnt-=1

    return mask

#@5
#--  Read in release 5 GRACE data, corrected for degree 1, 
#--  C20, Glacial Isostatic Adjustment (GIA), and converted to mass

def read_grace_data_rl05(grace_data_dir=None,geocenter_file=None,
                         c20_file=None, gia_file=None):
    if grace_data_dir == None:
        sys.exit('Must specify location of GRACE data')

    command='ls '+grace_data_dir+'GSM*_0005'
    x2=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True')
    infiles = np.asarray(x2.communicate()[0].split(),dtype='string')
    tm=infiles.size

    #--  Get mid-month time values in days from jan 2002  -------------
    gtime=np.zeros(tm)
    for t in range(tm):
        istr=infiles[t]

        #--  Determine time [days from 2002.01.01]
        y=istr.find('GSM-')
        
        yr        = np.int32(istr[y+6:y+10])
        yr2       = np.int32(istr[y+14:y+18])
        start_day = np.int32(istr[y+10:y+13])
        end_day   = np.int32(istr[y+18:y+21])
        #print yr,start_day,yr2,end_day

        istart=0     
        if yr != yr2:
            end_day=(yr2-yr)*365+end_day
        for iyr in range(2002,yr):
            if np.mod(iyr,4) == 0:
                dpm=[31,29,31,30,31,30,31,31,30,31,30,31]
            else:
                dpm=[31,28,31,30,31,30,31,31,30,31,30,31]
            istart=istart+np.sum(dpm)
            
        gtime[t]=np.mean([istart+start_day,istart+end_day])

    #--  Specify maximum SH degree  ---------------------
    lm=60

    #--  Read in spherical harmonic coefficients  -------
    clm=np.zeros((lm+1,lm+1,tm),dtype=np.float64)
    slm=np.zeros((lm+1,lm+1,tm),dtype=np.float64)
    clm1=np.zeros((lm+1,lm+1),dtype=np.float64)
    slm1=np.zeros((lm+1,lm+1),dtype=np.float64)

    for t in range(tm):
        datafile=infiles[t]
        read_csr_rl05_sh_coefficients(clm1,slm1,datafile)
        clm[:,:,t]=clm1
        slm[:,:,t]=slm1

    #--  set lowest degrees to zero  ---------------------
    clm[0,0,:]   = 0.0
    clm[1,0:1,:] = 0.0
    slm[0,0,:]   = 0.0
    slm[1,0:1,:] = 0.0

    #--  Use grace/omct based geocenter values  -----------------
    if geocenter_file != None:
        x=read_degree_one_coefficients(geocenter_file, gtime, 
                                       convert_to_stokes=True)
        clm[1,0,:]=x['c10']
        clm[1,1,:]=x['c11']
        slm[1,1,:]=x['s11']

    #--  Use CSR SLR based C20 values  --------------------------
    if c20_file != None:
        x=read_c20_coefficient(c20_file, gtime)
        clm[2,0,:]=x['c20']
    else:
        sys.exit('Must provide valid C20 file')

    #--  Remove GIA estimate  --------------------------
    if gia_file != None:
        x=read_gia_coefficients(gia_file)
        gia_clm = x['gia_clm'][0:lm+1,0:lm+1]
        gia_slm = x['gia_slm'][0:lm+1,0:lm+1]

        if 1 == 2: # plot for debugging purposes
            x0=0.1
            w0=0.95-x0
            y0=0.1
            h0=0.95-y0
            
            fig1=plt.figure()
            ax1=fig1.add_axes([x0,y0,w0,h0])
            lnum=20
            ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
            if 1 == 1:
                level=np.max(gia_clm)*ilevel
                con=plt.contourf(np.transpose(gia_clm),levels=level)
            else:
                level=np.max(gia_slm)*ilevel
                con=plt.contourf(np.transpose(gia_slm),levels=level)
            plt.ylabel('l')
            plt.xlabel('m')
            plt.title('clm/slm')
            plt.show()
            stop
        
        for t in range(tm):
            clm[:,:,t]-=(gia_clm*(gtime[t]-np.mean(gtime))/365.0)
            slm[:,:,t]-=(gia_slm*(gtime[t]-np.mean(gtime))/365.0)
    else:
        sys.exit('Must provide valid GIA file')

    return {'clm':clm,'slm':slm,'time':gtime}

#@6
#--  Read in CSR release 5 GRACE unconstrained SH coefficients
#--  
def read_csr_rl05_sh_coefficients(gclm,gslm,datafile):

    lmax=gclm.shape[0]
    stokes=np.zeros((2))
    errors=np.zeros((2))

    with open(datafile,'r') as w:
        tmp=(w.read()).splitlines()

    for n in range(0,len(tmp)):
        x=tmp[n].split()
        if len(x) > 0:
            if x[0] == 'SHM':
                lm = np.int32(x[1])
    
    #print 'lm: ',lm

    for n in range(0,len(tmp)):
        x=tmp[n].split()
        if len(x) > 0:
            if x[0] == 'GRCOF2':
                l1=np.int32(x[1])
                m1=np.int32(x[2])
                gclm[l1,m1] = np.float64(x[3])
                gslm[l1,m1] = np.float64(x[4])

    if 1 == 2: # plot for debugging purposes
        x0=0.1
        w0=0.95-x0
        y0=0.1
        h0=0.95-y0
        
        fig1=plt.figure()
        ax1=fig1.add_axes([x0,y0,w0,h0])
        lnum=20
        ilevel=(np.arange(lnum,dtype=float)/np.float(lnum-1) - 0.5)*2.
        if 1 == 2:
            level=np.max(gclm)*ilevel
            con=plt.contourf(np.transpose(gclm),levels=level)
        else:
            level=0.5*np.max(gslm)*ilevel
            con=plt.contourf(np.transpose(gslm),levels=level)
        plt.ylabel('l')
        plt.xlabel('m')
        plt.title('clm/slm')
        plt.show()
        stop

#@7
#--  Read in degree one grace coefficients
#--  
def read_degree_one_coefficients(geocenter_file, gtime, 
                                 convert_to_stokes=False):
    with open(geocenter_file,'r') as w:
        x=(w.read()).splitlines()
    nm=len(x)
    itime=[]
    ic10=[] ; ic11=[] ; is11=[]
    for n in range(3,nm):
        tmp=x[n].split()
        itime=np.append(itime,np.float64(tmp[0]))
        ic10=np.append(ic10,np.float64(tmp[1]))
        ic11=np.append(ic11,np.float64(tmp[2]))
        is11=np.append(is11,np.float64(tmp[3]))
    itime=(itime - 2002) * 365

    fint = interpolate.interp1d(itime,ic10)
    c10 = fint(gtime)
    fint = interpolate.interp1d(itime,ic11)
    c11 = fint(gtime)
    fint = interpolate.interp1d(itime,is11)
    s11 = fint(gtime)

    if 1 == 2: # plot for debugging purposes
        fig1=plt.figure()
        ax1=fig1.add_axes()
        plt.plot(itime,is11,color='g')
        plt.plot(gtime,s11,'bo')
        plt.ylabel('[]')
        plt.xlabel('time')
        plt.title('geocenter interpolation')
        plt.show()

    #  convert from mass to stokes
    if convert_to_stokes:
        rho_e = 5.517e3
        a     = 6.371e6
        coef  = (1.0)/(2.0*np.arange(2,dtype=np.float64)+1.0)/(a*rho_e/3.0)
        
        c10 = c10*coef[1]
        c11 = c11*coef[1]
        s11 = s11*coef[1]

    return {'c10':c10,'c11':c11,'s11':s11}

#@8
#--  Read in CSR SLR-based C20 coefficient values
#--  
def read_c20_coefficient(c20_file, gtime):
    with open(c20_file,'r') as w:
        x=(w.read()).splitlines()
    nm=len(x)

    for n in range(nm-1):
         tmp=x[n].split()
         if len(tmp) > 0:
             if tmp[0] == 'PRODUCT:':
                 nstart = n

    itime=[]
    ic20=[]
    ac20=[]
    scale_factor = np.float64(1.0e-10)
    for n in range(nstart,nm):
        tmp=x[n].split()
        if len(tmp) >= 5:
            itime=np.append(itime,np.float64(tmp[1]))
            ac20=np.append(ac20,np.float64(tmp[2]))
            ic20=np.append(ic20,scale_factor*np.float64(tmp[3]))

    itime=(itime - 2002) * 365
    ac20-=np.mean(ac20)
    ic20-=np.mean(ic20)

    fint = interpolate.interp1d(itime,ic20)
    c20 = fint(gtime)
    fint = interpolate.interp1d(itime,ac20)
    bc20 = fint(gtime)

    if 1 == 2: # plot for debugging purposes
        fig1=plt.figure()
        ax1=fig1.add_axes()
        plt.plot(itime,ic20,color='g')
        #plt.plot(itime,ac20,'bo')
        plt.plot(gtime,c20,'ro')
        plt.ylabel('[]')
        plt.xlabel('time')
        plt.title('C20 interpolation')
        plt.show()

    return {'c20':c20}

#@9
#--  Read in GIA coefficients  ----------------------
#--  
def read_gia_coefficients(gia_file):
    with open(gia_file,'r') as w:
        x=(w.read()).splitlines()
    nm=len(x)

    tmp=x[nm-1].split()
    lm=np.int32(tmp[0])

    clm=np.zeros((lm+1,lm+1),dtype=np.float64)
    slm=np.zeros((lm+1,lm+1),dtype=np.float64)

    for n in range(1,nm):
         tmp=x[n].split()
         l1=np.int32(tmp[0])
         m1=np.int32(tmp[1])
         clm[l1,m1] = np.float64(tmp[2])
         slm[l1,m1] = np.float64(tmp[3])

    return {'gia_clm':clm,'gia_slm':slm}

#@10
#--  Calculate spherical harmonic coefficients for a mask
def gen_mask_sh_coefficients(mask,phi,th,gclm,gslm,plm=None):
    lmax=gclm.shape[0] - 1
    mmax=gclm.shape[0] - 1

    # equispaced grid is assumed
    dphi = np.abs(phi[0] - phi[1])
    dth = np.abs(th[0] - th[1])

    phi2d, th2d = np.meshgrid(phi,th)

    bjm=th.size
    bim=phi.size

    fmask=np.float64(np.copy(mask))
    for i in range(bim):
        fmask[:,i] = fmask[:,i]*np.sin(th)*dth*dphi

    # use fast associated legendre function
    if plm is None:
        plm=gen_plm(lmax,np.cos(th))

    for l in range(lmax+1):
        for m in range(l+1):
            # scipy version is relatively slow
            # input: m=order,n=degree,theta=azimuth,phi=colat
            # ylm = np.float64(sph_harm(m, l, phi2d, th2d))
            # geodesy_normalization = 
            # (np.power(-1,m) * np.power((4.0 * (1+1*(m > 0)))*np.pi,0.5) 
            # clm=np.real(np.copy(ylm)) * geodesy_normalization
            # slm=np.imag(np.copy(ylm)) * geodesy_normalization

            clm = np.outer(plm[l,m,:],np.cos(m*phi))
            slm = np.outer(plm[l,m,:],np.sin(m*phi))

            norm = 1.0 / (4.0 * np.pi)

            gclm[l,m] = np.sum(np.multiply(fmask,clm)) * norm
            gslm[l,m] = np.sum(np.multiply(fmask,slm)) * norm


#@11
#--  Calculate associated legendre polynomials with geodesy normalization
def gen_plm(lmax,costheta):
#  Uses Martin Mohlenkamp's recursion relation; returns plms 
#  (geodesy normalization) in plm;  costheta is cos(co-lat)

    jm=costheta.size
    plm=np.zeros((lmax+1,lmax+1,jm))
    ptemp=np.zeros((lmax+1,jm))    
    rsin=np.power((1.0-np.power(costheta,2)),0.5)

    for mm in range(lmax):
        #  Initialize recurrence relation       
        ptemp[0,:]=1.0/np.sqrt(2.0)       
        for j in range(1,mm+1):
            ptemp[0,:]=ptemp[0,:]*np.sqrt(1.0+1.0/2.0/np.float64(j))
        
        if ((lmax-mm) > 0):
            ptemp[1,:]=(2.0*costheta*ptemp[0,:] 
                        *np.sqrt(1.0+(np.float64(mm)-0.5)/np.float64(1)) 
                        *np.sqrt(1.0-(np.float64(mm)-0.5)/np.float64(1+2*mm)))
            for k in range(2,(lmax-mm+1)):
                ptemp[k,:]=(2.0*costheta*ptemp[k-1,:] 
                            *np.sqrt(1.0+(np.float64(mm)-0.5)/np.float64(k)) 
                            *np.sqrt(1.0-(np.float64(mm)-0.5)/np.float64(k+2*mm)) 
                            -ptemp[k-2,:]*np.sqrt(1.0+4.0/np.float64(2*k+2*mm-3)) 
                            *np.sqrt(1.0-1.0/np.float64(k))
                            *np.sqrt(1.0-1.0/np.float64(k+2*mm)))

        #---  Normalization is geodesy convention  ------------------------!
        for l in range(mm,lmax+1):
            if (mm == 0):
                plm[l,mm,:]=np.sqrt(2.0)*ptemp[l-mm,:]
            else:
                plm[l,mm,:]=2.0*np.power(rsin,mm)*ptemp[l-mm,:]
    return plm

#@12
#-- convert from 2d to 1d, for degree 2 and higher 
def vectorize_sh_coefs(clm,slm):
    lm=clm.shape[0] -1
    ltotc=(np.power(lm,2)+3*lm-4)/2         #for l=2,lm; m=0,l (i.e. CLM)
    ltots=(np.power(lm,2)+lm-2)/2           #for l=2,lm; m=1,l (i.e. SLM)
    ltot=ltotc+ltots

    dtype = clm.dtype
    #dvect = np.zeros(ltot,dtype=np.float64)
    dvect = np.zeros(ltot,dtype=dtype)
    cnt=0
    # convert clm to vector format
    for l in range(2,lm+1):
        for m in range(0,l+1):
            dvect[cnt]=clm[l,m]
            cnt+=1
            
    # convert slm to vector format
    for l in range(2,lm+1):
        for m in range(1,l+1):
            dvect[cnt]=slm[l,m]
            cnt+=1

    return dvect

#@13
#-- revert vectorized coefficients to 2d 
def devectorize_sh_coefs(cslm,lm):
    clm=np.zeros((lm+1,lm+1),dtype=np.float64)
    slm=np.zeros((lm+1,lm+1),dtype=np.float64)
    cnt=0
    for l in range(2,lm+1):
        for m in range(0,l+1):
            clm[l,m]=cslm[cnt]
            cnt+=1

    for l in range(2,lm+1):
        for m in range(1,l+1):
            slm[l,m]=cslm[cnt]
            cnt+=1

    return {'clm':clm,'slm':slm}

#@13
#--  create a gaussian mask centered at a specified point
#--  with a specified half-width (all inputs are [degrees])
def gauss_mask(hw,lon1,lat1,lon,lat):
    phi1  = (np.pi/180.)*lon1
    th1   = (np.pi/180.)*(90. - lat1)
    phi   = (np.pi/180.)*lon
    th    = (np.pi/180.)*(90. - lat)
    radhw = (np.pi/180.)*hw
 
    im=lon.size
    jm=lat.size
    sinth1=np.sin(th1)
    costh1=np.cos(th1)
    
    tmp3=np.outer(np.ones((jm)),np.cos(phi-phi1))
    tmp1=np.outer(np.cos(th),costh1*np.ones((im)))
    tmp2=np.outer(np.sin(th),sinth1*np.ones((im)))

    cosalpha=tmp1+tmp2*tmp3
    ind=np.where(cosalpha > 1.0)
    cosalpha[ind]=1.0
    ind=np.where(cosalpha < -1.0)
    cosalpha[ind]=-1.0
    
    gmask=np.exp(-(np.power(np.arccos(cosalpha)/(np.sqrt(2.0)*radhw),2)))
    
    #ind=np.where(not np.isfinite(gmask))
    #if ind[0].size > 0:
    # if any non-finite points remain then abort
    if np.any(np.logical_not(np.isfinite(gmask))):
        sys.exit('error in gauss_mask')

    return gmask

#@14
#-- 
def create_alpha_prime(blon,blat,tile_index,ocn_min_ndx,
                            tile_mask,lmask,omask, 
                            hw_land=None,hw_ocn=None):
    #--  alpha_prime is the matrix that transforms the vector x 
    #--  to the vector minus its regional average: x -> (x-x_avg)
    #--  thus changing the problem from minimizing x to minimizing 
    #--  the departure of x from the regional average
    
    #  indices greater than ocn_min_ndx designate land points
    itot=tile_index.size
    bim =blon.size
    bjm =blat.size
    
    alpha_prime=np.zeros((itot,itot),dtype=np.float64)
    
    blon2d, blat2d = np.meshgrid(blon,blat)
    
    #--  check that both gaussian halfwidths exist
    if (((hw_land is not None) and (hw_ocn is None)) or 
        ((hw_land is None) and (hw_ocn is not None))):
        sys.exit('must specify both land and ocean correlation lengthscales')

    #--  if no hw values specified, create identity matrix
    if ((hw_land is None) and (hw_ocn is None)):
        print 'creating diagonal alpha_prime'
        #--  Loop over each tile
        for i0 in range(0,itot):
            alpha_prime[i0,i0]=1.0

    #--  if both hw values specified, create matrix w/ off-diagonal terms
    if ((hw_land is not None) and (hw_ocn is not None)):
        print 'creating non-diagonal alpha_prime'

    #--  Loop over each tile  ---------------------------------
    for i0 in range(0,itot):
        #--  Land tiles  --------------------------------------
        if tile_index[i0] > ocn_min_ndx:
            ind=np.where(tile_index[i0] == tile_mask)
            if ind[0].size > 0:
                #--  calculate center of each tile
                mlon=np.mean(blon2d[ind])
                mlat=np.mean(blat2d[ind])
                # vary correlation hw w/ latitude
                #hwlat=hw_land*(1.-mlat/180.) #reduce by half at pole
                hwlat=hw_land*(1.-mlat/135.) #reduce by 2/3 at pole
                gmask=gauss_mask(hwlat,mlon,mlat,blon,blat)
              
                gmask=gmask * np.float64(lmask)
              
                gmask=gmask/np.sum(gmask)
                for j0 in range(0,itot):
                    if tile_index[j0] > ocn_min_ndx:
                        ind2=np.where(tile_index[j0] == tile_mask)
                        if ind2[0].size > 0:
                            alpha_prime[i0,j0] = -np.sum(gmask[ind2])
                    if i0 == j0:
                        alpha_prime[i0,j0]+=1.0
        else:
            #--  Ocean tiles  ---------------------------------------------
            ind=np.where(tile_index[i0] == tile_mask)
            if ind[0].size > 0:
                mlon=np.mean(blon2d[ind])
                mlat=np.mean(blat2d[ind])
                # vary correlation hw w/ latitude
                hwlat=hw_ocn*(1.-mlat/135.)
                gmask=gauss_mask(hwlat,mlon,mlat,blon,blat)
              
                gmask=gmask * np.float64(omask)
              
                gmask=gmask/np.sum(gmask)
                for j0 in range(0,itot):
                    if tile_index[j0] < ocn_min_ndx:
                        ind2=np.where(tile_index[j0] == tile_mask)
                        if ind2[0].size > 0:
                            alpha_prime[i0,j0] = -np.sum(gmask[ind2])
                    if i0 == j0:
                        alpha_prime[i0,j0]+=1.0

    return alpha_prime

#@16
#-- 
def read_2d_covmat_csr_rl05(cfile,convert_to_mass=False,
                            reorder_matrix=False,full_matrix=False,
                            verbose=False,lovenumfile=None):
    verbose = True

    #--  calculate number of elements in covariance matrix  
    lm=60
    #ltot=(np.power(lm,2)+3*lm+2)/2 #for l=0,lm
    ltotc=(np.power(lm,2)+3*lm-4)/2 #for l=2,lm; m=0,l (i.e. CLM)
    ltots=(np.power(lm,2)+lm-2)/2   #for l=2,lm; m=1,l (i.e. SLM)
    ltot=ltotc+ltots

    with open(cfile,'r') as w:
        #--  read in file information  -------------------------------
        for i in range(0,6):
            x=(w.readline())
            #print x

        #--  read in number of parameters  ---------------------------
        nparams=np.int32(w.readline())
        print 'number of parameters in covariance matrix: ', nparams

        #--  read in coefficient ordering information
        nm=6  # there are six coefficients per line of this text file
        cnt=np.int32(nparams/nm)
        degorder=np.zeros((nparams,2))
        print nm, cnt
        for k in range(0,cnt+1):
            cstr=w.readline()
            jm=cstr.count('GEO')
            k0 = 0
            for j in range(0,jm):
                x=cstr.find('GEO',k0)
                k0=x+5
                degree=np.int32(cstr[k0:k0+2])
                order=np.int32(cstr[k0+3:k0+5])
                #print degree, order
                degorder[k*nm+j,0]=degree
                degorder[k*nm+j,1]=order

        #--  read in stokes coefficients [nparams - 6]
        x=np.asarray((w.readline().split()),dtype=np.float64)        
        cnum = x.size
        n = 0
        fld = x
        nm = nparams / cnum
        for n in range(1,nm+1):
            x=np.asarray((w.readline().split()),dtype=np.float64)        
            fld = np.append(fld,x)

        if verbose:
            print 'Stokes coefficients read'

        slm=np.zeros((lm+1,lm+1),dtype=np.float64)
        clm=np.zeros((lm+1,lm+1),dtype=np.float64)
        l1=0
        #l1=6 # don't include first six parameters
        for l in range(2,lm+1):
            clm[l,0]=fld[l1]
            l1+=1

        for l in range(2,lm+1):
            for m in range(1,l+1):
                clm[l,m]=fld[l1]
                slm[l,m]=fld[l1+1]
                l1+=2

        #--  read in sigmas  ----------------------------------------
        x=np.asarray((w.readline().split()),dtype=np.float64)        
        cnum = x.size
        n = 0
        fld = x
        nm = nparams / cnum
        for n in range(1,nm+1):
            x=np.asarray((w.readline().split()),dtype=np.float64)        
            fld = np.append(fld,x)
        variance=np.float64(fld[0:nparams])

        #--  read in scalefactor  -----------------------------------
        x=np.asarray((w.readline().split()),dtype=np.float64)        
        cnum = x.size
        n = 0
        fld = x
        nm = nparams / cnum
        for n in range(1,nm+1):
            x=np.asarray((w.readline().split()),dtype=np.float64)        
            fld = np.append(fld,x)
        # (currently ignoring scale factors...)

        if verbose:
            print 'Sigmas and scale factors read'

        #--  read in full covariance matrix (upper triangle)
        covariance=np.zeros((ltot,ltot),dtype=np.float64)
        nm=4
        for l in range(0,ltot):
            cnt2=(ltot-l)/nm -1 + (np.mod((ltot-l),nm) > 0)
            for k in range(0,cnt2+1):
                x=np.asarray((w.readline().split()),dtype=np.float64)        
                jm=x.size
                covariance[l,l+k*nm:l+k*nm+jm]=x

    if verbose:
        print 'Covariance matrix elements read'

    if full_matrix:
        for l in range(1,ltot):
            covariance[l,0:l-1]=covariance[0:l-1,l]

    if reorder_matrix:
        print 'attempting to reorder covariance matrix'
        degorder2=np.zeros((ltot,2))
        #--  first setup alternate index
        lc=0
        for l in range(2,lm+1):
            for m in range(0,l+1):
                degorder2[lc,0]=l
                degorder2[lc,1]=m
                lc+=1
                
        for l in range(2,lm+1):
            for m in range(1,l+1):
                degorder2[lc,0]=l
                degorder2[lc,1]=m
                lc+=1

        ltol=np.zeros((ltot),dtype=np.int32)
        lc=0
        m=0
        for l in range(2,lm+1):
            ind=np.where(np.bitwise_and((degorder2[:,0] == l),
                                        (degorder2[:,1] == m)))[0]
            ltol[lc]=ind[0]
            lc+=1
            
        for l in range(2,lm+1):
            for m in range(1,l+1):
                ind=np.where(np.bitwise_and((degorder2[:,0] == l),
                                            (degorder2[:,1] == m)))[0]
                #            print, ind
                ltol[lc]=ind[0]
                ltol[lc+1]=ind[1]
                lc+=2
                
        #--  now convert covariance matrix
        covariance2=np.zeros((ltot,ltot),dtype=np.float64)
        variance2=np.zeros((ltot),dtype=np.float64)
        for j1 in range(0,ltot):
            variance2[ltol[j1]]=variance[j1]
            for j2 in range(0,ltot):
                covariance2[ltol[j1],ltol[j2]]=covariance[j1,j2]

        covariance=covariance2
        degorder=degorder2
        variance=variance2

    if convert_to_mass:
        #  convert to mm^2  
        rho_e=5.517e3
        a=6.371e6
        kl=get_love(lovenumfile,lm)
        coef=(2.0*np.arange(lm+1)+1.0)/(1.0+kl)*(a*rho_e/3.0)
        for l in range(2,lm+1):
            for k in range(2,lm+1):
                tmp=coef[l]*coef[k]*1.0e-12
                clmcov[l,0:l,k,0:k]=clmcov[l,0:l,k,0:k]*tmp
                slmcov[l,0:l,k,0:k]=slmcov[l,0:l,k,0:k]*tmp

    return {'clm':clm,'slm':slm,'covariance':covariance,'degorder':degorder,'variance':variance}

#@17
#-- 
def get_love(lfile,lmax):
    kl_out=np.zeros((lmax+1),dtype=np.float32)
    with open(lfile,'r') as f:
        tmp=f.readlines()

    tmp=tmp[2:lmax+3]
    for l in range(0,lmax+1):
        x=tmp[l].split()
        kl_out[l] = np.float64(x[2].replace('D','E'))
    #for l in range(0,lmax+1):
    #    print l,kl_out[l]
    return kl_out

#@18
#-- 
def convert_stokes_to_mass(clm,slm,kl,rad=None):

    #kl are love numbers, units are mm
    lm=clm.shape[0] - 1
    rho_e=5.517e3
    a=6.371e6
    coef=(2.0*np.arange(lm+1)+1.0)/(1.0+kl)*(a*rho_e/3.0)

    if rad is not None:
        wl=gauss_weights(rad,lm)
        for l in range(0,lm+1):
            clm[l,0:l+1]=clm[l,0:l+1]*coef[l]*(2.0*np.pi*wl[l])
            slm[l,0:l+1]=slm[l,0:l+1]*coef[l]*(2.0*np.pi*wl[l])
    else:
        for l in range(0,lm+1):
            clm[l,0:l+1]=clm[l,0:l+1]*coef[l]
            slm[l,0:l+1]=slm[l,0:l+1]*coef[l]

#@19
#-- 
def gauss_weights(hw,lm):
    wl=np.zeros((lm+1),dtype=np.float64)
    if (hw < 1.0e-6):
        wl[:]=1.0/(2.0*np.pi)
    else:
        b=np.log(2.0)/(1.0-np.cos(np.float64(hw/6371.0)))
        wl[0]=1.0/(2.0*np.pi)
        wl[1]=wl(0)*((1.0+np.exp(-2.0*b))/(1.0-np.exp(-2.0*b))-1.0/b)
        valid=True
        l=2
        while(valid and l < lm):
            wl[l]=(1.0-2.0*l)/b*wl[l-1]+wl[l-2]
            if (np.abs(wl[l]) < 1.0e-7):
                wl[l:lm+1] = 1.0e-7
                valid=False
            l+=1
    return wl

#@20
#--
def calc_sig(clm,slm,phi,th,plm=None):

    thmax=th.size
    phimax=phi.size

    lmax=clm.shape[0] - 1

    # use fast associated legendre function
    if plm is None:
        plm=gen_plm(lmax,np.cos(th))

    #--  Calculate fourier coefficients from legendre coefficients  -----! 
    d_cos=np.zeros((lmax+1,thmax),dtype=np.float64)
    d_sin=np.zeros((lmax+1,thmax),dtype=np.float64)
    for k in range(0,thmax):
        d_cos[:,k]=np.sum(plm[0:lmax+1,0:lmax+1,k]*clm[0:lmax+1,0:lmax+1],axis=0)
        d_sin[:,k]=np.sum(plm[0:lmax+1,0:lmax+1,k]*slm[0:lmax+1,0:lmax+1],axis=0)

    #--  Final signal recovery from fourier coefficients  ---------------!
    sig_out=np.zeros((phimax,thmax),dtype=np.float32)
    cosmp=np.cos(np.outer(phi,np.arange(lmax+1)))
    sinmp=np.sin(np.outer(phi,np.arange(lmax+1)))
    sig_out=np.dot(cosmp,d_cos)+np.dot(sinmp,d_sin)

    return sig_out

#@21
#--
def invert_matrix(cov,pseudo=False):
    import netCDF4 as netcdf4

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    itot = cov.shape[0]

    writenum = 0

    try:
        #usebuiltin = True
        usebuiltin = False

        if usebuiltin: # use python/numpy/scipy built in routines

            if pseudo:
                inv=linalg.pinv2(cov) # svd
            else:
                print 'Using vanilla scipy inversion'
                inv=linalg.inv(cov) 
        else:
            # use lu decomposition to solve for inverse matrix
            print 'inversion using lu decomposition'
            x=linalg.lu(cov,permute_l=True)
            pl=x[0] ; u=x[1]

            e = np.identity(itot)
            x=linalg.solve(pl,np.float64(e))
            inv=linalg.solve_triangular(u,x)

            if 1 == 2:
                outfile='lu_decomp.py.nc'
                w = netcdf4.Dataset(outfile, 'w', format='NETCDF4')
                w.createDimension('itot',np.int32(itot))
                
                wbbt  = w.createVariable('bbt','f8',('itot','itot'))
                wibbt  = w.createVariable('ibbt','f8',('itot','itot'))
                wbbt[:,:]  = cov
                wibbt[:,:]  = inv
                w.close()
                stop

        status=0
    except linalg.LinAlgError:
        print 'not invertible using scipy routines'
        status=-1
        
    return {'inverse':inv,'status':status}

