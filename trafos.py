import numpy as np
r2d = 180./np.pi
d2r = np.pi/180.

# Basic definitions from the introduction to the Hipparcos catalog
# ================================================================
# Directions to the Galactoc north pole
RA_ngp = 192.85948 # degree
DE_ngp = +27.12825 # degree
# Origin of the Galactic longitude
l_Omega = 32.93192 # degree

# Transformation matrix between equatorial and galactic coordinates:
# [x_G  y_G  z_G] = [x  y  z] . A_G
A_G = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                [-0.8734370902, -0.4448296300, -0.1980763734],
                [-0.4838350155, +0.7469822445, +0.4559837762]])
# inverse matrix
iA_G = np.linalg.inv(A_G)
# Conversion between 1 A.U./yr in km/s
A_v = 4.74047

# ================================================================
# ================================================================
# ================================================================
# ================================================================

def eq2galCoords(RA,DE):
    '''
    Computes Galactic angular coordinates via the method descibed
    in the Introduction to the Hipparcos catalog.

    Input:

    RA   - right ascension in degree (1d-array or single)
    DE   - declination in degree (1d-array or single)

    Output:

    l    - Galactic longitude in degree
    b    - Galactic latitude in degree
    
    History:
    Written by Til Piffl                          May 2013
    '''

    ra = np.array(RA).reshape(-1)*d2r
    de = np.array(DE).reshape(-1)*d2r
    assert len(ra) == len(de)

    sde,cde = np.sin(de),np.cos(de)
    sra,cra = np.sin(ra),np.cos(ra)
    
    aux0 = A_G[0,0]*cde*cra + A_G[1,0]*cde*sra + A_G[2,0]*sde
    aux1 = A_G[0,1]*cde*cra + A_G[1,1]*cde*sra + A_G[2,1]*sde
    aux2 = A_G[0,2]*cde*cra + A_G[1,2]*cde*sra + A_G[2,2]*sde

    b = np.arcsin(aux2)*r2d
    l = np.arctan2(aux1,aux0)*r2d
    l[l<0] += 360.

    if len(l) == 1:
        return l[0],b[0]
    else:
        return l,b

# ================================================================

def gal2eqCoords(L,B):
    '''
    Computes Equatorial angular coordinates via the method descibed
    in the Introduction to the Hipparcos catalog.

    Input:

    L    - Galactic longitude in degree (1d-array or single)
    B    - Galactic latitude in degree (1d-array or single)

    Output:

    RA   - right ascension in degree
    DE   - declination in degree
    
    History:
    Written by Til Piffl                          May 2013
    '''

    l = np.array(L).reshape(-1)*d2r
    b = np.array(B).reshape(-1)*d2r
    assert len(l) == len(b)
    sb,cb = np.sin(b),np.cos(b)
    sl,cl = np.sin(l),np.cos(l)
    
    aux0 = iA_G[0,0]*cb*cl + iA_G[1,0]*cb*sl + iA_G[2,0]*sb
    aux1 = iA_G[0,1]*cb*cl + iA_G[1,1]*cb*sl + iA_G[2,1]*sb
    aux2 = iA_G[0,2]*cb*cl + iA_G[1,2]*cb*sl + iA_G[2,2]*sb

    de = np.arcsin(aux2)*r2d
    ra = np.arctan2(aux1,aux0)*r2d
    ra[ra<0] += 360.

    if len(ra) == 1:
        return ra[0],de[0]
    else:
        return ra,de

# ================================================================

def galAngular2Cartesian(L,B,Distance,posSun=[-8.0,0.0,0.015]):
    '''
    Computes Equatorial angular coordinates via the method descibed
    in the Introduction to the Hipparcos catalog.

    Input:

    L        - Galactic longitude in degree (1d-array or single)
    B        - Galactic latitude in degree (1d-array or single)
    Distance - in kpc (1d-array or single)

    optional parameter:

    posSun   - Position of the Sun in Galactocentric cartesian
               coordinates in kpc (default values [-8,0,0.015])


    Output:

    x - x component of stars position vector in kpc, with the
        x-axis pointing towards the Galactic center 
    y - y component of stars position vector in kpc, with the
        y-axis pointing in the direction of the Galactic rotation
    z - z component of stars position vector in kpc, with the
        x-axis pointing towards the Galactic north pole
    
    History:
    Written by Til Piffl                          May 2013
    '''

    l = np.array(L).reshape(-1)*d2r
    b = np.array(B).reshape(-1)*d2r
    assert len(l) == len(b)
    d = np.array(Distance).reshape(-1)
    assert len(l) == len(d)
    x = d * np.cos(l) * np.cos(b) + posSun[0]
    y = d * np.sin(l) * np.cos(b) + posSun[1]
    z = d             * np.sin(b) + posSun[2]
    if len(x) == 1:
        return x[0],y[0],z[0]
    else:
        return x,y,z

# ================================================================

def getGalVelocity(RA,DE,Distance,RV,pmRA,pmDE,\
                       UVWsun=[11.1,12.24,7.25],vLSR=220.):
    '''
    Computes Galactocentric space velocities via the method descibed
    in the Introduction to the Hipparcos catalog. 
    See also Johnson & Soderblom (1987) for a detailed explanation.

    Input:

    RA       - right ascension in degree (1d-array or single)
    DE       - declination in degree (1d-array or single)
    Distance - in kpc (1d-array or single)
    RV       - heliocentric radial velocity (1d-array or single)
    pmRA     - mu_alpha* = mu_alpha x cos(DE): de-projected proper motion
               in right ascension in mas/yr (1d-array or single)
    pmDE     - mu_delta: proper motion in declination 
               in mas/yr (1d-array or single)

    optional parameters:

    UVWsun   - peculiar velocity of the Sun in km/s (default values are
               [11.1,12.24,7.25] from Schoenrich, Binney & Dehnen (2010)
    vLSR     - Local standard of rest in km/s (default values: 220.)


    Output:

    vx       - Velocity component towards the Galactic center in km/s
    vy       - Velocity component in direction of the Galactic rotation
               ((l,b) = (90,0) degree) in km/s
    vz       - Velocity component towards the Galactic north pole
               (b = 90 degree) in km/s
    
    History:
    Written by Til Piffl                          May 2013
    '''

    ra = np.array(RA).reshape(-1)*d2r
    de = np.array(DE).reshape(-1)*d2r
    assert len(ra) == len(de)
    d = np.array(Distance).reshape(-1)
    assert len(ra) == len(d)
    mu_ra = np.array(pmRA).reshape(-1)
    assert len(ra) == len(mu_ra)
    mu_de = np.array(pmDE).reshape(-1)
    assert len(ra) == len(mu_de)
    rv = np.array(RV).reshape(-1)
    assert len(ra) == len(rv)
    
    sra,cra = np.sin(ra),np.cos(ra)
    sde,cde = np.sin(de),np.cos(de)

    aux0 = mu_ra*A_v*d
    aux1 = mu_de*A_v*d
    aux2 = rv

    # Space velocities in Equatorial cartesian coordinate system
    vx_E = -sra*aux0 -sde*cra*aux1 + cde*cra*aux2
    vy_E =  cra*aux0 -sde*sra*aux1 + cde*sra*aux2
    vz_E =            cde    *aux1 + sde    *aux2

    vx = A_G[0,0]*vx_E + A_G[1,0]*vy_E + A_G[2,0]*vz_E + UVWsun[0]
    vy = A_G[0,1]*vx_E + A_G[1,1]*vy_E + A_G[2,1]*vz_E + UVWsun[1] + vLSR
    vz = A_G[0,2]*vx_E + A_G[1,2]*vy_E + A_G[2,2]*vz_E + UVWsun[2]

    if len(vx) == 1:
        return vx[0],vy[0],vz[0]
    else:
        return vx,vy,vz

# ================================================================

def getProperMotions(RA,DE,Distance,U,V,W):
    '''
    Computes the heliocentric line-of-sight velocity and proper motions
    in the Equatorial coordinate system via the inversion of the  method
    descibed in the Introduction to the Hipparcos catalog. 
    See also Johnson & Soderblom (1987) for a detailed explanation.

    Input:

    RA       - right ascension in degree (1d-array or single)
    DE       - declination in degree (1d-array or single)
    Distance - in kpc (1d-array or single)
    U        - Velocity component towards the Galactic center in km/s
               (1d-array or single)
    V        - Velocity component in direction of the Galactic rotation
               ((l,b) = (90,0) degree) in km/s (1d-array or single)
    W       - Velocity component towards the Galactic north pole
               (b = 90 degree) in km/s (1d-array or single)
    The three velocity components U,V,W must be in the solar rest-frame!

    Output:

    RV       - heliocentric radial velocity
    pmRA     - mu_alpha* = mu_alpha x cos(DE): de-projected proper motion
               in right ascension in mas/yr
    pmDE     - mu_delta: proper motion in declination 
               in mas/yr
    
    History:
    Written by Til Piffl                          May 2013
    '''

    ra = np.array(RA).reshape(-1)*d2r
    de = np.array(DE).reshape(-1)*d2r
    assert len(ra) == len(de)
    d = np.array(Distance).reshape(-1)
    assert len(ra) == len(d)
    u = np.array(U).reshape(-1)
    assert len(ra) == len(u)
    v = np.array(V).reshape(-1)
    assert len(ra) == len(v)
    w = np.array(W).reshape(-1)
    assert len(ra) == len(w)
    
    sra,cra = np.sin(ra),np.cos(ra)
    sde,cde = np.sin(de),np.cos(de)

    u_E = iA_G[0,0]*u + iA_G[1,0]*v + iA_G[2,0]*w
    v_E = iA_G[0,1]*u + iA_G[1,1]*v + iA_G[2,1]*w
    w_E = iA_G[0,2]*u + iA_G[1,2]*v + iA_G[2,2]*w
    
    aux0 = -sra    *u_E + cra    *v_E
    aux1 = -sde*cra*u_E - sde*sra*v_E + cde*w_E
    rv   =  cde*cra*u_E + cde*sra*v_E + sde*w_E
    
    pmRA = aux0 / A_v / d
    pmDE = aux1 / A_v / d
    
    if len(rv) == 1:
        return rv[0],pmRA[0],pmDE[0]
    else:
        return rv,pmRA,pmDE

# ================================================================

def AngularDistance(phi1,theta1,phi2,theta2,units='deg'):
    '''
    Computes the angular distance between vectors 1 and 2 in a
    spherical coordinate system (r,phi,theta).

    Input:

    phi1,theta1    - postion angles of vector 1 (1d-arrays or floats)
    phi2,theta2    - postion angles of vector 2 (1d-arrays or floats)
    units          - if ='deg' the input values are assumed to be in
                     degree and the output is also given in degree.
                     Otherwise radians are assumed and returned.

    Output:

    Angular distance in degree or radians.

    History:
    Written by Til Piffl                              May 2013
    '''

    if units == 'deg':
        RA1 = ra1*d2r
        DE1 = de1*d2r
        RA2 = ra2*d2r
        DE2 = de2*d2r
    else:
        RA1 = ra1
        DE1 = de1
        RA2 = ra2
        DE2 = de2
    dRA = abs(RA1 - RA2)
    sdRA,cdRA = np.sin(dRA),np.cos(dRA)
    sde1,cde1 = np.sin(DE1),np.cos(DE1)
    sde2,cde2 = np.sin(DE2),np.cos(DE2)
    dist = arctan2(sqrt((cde1*sdRA)**2 + (cde2*sde1 - sde2*cde1*cdRA)**2),
                       sde2*sde1 + cde2*cde1*cdRA)
    if units == 'deg':
        dist *= r2d
    return abs(dist)

# ================================================================

