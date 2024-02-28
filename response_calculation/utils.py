import numpy as np

dt = 1.e-4 # us
tmax = 20 # us

pixelPitch = 0.4434 # cm
pixelHeight = 0.004 # cm
pixelWidth = 0.35 # cm
pixelRadius = 0.07 # cm

# pixelPitch = 0.38 # cm
# pixelHeight = 0.004 # cm
# pixelWidth = 0.3 # cm
# pixelRadius = 0.06 # cm

def mag(vect):
    return np.sqrt(np.sum(np.power(vect, 2)))

def norm(vect):
    return vect/mag(vect)

def get_perpendicular_vectors(vect):
    # TODO
    # perp1 = np.array([1, 0, 0])
    # perp2 = np.array([0, 1, 0])
    perp1 = np.random.randn(3)
    perp1 -= perp1.dot(vect)*vect
    perp1 /= np.linalg.norm(perp1)

    perp2 = np.cross(vect, perp1)
    
    return perp1, perp2

def drift_model(E, temperature = 89):
    magE = mag(E)
    
    if magE > 4.0:
        magE = 4.0

    if ( temperature < 87.0 ) or ( temperature > 94.0 ):
        print ( "DriftModel Warning!: Temperature value of ")
        print ( temperature )
        print ( "K is outside of range covered by drift velocity" )
        print ( " parameterization.  Returned value may not be correct" )

    tShift = -87.203 + temperature
    xFit = 0.0938163 - 0.0052563*tShift - 0.0001470*tShift*tShift
    uFit = 5.18406 + 0.01448*tShift - 0.003497*tShift*tShift - 0.000516*tShift*tShift*tShift
    
    # Icarus Parameter set
    # Use as default
    P1 = -0.04640 # K^-1
    P2 = 0.01712 # K^-1
    P3 = 1.88125 # (kV/cm)^-1
    P4 = 0.99408 # kV/cm
    P5 = 0.01172 # (kV/cm)^-P6
    P6 = 4.20214
    T0 = 105.749 # K

    # Walkowiak Parameter set
    P1W = -0.01481; # K^-1
    P2W = 0.0075; # K^-1
    P3W = 0.141; # (kV/cm)^-1
    P4W = 12.4; # kV/cm
    P5W = 1.627; # (kV/cm)^-P6
    P6W = 0.317;
    T0W = 90.371; # K

    # From Craig Thorne . . . currently not documented
    # smooth transition from linear at small fields to
    # icarus fit at most fields to Walkowiak at very high fields
    if magE < xFit:
        vd = magE*uFit
    elif magE < 0.619:
        vd = ((P1*(temperature-T0)+1)
	      *(P3*magE*np.log(1+P4/magE) + P5*np.power(magE, P6))
	      +P2*(temperature-T0))
    elif magE < 0.699:
        vd = 12.5*(magE - 0.619)*((P1W*(temperature-T0W)+1)
			          *(P3W*magE*np.log(1+P4W/magE) + P5W*np.power(magE, P6W))
			          +P2W*(temperature-T0W)) \
            + 12.5*(0.699 - magE)*((P1*(temperature-T0)+1)
			           *(P3*magE*np.log(1+P4/magE) + P5*np.power(magE, P6))
			           +P2*(temperature-T0))
    else:
        vd = ((P1W*(temperature-T0W)+1)
	      *(P3W*magE*np.log(1+P4W/magE) + P5W*np.power(magE, P6W))
	      +P2W*(temperature-T0W))

    vd /= 10

    direction = -norm(E)
    
    return direction*vd; # cm/us

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
vD = abs(drift_model(0.5))

zR = 1.2
# zD = 0.2
zD = 1.2

# dtR = 5.e-2
dtR = 1.e-1

