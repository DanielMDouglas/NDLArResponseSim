import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path

from utils import *
from induced_current import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--drift', type = str,
                        required = True,
                        help = "input file describing the drift electric field")
    parser.add_argument('-w', '--weighting', type = str,
                        required = True,
                        help = "input file describing the weighting field")

    parser.add_argument('-o', '--outputDir', type = str,
                        required = True,
                        help = "output directory for the resulting drift path data")
    
    args = parser.parse_args()

    print ("loading drift field")
    if '.vtu' in args.drift:
        driftField = field(args.drift, type = 'elmer')
    elif '.npy' in args.drift:
        driftField = field(args.drift, type = 'pickle')

    print ("loading weighting field")
    if '.vtu' in args.weighting:
        weightField = field(args.weighting, type = 'elmer')
    elif '.npy' in args.drift:
        weightField = field(args.weighting, type = 'pickle')

    
    z0 = 1.2
    
    nDigits = 5
    # xMin = 0.02217
    # xMax = 0.64293
    # deltaX = 0.04434
    # deltaX = 0.038
    deltaX = 0.1*pixelPitch

    # nDigits = 8

    nPix = 3.5
    pitch = pixelPitch
    # deltaX = 0.02217
    # deltaX = 0.011085

    xMin = deltaX/2
    xMax = nPix*pitch - deltaX/2
    
    nStepsX = round((xMax - xMin)/deltaX + 1)

    yMin = xMin
    deltaY = deltaX

    for x0raw in np.linspace(xMin, xMax, nStepsX):
        x0 = round(x0raw, nDigits)

        yMax = x0
        nStepsY = round((yMax - yMin)/deltaY + 1)

        for y0raw in np.linspace(yMin, yMax, nStepsY):
            y0 = round(y0raw, nDigits)
            outFile = os.path.join(args.outputDir, '_'.join([str(x0), str(y0)])+'.npy')

            print (x0, y0, outFile)

            if not os.path.exists(outFile):
            # if True:

                print ("drifting...")
                thisCharge = charge(np.array([x0, y0, z0], dtype = 'float64'))

                while not thisCharge.is_finished():
                    thisCharge.step(driftField, weightField)

                thisCharge.correct()
                
                thisCharge.print_to_file(outFile)

            else:
                print ("using previous calculation...")
