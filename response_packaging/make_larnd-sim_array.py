import numpy as np
import matplotlib.pyplot as plt
import argparse

from os import path

from utils import *
from induced_current import *
from plots import *

def reformat_current(data, tFactor = 1000, tMax = 100):
    t, i, icorrected = data.t, data.i, data.icorrected

    # if dt*np.sum(icorrected) < 0.5:
    #     icorrected -= np.sum(icorrected)/len(icorrected)

    padLen = tFactor - len(icorrected)%tFactor
    padded = np.concatenate((icorrected, np.zeros(padLen)))
    coarser = np.average(np.reshape(padded, (padded.shape[0]/tFactor, tFactor)), axis = 1)

    padLen = tMax - len(coarser)
    newCurrent = np.concatenate((coarser, np.zeros(padLen)))

    newdt = dt*tFactor
    print ("rebinning with dt = " + str(newdt))
    newT = newdt*np.arange(len(newCurrent))

    print (newCurrent[-1])

    return newT, newCurrent

plots = [{"class": iVt,
          "make": True}]

# reformatArgs = {'tFactor': 10, # dt = 5.e-2
#                 'tMax': 820}
# reformatArgs = {'tFactor': 100, # dt = 1.e-1
#                 'tMax': 83}
# reformatArgs = {'tFactor': 100, # dt = 1.e-1
#                 'tMax': 89}
reformatArgs = {'tFactor': 500,
                'tMax': 177}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', metavar = 'inputDir',
                        type = str, nargs = '+', 
                        help = "input file describing a charge drift path")
    parser.add_argument('-o', '--output', type = str,
                        required = True,
                        help = "destination for the resulting drift path data")

    args = parser.parse_args()

    inputDir = args.inputDir[0]

    print (inputDir)

    dataList = []

    nPix = 3.5

    # response = np.ndarray((15, 15, 820), dtype = float)
    # response = np.ndarray((15, 15, reformatArgs['tMax']), dtype = float)
    response = np.ndarray((int(10*nPix), int(10*nPix), reformatArgs['tMax']), dtype = float)
    
    z0 = 1.2
    
    nDigits = 5
    # deltaX = 0.04434
    # deltaX = 0.038
    deltaX = 0.1*pixelPitch
    
    pitch = pixelPitch

    xMin = deltaX/2
    xMax = nPix*pitch - deltaX/2
    
    nStepsX = round((xMax - xMin)/deltaX + 1)

    # print ("nstepsx", nStepsX)
    # input()
    
    xSpace = np.linspace(xMin, xMax, nStepsX)
    
    yMin = xMin
    deltaY = deltaX

    for i, x0raw in enumerate(xSpace):
        x0 = round(x0raw, nDigits)

        yMax = x0
        nStepsY = round((yMax - yMin)/deltaY + 1)

        ySpace = np.linspace(yMin, yMax, nStepsY)
        
        for j, y0raw in enumerate(ySpace):
            y0 = round(y0raw, nDigits)
            infile = path.join(inputDir, '_'.join([str(x0), str(y0)])+'.npy')

            print (x0, y0, infile)

            if not path.exists(infile):
                print ("not found!")
            else:
                print ("drift data found!")

                data = pathData(np.load(infile).T)

                # make_plots(plots, [data])

                coarseCurrent = reformat_current(pathData(np.load(infile).T),
                                                 **reformatArgs)
                data.t, data.icorrected = coarseCurrent
                data.dt = dt*reformatArgs['tFactor']
                print (data.dt, data.dt*np.sum(data.icorrected))
                dataList.append(data)

                response[i][j] = data.icorrected
                response[j][i] = data.icorrected
                # print (data.t)
                
                # make_plots(plots, [data])

                # plt.show()

    make_plots(plots, dataList)
    plt.show()

    np.save(args.output, response)
