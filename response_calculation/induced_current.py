import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator

from utils import *

class charge:
    def __init__(self, initPos):
        self.initPos = initPos
        self.pos = initPos

        # the drift field is defined for +/- pixelPitch/2
        # translate the path to that domain
        self.driftOffset = np.zeros(3)

        x = initPos[0]
        while x > pixelPitch/2:
            x -= pixelPitch
            self.driftOffset[0] -= pixelPitch
        while x < -pixelPitch/2:
            x += pixelPitch
            self.driftOffset[0] += pixelPitch

        y = initPos[1]
        while y > pixelPitch/2:
            y -= pixelPitch
            self.driftOffset[1] -= pixelPitch
        while y < -pixelPitch/2:
            y += pixelPitch
            self.driftOffset[1] += pixelPitch

        # the response field is defined for quadrant I
        # rotate the path to that domain
        self.responseRotation = np.eye(3)

        x = initPos[0]
        if x < 0:
            self.responseRotation[0] *= -1 

        y = initPos[1]
        if y < 0:
            self.responseRotation[1] *= -1 

        # the calibration path is to determine
        # the amount of integration error based on
        # the known integral of the path if it were collected (1)
        
        # rotate the path to that domain
        self.calibResponseRotation = np.eye(3)

        x = initPos[0] + self.driftOffset[0]
        if x < 0:
            self.calibResponseRotation[0] *= -1 

        y = initPos[1] + self.driftOffset[1]
        if y < 0:
            self.calibResponseRotation[1] *= -1 

            
        # print ("offset:", self.driftOffset)

        self.t = 0

        self.tHist = []
        self.posHist = []
        self.velHist = []
        self.curHist = []
        self.CorrectedCurHist = []
        self.calibCurHist = []
        
        self.Ehist = []
        self.Whist = []
        
    def step(self, driftField, weightField):
        
        driftPos = self.pos + self.driftOffset
        responsePos = np.dot(self.responseRotation, self.pos)
        calibResponsePos = np.dot(self.calibResponseRotation, driftPos)

        # print ("t:", self.t)
        # print ("pos:", self.pos)
        # print ("drift pos:", driftPos)
        # print ("response pos:", responsePos)
        # print ("calib response pos:", calibResponsePos)
        # print ("is finished:", self.is_finished())

        # z beyond which the weighting field is not defined
        if self.pos[2] >= zR:
            localE = np.array([0, 0, 0.5])
            localW = np.array([0, 0, 0])
            calibW = np.array([0, 0, 0])

        # z beyond which the drift field is not defined
        elif self.pos[2] >= zD:
            localE = np.array([0, 0, 0.5])
            localW = np.dot(self.responseRotation,
                            weightField.value(responsePos))
            calibW = np.dot(self.calibResponseRotation,
                            weightField.value(calibResponsePos))
            
        else:
            localE = driftField.value(driftPos)
            localW = np.dot(self.responseRotation,
                            weightField.value(responsePos))
            calibW = np.dot(self.calibResponseRotation,
                            weightField.value(calibResponsePos))

        
        # print ("localE:", localE)
        v = drift_model(localE)
        # print ("vel:", v)
        i = -np.dot(v, localW)
        # print ("cur:", i)
        iCalib = -np.dot(v, calibW)
 
        self.tHist.append(self.t)
        self.posHist.append(np.copy(self.pos))
        self.velHist.append(v)
        self.curHist.append(i)
        self.calibCurHist.append(iCalib)
        self.Ehist.append(localE)
        self.Whist.append(localW)
        
        self.pos += v*dt
        self.t += dt

    def is_finished(self):
        if self.is_in_backplane():
            return True
        elif self.is_in_pixel():
            return True
        elif self.t > tmax:
            return True
        return False

    def correct(self):
        if len(self.calibCurHist):
            f = 1./(dt*np.sum(self.calibCurHist))
            print (np.sum(self.calibCurHist))
            print (np.sum(self.curHist))
            print ("Final correction factor:", f)
            f = 1
            self.calibCurHist = np.array([f*ci for ci in self.curHist])
            print (np.sum(self.calibCurHist))

            # if dt*np.sum(self.calibCurHist) < 0.5:
            #     self.calibCurHist -= np.sum(self.calibCurHist)/len(self.calibCurHist)

            print ("Corrected Q tot:", dt*np.sum(self.calibCurHist))

    def is_in_pixel(self):
        W = pixelWidth # pixel width
        R = pixelRadius # corner radius

        x, y, z = self.pos + self.driftOffset
        
        is_below_top = z <= pixelHeight

        if not is_below_top:
            return False

        else:
            is_in_A = (abs(y) <= W/2) and (abs(x) <= W/2 - R)
            is_in_B = (abs(x) <= W/2) and (abs(y) <= W/2 - R)
            is_in_C = (np.power(abs(x) - (W/2 - R), 2) + np.power(abs(y) - (W/2 - R), 2) <= np.power(R, 2))

            is_in_xy = is_in_A or is_in_B or is_in_C
        
            return is_in_xy

    def is_in_backplane(self):
        return self.pos[2] <= 0
            
    def print_to_file(self, outfile):
        outArray = np.concatenate([np.expand_dims(self.tHist, 1),
                                   self.posHist,
                                   self.velHist,
                                   np.expand_dims(self.curHist, 1),
                                   np.expand_dims(self.calibCurHist, 1),
                                   self.Ehist,
                                   self.Whist],
                                  axis = 1)
        # print (outArray.shape)
        np.save(outfile, outArray)

class field:
    def __init__(self, infile, type = 'simulia'):
        self.infile = infile
        if type == 'elmer':
            print ("loading field from vtu")
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy

            # Read the source file.
            print ("initializing reader")
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(infile)
            reader.Update()  # Needed because of GetScalarRange
            output = reader.GetOutput()

            print ("initializing locator")
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(output)
            locator.BuildLocator()

            print ("loading field array")
            eField = vtk_to_numpy(output.GetPointData().GetArray("electric field"))
            print ("loading node array")
            nodes = vtk_to_numpy(output.GetPoints().GetData())

            # find the cell that pos lies within
            def interp(pos):
                cellID = locator.FindCell(pos)
                cell = output.GetCell(cellID)

                # get the four primary points of the tetrahedron
                pointIDs = [cell.GetPointId(i) for i in range(4)]

                nodeLocs = [nodes[i] for i in pointIDs]
                nodeVals = [eField[i] for i in pointIDs]
                
                # build the matrix that relates 3D points to barycentric coordinates
                rot = np.matrix([np.concatenate([[1], nodeLoc]) for nodeLoc in nodeLocs]).T
                interpVec = np.concatenate([[1], pos])
                baryCoords = np.array(np.dot(rot.I, interpVec)).flatten()

                # do the interpolation
                result = np.dot(baryCoords, nodeVals)

                return result

            self.value = interp
                
            # points = cell.GetPoints()
            # nodeLocs = vtk_to_numpy(points.GetData())[:4]  

            
            # print ("loading field array")
            # eField = vtk_to_numpy(output.GetPointData().GetArray("electric field"))
            # print ("loading node array")
            # nodes = vtk_to_numpy(output.GetPoints().GetData())

            # probe = vtk.vtkProbeFilter()
            # probe.SetSource(output)
            
            # def interpEx(loc):
            #     ps = vtk.vtkPointSource
            #     probe.SetInputConnections()
                
            # print ("constructing interpolator in x")
            # # self.interpEx = NearestNDInterpolator(nodes, eField[:,...])
            # self.interpEx = LinearNDInterpolator(nodes, eField[:,...])
            # print ("constructing interpolator in y")
            # # self.interpEy = NearestNDInterpolator(nodes, eField[:,1])
            # self.interpEy = LinearNDInterpolator(nodes, eField[:,1])
            # print ("constructing interpolator in z")
            # # self.interpEz = NearestNDInterpolator(nodes, eField[:,2])
            # self.interpEz = LinearNDInterpolator(nodes, eField[:,2])
            
        if type == 'simulia':
            x, y, z, Ex, Ey, Ez = np.loadtxt(infile, skiprows = 2).T
            
            x = np.unique(x)/10 # cm
            y = np.unique(y)/10 # cm
            z = np.unique(z)/10 # cm
            
            Ex = np.reshape(Ex,
                            (len(x), len(y), len(z)),
                            order = 'F')/1.e5 # kV/cm
            Ey = np.reshape(Ey,
                            (len(x), len(y), len(z)),
                            order = 'F')/1.e5 # kV/cm
            Ez = np.reshape(Ez,
                            (len(x), len(y), len(z)),
                            order = 'F')/1.e5 # kV/cm

            print ("constructing interpolator in x")
            self.interpEx = RegularGridInterpolator((x, y, z), Ex)
            print ("constructing interpolator in y")
            self.interpEy = RegularGridInterpolator((x, y, z), Ey)
            print ("constructing interpolator in z")
            self.interpEz = RegularGridInterpolator((x, y, z), Ez)

            def interp(pos):
                return np.concatenate([self.interpEx(pos.T),
                                       self.interpEy(pos.T),
                                       self.interpEz(pos.T)])

            self.value = interp

        elif type == 'solver':
            x, y, z, pot = np.loadtxt(infile, delimiter = ',').T

            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)

            pot = np.reshape(pot,
                             (len(x), len(y), len(z)),
                             order = 'C') # kV

            Ex = -np.diff(pot, axis = 0)/np.expand_dims(np.expand_dims(np.diff(x), 1), 2)
            Ey = -np.diff(pot, axis = 1)/np.expand_dims(np.expand_dims(np.diff(y), 0), 2)
            Ez = -np.diff(pot, axis = 2)/np.expand_dims(np.expand_dims(np.diff(z), 0), 1)

            xMids = 0.5*(x[1:] + x[:-1])
            yMids = 0.5*(y[1:] + y[:-1])
            zMids = 0.5*(z[1:] + z[:-1])

            print ("constructing interpolator in x")
            self.interpEx = RegularGridInterpolator((xMids, y, z), Ex)
            print ("constructing interpolator in y")
            self.interpEy = RegularGridInterpolator((x, yMids, z), Ey)
            print ("constructing interpolator in z")
            self.interpEz = RegularGridInterpolator((x, y, zMids), Ez)

            def interp(pos):
                return np.concatenate([self.interpEx(pos.T),
                                       self.interpEy(pos.T),
                                       self.interpEz(pos.T)])

            self.value = interp

        elif type == 'pickle':
            print ("loading pickled field at " + infile)
            self = np.load(infile)[0]

    def pickle(self, outfileName):
        print ("pickling field to ", outfileName)
        np.save(outfileName, np.array([self]))
        
    # def value(self, pos):
    #     return np.concatenate([self.interpEx(pos.T),
    #                            self.interpEy(pos.T),
    #                            self.interpEz(pos.T)])
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', '--xinit', type = float,
                        default = 0,
                        help = "initial x position of drifting charge")
    parser.add_argument('-y', '--yinit', type = float,
                        default = 0,
                        help = "initial y position of drifting charge")
    parser.add_argument('-z', '--zinit', type = float,
                        default = 1.2,
                        help = "initial z position of drifting charge")

    parser.add_argument('-d', '--drift', type = str,
                        required = True,
                        help = "input file describing the drift electric field")

    parser.add_argument('-f', '--field', action = 'store_true',
                        help = "use this flag if the input is from the field_solver framework")

    parser.add_argument('-w', '--weighting', type = str,
                        required = True,
                        help = "input file describing the weighting field")

    parser.add_argument('-o', '--output', type = str,
                        required = True,
                        help = "destination for the resulting drift path data")

    
    args = parser.parse_args()

    x0, y0, z0 = args.xinit, args.yinit, args.zinit

    print ("loading drift field")
    # if args.field:
    #     driftField = field(args.drift, type = 'solver')
    # else:
    driftField = field(args.drift, type = 'elmer')

    print ("loading weighting field")
    # if args.field:
    #     weightField = field(args.weighting, type = 'solver')
    # else:
    weightField = field(args.weighting, type = 'elmer')

    # print ("loading weighting field")
    # weightField = field(args.weighting)

    thisCharge = charge(np.array([x0, y0, z0], dtype = 'float64'))
    
    while not thisCharge.is_finished():
        thisCharge.step(driftField, weightField)

    thisCharge.correct()

    thisCharge.print_to_file(args.output)
