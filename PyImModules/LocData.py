# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:40 2019

@author: Lakshminarayanan Mohana Kumar
"""

import os
import time
from tqdm import tqdm
import copy

import numpy as np
from netCDF4 import Dataset
#import skimage

import PyImModules.PyDataReadWrite as pydtrw
from PyImModules.FileFolderUtils import list_files


class LocData:
    '''A class with data and associated functions to easily define, store and manipulate location data.'''
    
    __version__ = '1.0'

    
    def __init__(self, locdata, axesorder='xyz', axis_dir='row'):
        '''Initializes tomodata, 'data' should nbe numpy.ndarray.
        Recommended axesorder='xyz' and axis_dir='row' which means 
        each column vector is a point in 3D space of the format (x,y,z).'''
        self.axis_dir = axis_dir
        if type(locdata) == np.ndarray:
            if axis_dir == 'col':
                self.data = np.uint16(locdata.T)
            elif axis_dir == 'row':
                self.data = np.uint16(locdata)
            else:
                print("\nInput to axis_dir=", axis_dir, " not recognized.")
                raise ValueError
        else:
            print("\ndata is not of type numpy.ndarray.")
            raise TypeError

        self.dim  = locdata.shape
        
        axes ={}
        if type(axesorder) == str and len(axesorder) == 3:
            for indx, axis in enumerate(axesorder):
                axes[axis] = indx
            self.axesorder = axesorder
            self.axes = axes
        else:
            print("\nInvalid input for axesorder.")
            raise ValueError
    
    
    ### Values on a particular axis
    def axis_values(self, axis='z'):
        '''Return all values of x, y or z axes
        (specified axis) of locdata
        '''
        return self.data[self.axes[axis],:]
    
    def x_values(self):
        '''Return all values of x axis of locdata'''
        return self.axis_values(axis='x')
    
    def y_values(self):
        '''Return all values of y axis of locdata'''
        return self.axis_values(axis='y')
    
    def z_values(self):
        '''Return all values of z axis of locdata'''
        return self.axis_values(axis='z')
    
    
    
    ### Unique values on a particular axis
    def ax_uniqvals(self, axis='z'):
        '''Return unique values of x, y or z axis
        (specified axis) of locdata'''
        return np.unique(self.axis_values(axis))
    
    def x_uniqvals(self):
        '''Return unique values of x axis of locdata'''
        return np.unique(self.x_values())
        
    def y_uniqvals(self):
        '''Return unique values of y axis of locdata'''
        return np.unique(self.y_values())
    
    def z_uniqvals(self):
        '''Return unique values of z axis of locdata'''
        return np.unique(self.z_values())
    
    
    
    ### Extremum values on a particular axis
    def axis_min(self, axis='z'):
        '''Return min value of x, y or z axis (specified axis) of locdata'''
        return np.min(self.axis_values(axis))
    
    def x_min(self):
        '''Return min value of x axis of locdata'''
        return np.min(self.x_values())
    
    def y_min(self):
        '''Return min value of y axis of locdata'''
        return np.min(self.y_values())
    
    def z_min(self):
        '''Return min value of z axis of locdata'''
        return np.min(self.z_values())
    
    def axis_max(self, axis='z'):
        '''Return min value of x, y or z axis (specified axis) of locdata'''
        return np.max(self.axis_values(axis))
    
    def x_max(self):
        '''Return min value of x axis of locdata'''
        return np.max(self.x_values())
    
    def y_max(self):
        '''Return min value of y axis of locdata'''
        return np.max(self.y_values())
    
    def z_max(self):
        '''Return min value of z axis of locdata'''
        return np.max(self.z_values())
    
    
    ### Axes lengths
    def axes_lengths(self):
        '''Length of all axes: line(1D), area(2D) or vol(3D).'''
        axlens = {}
        
        for axis in self.axes.keys():
            axlens[axis] = self.axis_max(axis) - self.ax_min(axis)
        return axlens



    def centroid(self):
        '''Centroid of locdata of any dimensions: line(1D), area(2D) or vol(3D).'''
        centroid = {}
        for axis in self.axes.keys():
            centroid[axis] = (np.average(self.axis_values(axis)))
        return centroid




class VolLocData(LocData):
    '''Class for location data of a volume, i.e. in 3D space. Inherits from LocData'''
    
    def compute_volume(self):
        return self.data.shape[1]
    
    
    def extract_slicedata(self, axis_val, normal_axis='z'):
       if any(self.data[self.axes[normal_axis]] == axis_val):
           slice_ = copy.deepcopy(self)
           indices = np.array(np.where(slice_.data[self.axes[normal_axis]] == axis_val)).flatten()
           slice_.data = np.take(slice_.data, indices, axis=1)
           return slice_.data
       else:
           print("\nGiven value of axis_val not found anywhere in ", normal_axis,".")
           raise ValueError
    
    
    def extract_slice(self, axis_val, normal_axis='z'):
       if any(self.data[self.axes[normal_axis]] == axis_val):
           axesorder = ''.join([axis for axis in self.axes.keys() if axis is not normal_axis])
           axes_ = [self.axes[axis] for axis in axesorder]
           indices = np.array(np.where(self.data[self.axes[normal_axis]] == axis_val)).flatten()
           slicedata = np.array([[self.data[axes_[0],i], self.data[axes_[1],i]]  for i in indices]).T
           return SliceLocData(slicedata, axesorder)
       else:
           print("\nGiven value of axis_val not found anywhere in ", normal_axis,".")
           raise ValueError
           
    def slice_area(self, axis_val, normal_axis='z'):
       if any(self.data[self.axes[normal_axis]] == axis_val):
           row = copy.deepcopy(self.data[self.axes[normal_axis]])
           row = row + 1
           row[row < axis_val+1] = 0
           row[row > axis_val+1] = 0
           area = np.count_nonzero(row)
           
           return area
       else:
           print("\nGiven value of axis_val not found anywhere in ", normal_axis,".")
           raise ValueError
        


class SliceLocData(LocData):
    '''Class for location data of a slice, i.e. in 2D space. Inherits from LocData'''
    
    def __init__(self, locdata, axesorder='xy', axis_dir='row'):
        '''Initializes tomodata, 'data' should nbe numpy.ndarray.
        Recommended axesorder='xyz' and axis_dir='row' which means 
        each column vector is a point in 3D space of the format (x,y,z).'''
        self.axis_dir = axis_dir
        if type(locdata) == np.ndarray:
            if axis_dir == 'col':
                self.data = np.uint16(locdata.T)
            elif axis_dir == 'row':
                self.data = np.uint16(locdata)
            else:
                print("\nInput to axis_dir=", axis_dir, " not recognized.")
                raise ValueError
        else:
            print("\ndata is not of type numpy.ndarray.")
            raise TypeError

        if self.data.shape[0]==2:
            self.dim  = self.data.shape
        else:
            print("\nData is not 2-Dimensional.")
            raise ValueError
        
        axes ={}
        if type(axesorder) == str and len(axesorder) == 2:
            for indx, axis in enumerate(axesorder):
                axes[axis] = indx
            self.axesorder = axesorder
            self.axes = axes
        else:
            print("\nInvalid input for axesorder.")
            raise ValueError
    
    
    def compute_area(self):
        return self.data.shape[1]

#------------------------------------------------------------------------------
### TESTING
        
# =============================================================================
# if __name__ == "__main__":
#     filename1 = "C:\\image_proc_wkspace\\Integrated_PyImageAnalysis\\Data\\locdata\\locdata_fib.pickle"
#     locdata_fib = pydtrw.read_picklefile(filename1)
#     filename2 = "C:\\image_proc_wkspace\\Integrated_PyImageAnalysis\\Data\\locdata\\locdata_sfrc.pickle"
#     locdata_sfrc = pydtrw.read_picklefile(filename2)
#     
#     ax_val = locdata_fib.z_uniqvals()[100]
#     slice_fib = locdata_fib.extract_slice(ax_val)
#     slice_sfrc = locdata_sfrc.extract_slice(ax_val)
#     
#     print("\nArea Fraction: ", slice_fib.compute_area()/slice_sfrc.compute_area())
#     print("\nCentroid of SFRC: ", slice_sfrc.centroid())
#     
#     slice_fib = locdata_fib.extract_slicedata(ax_val)
#     print(slice_fib.data)
#     
# =============================================================================
