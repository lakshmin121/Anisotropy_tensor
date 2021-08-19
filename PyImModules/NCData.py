# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:58:22 2019
@author: Lakshminarayanan Mohana Kumar

updated on 18th July 2021
"""

import os
import gc
import time
from tqdm import tqdm

import numpy as np
from netCDF4 import Dataset
#import skimage

from PyImModules.FileFolderUtils import list_files


__version__ = '4.5'

ncdtypes = ('f4', 'f8', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u8')
npdtypes = (np.float32, np.float32, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint64)
nc2np_dtype_map = dict(zip(ncdtypes, npdtypes))
netcdf4_types = ('i8', 'u1', 'u2', 'u8')  # datatypes only supported by netcdf4


def nc2np_dtype(nctype):
    if nctype in nc2np_dtype_map.keys():
        return nc2np_dtype_map[nctype]
    else:
        raise ValueError("invalid nc datatype: {}".format(nctype))


def np2nc_dtype(nptype):
    if nptype in nc2np_dtype_map.values():
        for k, v in nc2np_dtype_map.items():
            if nptype == v:
                return k
    else:
        raise ValueError("invalid np datatype: {}".format(nptype))


class NCData:
    '''Class to handle image data in the '.nc' format. Mainly useful to read and wrtie '.nc' files.
    Easy to convert from NCData format to TomoData.'''
    def __init__(self, ncdata, ncdim=None, var='tomo', vartype=None, ncformat='NETCDF3_CLASSIC'):
        if vartype is None:
            dtype = ncdata.dtype
            vartype = np2nc_dtype(dtype)
        else:
            dtype = nc2np_dtype(vartype)
        if ncdim is None:
            ncdim = ncdata.shape
        self.ncdim  = ncdim
        self.vartype = vartype
        self.ncdata = ncdata.astype(dtype)
        self.var = var
        if vartype in netcdf4_types:
            # ncformat = 'NETCDF4_CLASSIC'
            ncformat = 'NETCDF4'
        self.ncformat = ncformat

    def __repr__(self):
        string = ''
        for k, v in vars(self).items():
            if not k=='ncdata':
                string = string + '\n' + k + ': ' + str(v)
        return string

    def read_ncfile(filename, var='tomo'):
        dataset0 = Dataset(filename)
        ncdim = dataset0.variables[var][:,:,:].shape
        ncformat = str(dataset0.data_model)
        ncdata = np.array(dataset0.variables[var][:,:,:])
        ncobj = NCData(ncdata, ncdim, ncformat)
        dataset0.close()
        
        return ncobj

    def read_ncfiles(filepath, var='tomo'):
        '''Read '.nc' files and save data as list of NCData objects'''
        mstart = time.time()
        ncfiles = list_files(filepath, file_extn='.nc')
        print(len(ncfiles), " '.nc' files detected.\n")
        
        ncobjects = []
        with tqdm(ncfiles, desc="Reading .nc files", ncols=100) as pbar:
            for ncfile in ncfiles:
                if ncfile:
                    filename = os.path.join(filepath, ncfile)
                else:
                    filename = filepath
                    ncfiles[ncfiles==ncfile] = filename #updating '' to filename
                ncobj = NCData.read_ncfile(filename)
                ncobjects.append(ncobj)
                pbar.update()
        
        if len(ncobjects):
            return ncobjects, ncfiles
        
        mstop = time.time()
        print("\n time taken: ", mstop-mstart)

    def write_ncfile(self, filename):
        zdim, ydim, xdim = self.ncdim
    
        tosave_ncfile = Dataset(filename, 'w', format=self.ncformat)
        tosave_ncfile.createDimension('tomo_zdim', zdim)
        tosave_ncfile.createDimension('tomo_xdim', xdim)
        tosave_ncfile.createDimension('tomo_ydim', ydim)
        try:
            tomo = tosave_ncfile.createVariable(self.var, self.vartype, ('tomo_zdim', 'tomo_ydim', 'tomo_xdim'))
            #temp1 = np.uint16(tomodata_concr[fn_save])
            #tomo[:,:,:] = temp1[0:flen[fn_save],0:ydim,0:xdim]
            tomo[:,:,:] = self.ncdata #temp1[0:flen[fn_save],0:ydim,0:xdim]
        except RuntimeError:
            if os.path.exists(filename):
                tosave_ncfile.close()
                os.remove(filename)
                tosave_ncfile = Dataset(filename, 'w', format=self.ncformat)
                tosave_ncfile.createDimension('tomo_zdim', zdim)
                tosave_ncfile.createDimension('tomo_xdim', xdim)
                tosave_ncfile.createDimension('tomo_ydim', ydim)

                tomo = tosave_ncfile.createVariable(self.var, self.vartype, ('tomo_zdim', 'tomo_ydim', 'tomo_xdim'))
                # temp1 = np.uint16(tomodata_concr[fn_save])
                # tomo[:,:,:] = temp1[0:flen[fn_save],0:ydim,0:xdim]
                tomo[:, :, :] = self.ncdata  # temp1[0:flen[fn_save],0:ydim,0:xdim]
        tosave_ncfile.close()
        gc.collect()

    def combine_ncobjects(ncobjects):
        data = []
        for ncobject in ncobjects:
            if not len(data):
                data = ncobject.ncdata
            else:
                data = np.concatenate((data, ncobject.ncdata))
                
        return NCData(data)

    @classmethod
    def TomoData2NCData(cls, tomobj, var='tomo', ncformat='NETCDF3_CLASSIC'):
        return cls(tomobj.data, var)