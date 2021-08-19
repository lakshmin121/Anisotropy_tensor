# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:44:30 2019
Last edited on Thursday Jul 11 12:45
All utility functions to read and write data.
@author: Lakshminarayanan Mohana Kumar
"""

import os
#import sys
import time
import gc
from tqdm import tqdm

import numpy as np
import pickle

import skimage
import imageio
from netCDF4 import Dataset
#import tifffile as tiff
#from skimage.external import tifffile as tiff

from PyImModules.NCData import NCData
from PyImModules.FileFolderUtils import list_files, replace_extn
#from FileFolderUtils import is_file_or_folder


__version__ = '4.1'

# =============================================================================
#  Changes:
# __version__ = '4.0':
#     read_multiple_files() and write_multiple_files() included.

# =============================================================================
#------------------------------------------------------------------------------
cwd=os.getcwd()
filetype_to_readfunc = {'.nc'       :   'read_ncfile',
                        '.pickle'   :   'read_picklefile',
                        '.npy'      :   'read_npyfile',
                        'image'     :   'read_imagefile'
                        }

filetype_to_writefunc = {'.nc'      :   'write_ncfile',
                        '.pickle'   :   'write_picklefile',
                        '.npy'      :   'write_npyfile',
                        'image'     :   'write_imagefiles'
                        }



#------------------------------------------------------------------------------
### READ FUNCTIONS

def read_ncfile(filename, var='tomo'):
    dataset0 = Dataset(filename)
    ncdim = dataset0.variables[var][:,:,:].shape
    ncformat = str(dataset0.data_model)
    ncdata = np.uint16(np.array(dataset0.variables[var][:,:,:]))
    ncobj = NCData(ncdata, ncdim, ncformat)
    
    return ncobj



def read_picklefile(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()

    return data



# Read npy file
def read_npyfile(filename):
    '''Reads .npy file.'''
    return np.load(filename)



#Read image file
def read_imagefile(filename, rgb2gray=True):
    imdata = imageio.imread(filename)
    imdata = np.array(imdata)
    dataformat = imdata.dtype
    
    if len(imdata.shape) > 2:
        if len(imdata.shape) == 3 and rgb2gray:
            #print("RGB image detected. Converting to grayscale...")
            # detect the data format - uint8, uint16
            try:
                int_order = int(str(dataformat)[4:])
            except ValueError:
                print(dataformat, " dtype not acceptable, uint data expected.")
            # convert rgb2grayscale with original format
            imdata = skimage.color.rgb2gray(imdata)*(2**int_order-1)
            imdata = imdata.astype(dataformat)
        else:
            print("Image not readable: ", len(imdata.shape), " dimensions encountered.")
    
    return imdata
  


def read_imagefiles(filepath, file_extn=None, rgb2gray=True, combine_data=False):
    '''Reads single or multiple JPEG, PNG or TIFF files with if filepath is file or folder
    respectively. Returns dataset (list of data from each file), and list of
    filenames.'''
    mstart = time.time()
    print("\n Reading image files...")
    
    imgfiles = []
    file_extn_list = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    if file_extn:
        imgfiles = list_files(filepath, file_extn)
        print(len(imgfiles), " ", file_extn, " files detected.\n")
    else:
        for extn in file_extn_list:
            files = list_files(filepath, extn)
            print(len(files), " ", extn, " files detected.\n")
            imgfiles += files

    dataset = []
    with tqdm(imgfiles, desc="Reading image files", ncols=100) as pbar:
        for imgfile in imgfiles:
            if imgfile:
                filename = os.path.join(filepath, imgfile)
            else:
                filename = filepath
                imgfiles[imgfiles==imgfile] = filename #updating '' to filename
            data = read_imagefile(filename, rgb2gray)
            dataset.append(data)
            pbar.update()
    
    if len(dataset):
        if combine_data:
            dataset = np.array(dataset)
#            print(type(dataset))
#            print(dataset.shape)
        return dataset, imgfiles
    
    mstop = time.time()
    print("\n time taken: ", mstop-mstart)



def read_multiple_files(filepath, file_extn, **kwargs): # Tested OK
    '''General function to read multiple files of a given type. Read function
    for single file of appropriate type will be chosen.'''
    mstart = time.time()
    
    files = list_files(filepath, file_extn)
    print(len(files), " %s files detected.\n" % file_extn)
    
    read_func = globals()[filetype_to_readfunc[file_extn]]
    
    fobjects = []
    with tqdm(files, desc="Reading %s files" % file_extn, ncols=100) as pbar:
        for file in files:
            if file:
                # this means nultiple nc files
                filename = os.path.join(filepath, file)
            else:
                # this means single nc file
                filename = filepath
                files[files==file] = filename #updating '' to filename
            
            fobj = read_func(filename, **kwargs)
            fobjects.append(fobj)
            pbar.update()
    
    if len(fobjects):
        mstop = time.time()
        print("\n time taken: ", mstop-mstart)
        return fobjects, files



def read_ncfiles(filepath, var='tomo'): # Tested OK
    '''Read '.nc' files and save data as list of NCData objects'''
    
    print(filepath)
    return read_multiple_files(filepath, file_extn='.nc', var=var)
    


def read_picklefiles(filepath): # Tested OK
    '''Reads single or multiple '.pickle' files with if filepath is file or folder
    respectively. Returns dataset (list of data from each file), and list of
    filenames.'''
    
    if not os.path.isdir(filepath) or os.path.isfile(filepath):
        if not os.path.splitext(filepath)[-1] == '.pickle':
            filepath = filepath + '.pickle'
    
    return read_multiple_files(filepath, file_extn='.pickle')



# Read npy files
def read_npyfiles(filepath):
    '''Reads single or multiple '.npy' files with if filepath is file or folder
    respectively. Returns dataset (list of data from each file), and list of
    filenames.'''
    
    return read_multiple_files(filepath, file_extn='.npy')



#------------------------------------------------------------------------------
### WRITE FUNCTIONS
    

def write_ncfile(ncobj, filename):        
    
    zdim, ydim, xdim = ncobj.ncdim

    tosave_ncfile = Dataset(filename,'w',format=ncobj.ncformat)
    tosave_ncfile.createDimension('tomo_zdim', zdim)
    tosave_ncfile.createDimension('tomo_xdim', xdim)
    tosave_ncfile.createDimension('tomo_ydim', ydim)
    tomo = tosave_ncfile.createVariable(ncobj.var,'i2',('tomo_zdim','tomo_ydim', 'tomo_xdim'))
    #temp1 = np.uint16(tomodata_concr[fn_save])
    #tomo[:,:,:] = temp1[0:flen[fn_save],0:ydim,0:xdim]
    tomo[:,:,:] = ncobj.ncdata #temp1[0:flen[fn_save],0:ydim,0:xdim]
    tosave_ncfile.close()
    gc.collect()

    
    
def write_picklefile(data, filename):
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    gc.collect()



def write_npyfile(data, filename):
    np.save(filename, data)
    gc.collect()



def write_imagefiles(dataset, filepath, filenames=[]):
    mstart = time.time()
    
    imgfiles = []
    if filenames:
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
            print("\n\nNew folder created.")
        imgfiles = [os.path.join(filepath,f) for f in filenames]
    else:
        imgfiles.append(filepath)
        if not type(dataset)==list:
            dataset = [dataset]
        
    try:
        print("\n", len(dataset), " items in dataset and ", len(imgfiles), " file names provided.")
        len(dataset) == len(imgfiles)
    except:
        raise Exception
    else:
#        maxval = np.iinfo(np.uint8).max
#        print(maxval)
        with tqdm(imgfiles, desc="Writing image files", ncols=100) as pbar:
            for data, imgfile in zip(dataset, imgfiles):
                if len(data.shape)==3 and data.shape(2)>3:
                    print("\n 3D data cannot be written as image.")
                    raise ValueError
                elif len(data.shape)>3:
                    print("\n Multi-dimensional array with more than 3 dimensions\
                          cannot be saved as image.")
                    raise ValueError
                else:
                    imageio.imwrite(imgfile, data)
                    pbar.update()
    finally:    
        mstop = time.time()
        print("\n time taken: ", mstop-mstart)
        gc.collect()



def write_multiple_files(fobjs, filepath, filenames=[], file_extn=None): # Tested OK
    '''Generic function to write multiple files.'''
    mstart = time.time()
    
    files = []
    if filenames:
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
            print("\n\nNew folder created.")
        files = [os.path.join(filepath,f) for f in filenames]
    else:
        files.append(filepath)
        if not type(fobjs)==list:
            fobjs = [fobjs]
    
    if file_extn is None:
        file_extn = os.path.splitext(files)[-1]
    elif not os.path.splitext(files[0])[-1] == file_extn:
        files = replace_extn(files, replaceby_extn=file_extn)
   
    write_func = globals()[filetype_to_writefunc[file_extn]]
    
    try:
        print("\n", len(fobjs), " NCData objects and ", len(files), " file names provided.")
        len(fobjs) == len(files)
    except:
        raise Exception
    else:
        with tqdm(files, desc="Writing '.nc' files", ncols=100) as pbar:
            for fobj, file in zip(fobjs, files):
                write_func(fobj, file)
                pbar.update()
    finally:    
        mstop = time.time()
        print("\n time taken: ", mstop-mstart)
        gc.collect()



def write_ncfiles(ncobjs, filepath, filenames=[]):
    '''Function to write multiple '.nc, files.'''
    write_multiple_files(ncobjs, filepath, filenames, file_extn='.nc')



def write_picklefiles(dataset, filepath, filenames=[]): # Tested OK
    '''Function to write multiple '.pickle, files.'''
    write_multiple_files(dataset, filepath, filenames, file_extn='.pickle')



def write_npyfiles(dataset, filepath, filenames=[]):
    '''Function to write multiple '.pickle, files.'''
    write_multiple_files(dataset, filepath, filenames, file_extn='.npy')



#------------------------------------------------------------------------------  
# =============================================================================
# # TESTING
# if __name__ == '__main__':
#     ostart = time.time()
#     filepath = "E:\\0ImageProcessing\\Threshold_technnique\\Data\\Tomo"
# #    ncobjs, ncfiles = read_ncfiles(filepath)
# #    dataset = []
# #    for ncobj in ncobjs:
# #        dataset.append(ncobj.ncdata)
# #    filenames = replace_extn(ncfiles, '.pickle')
# #    write_picklefiles(dataset, filepath, filenames)
#     dataset, filenames = read_picklefiles(filepath)
#     filepath = "E:\\0ImageProcessing\\Threshold_technnique\\Data\\Tomo_res"
#     ncobjs = []
#     for data in dataset:
#         ncobjs.append(NCData(data))
#     ncfiles = replace_extn(filenames, '.nc')
#     write_ncfiles(ncobjs, filepath, ncfiles)
# # =============================================================================
# #     write_ncfiles(ncobjs, filepath, ncfiles)
# #     dataset =[]
# #     for ncobj in ncobjs:
# #         dataset.append(ncobj.ncdata)
# #     # testing write_picklefiles
# #     filenames = replace_extn(ncfiles, '.pickle')
# #     write_picklefiles(dataset, filepath, filenames)
# #     
# #     #testing write_npyfiles
# #     filenames = replace_extn(ncfiles, '.npy')
# #     write_npyfiles(dataset, filepath, filenames)
# #     
# # =============================================================================
# # =============================================================================
# #     #testing write_imagefiles
# #     filenames = replace_extn(ncfiles, '.tiff')
# #     write_imagefiles(dataset, filepath, filenames)
# # =============================================================================
#     ostop = time.time()
# #    process = psutil.Process(os.getpid())   
# #    print(process.memory_info().rss/1024**2)  # in bytes 
# 
#     print("\n\n Overall time taken: ", ostop-ostart, " s.")
# =============================================================================