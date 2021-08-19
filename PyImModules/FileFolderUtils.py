# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:56:53 2019

@author: Lakshminarayanan Mohana Kumar
"""

import os
import fnmatch

cwd=os.getcwd()

__version__ = '4.4'

# =============================================================================
#  Changes:
# __version__ = '4.0':
#     verify_file_or_folder() changed to is_file_or_folder(filepath)
# version = 4.2: updates list_files, list_filepaths, and append2fname
# version = 4.3: included join_filepaths
# version = 4.4: fixed bug with append2fname
# =============================================================================

#------------------------------------------------------------------------------
def is_file_or_folder(filepath):
    ''' Detects whether a given filepath refers to a folder or file.
    Output is a string stating 'folder' or 'file', or None if both criteria are
    not met.'''
    
    if os.path.isdir(filepath):
        fileorfolder = 'folder'
    elif os.path.isfile(filepath):
        fileorfolder = 'file'
    else:
        fileorfolder = None
    print("\n Filepath refers to a ", fileorfolder, ".")
    return fileorfolder



def list_files(filepath, file_extn=None):
    '''Based on whether given filepath is 'folder' or 'file', lists all the files
    to be read. If file_extn is specified, only files with matching file-extension
    are included. The files selected in this function may be used in a read function
    to read data from these'''
    
    fileorfolder = is_file_or_folder(filepath)
    filenames = []
    if fileorfolder is None:
        # if neither file or folder implies path was not detected!
        raise FileNotFoundError
    else:
        if fileorfolder == 'folder':
            if file_extn is not None:
                # include only those files with specified file_extn
                filenames = [f for f in os.listdir(filepath) if os.path.splitext(f)[-1]==file_extn]
                # print(filenames)
            else:
                filenames = [f for f in os.listdir(filepath)]
                # print(filenames)
        elif fileorfolder == 'file':
            filenames.append('')
        filenames.sort()
        return filenames


def list_filepaths(filepath, file_extn=None):
    '''Based on whether given filepath is 'folder' or 'file', lists all the files
    to be read. If file_extn is specified, only files with matching file-extension
    are included. The files selected in this function may be used in a read function
    to read data from these'''

    fileorfolder = is_file_or_folder(filepath)
    filenames = []
    if fileorfolder is None:
        # if neither file or folder implies path was not detected!
        raise FileNotFoundError
    else:
        if fileorfolder == 'folder':
            if file_extn is not None:
                # include only those files with specified file_extn
                filenames = [os.path.join(filepath, f) for f in os.listdir(filepath) if
                             os.path.splitext(f)[-1] == file_extn]
                # print(filenames)
            else:
                filenames = [os.path.join(filepath, f) for f in os.listdir(filepath)]
                # print(filenames)
        elif fileorfolder == 'file':
            filenames.append('')
        filenames.sort()
        return filenames


def append2fname(filename_or_listoffilenames, add):
    """Append existing filename with a given str = add.
    This can be used to modify existing filenames without
    changing their extensions."""
    if type(filename_or_listoffilenames) == str:
        # if is_file_or_folder(filename_or_listoffilenames) is 'file':
        if fnmatch.fnmatch(filename_or_listoffilenames, '*.*'):
            fname_parts = filename_or_listoffilenames.split('.')
            fname_parts[-2] += str(add)
            filename = '.'.join(fname_parts)
            return filename
        # elif is_file_or_folder(filename_or_listoffilenames) is 'folder':
        else:
            return filename_or_listoffilenames + str(add)
    elif type(filename_or_listoffilenames) == list:
        filenames = []
        for fname in filename_or_listoffilenames:
            fname_parts = fname.split('.')
            fname_parts[-2] += str(add)
            filenames.append('.'.join(fname_parts))
        return filenames
    else:
        print("\n Input must be a filename (string) or a list of file-names (list of strings).")
        raise ValueError


def join_filepaths(folder, fnames):
    filepaths = [os.path.join(folder, f) for f in fnames]
    return filepaths


def get_extns(filename_or_listoffilenames):
    '''Replace extensions of a file name or a list_of_files by relaceby_extn,
    output the new list of files'''
    extns = []
    print(type(filename_or_listoffilenames))
    if type(filename_or_listoffilenames) == str:
        fname = filename_or_listoffilenames
        fnameparts = fname.split('.')
        extns.append(fnameparts[-1])
        print(extns)
        return extns
    elif type(filename_or_listoffilenames) == list:
        filenames = []
        for fname in filename_or_listoffilenames:
            fnameparts = fname.split('.')
            if not (len(extns) and (fnameparts[-1] in extns)):
                extns.append(fnameparts[-1])
        print(extns)
        return extns
    else:
        print("\n Input must be a filename (string) or a list of file-names (list of strings).")
        raise ValueError

def replace_extn(filename_or_listoffilenames, replaceby_extn='.dat'):
    '''Replace extensions of a file name or a list_of_files by relaceby_extn, 
    output the new list of files'''
    if replaceby_extn[0] == '.':
        replaceby_extn = replaceby_extn[1:] #checking if '.' is included in given extn name
    
    if type(filename_or_listoffilenames) == str:
        fname = filename_or_listoffilenames
        fnameparts = fname.split('.')
        fnameparts[-1] = replaceby_extn
        filename = '.'.join(fnameparts)
        return filename
    elif type(filename_or_listoffilenames) == list:
        filenames = []
        for fname in filename_or_listoffilenames:
            fnameparts = fname.split('.')
            fnameparts[-1] = replaceby_extn
            filenames.append('.'.join(fnameparts))
        return filenames
    else:
        print("\n Input must be a filename (string) or a list of file-names (list of strings).")
        raise ValueError