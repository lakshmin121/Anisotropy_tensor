# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:29:21 2019

@author: Lakshminarayanan Mohana Kumar
"""

import os
import gc
import warnings
import copy
# import time

# import math
import numpy as np
from skimage import filters
# from skimage import util
from scipy import ndimage
from skimage import img_as_float
from medpy.filter.smoothing import anisotropic_diffusion
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import PyImModules.PyDataReadWrite as pydatarw
from PyImModules.NCData import NCData
from PyImModules.LocData import LocData

# =============================================================================
# from IPython import get_ipython
# ipy = get_ipython()
# from tvtk.util import ctf
# from matplotlib.pyplot import cm
# =============================================================================

# =============================================================================
# version = 4.4 : snapshot, aniso_diff, and unsharpmask.
# version = 4.4.1 : plot improvements to snapshot, aniso_diff, and unsharpmask.
# version = 4.4.3 : triangle threshold included.
# version = 4.4.4 : updated hysteresis threshold
# version = 4.4.5 : update threshold to accept string values in threshold range. (2020/05/13)
# =============================================================================


class TomoData:
    '''A class with data and associated functions to easily define, store and manipulate tomographic data.'''
    
    __version__ = '4.4.4'
    autothresh_list =   {'Otsu':'autothresh_Otsu',
                        'minerror': 'autothresh_minerror',
                        'knee': 'autothresh_knee',
                        'triangle': 'autothresh_triangle'
                        }
    
    def __init__(self, data, axesorder='zyx', axis_dir='row'):
        '''Initializes tomodata, 'data' should nbe numpy.ndarray.
        Recommended axesorder='xyz' and axis_dir='row' which means 
        each column vector is a point in 3D space of the format (x,y,z).'''
        self.axis_dir = axis_dir
        if type(data) == np.ndarray:
            if axis_dir == 'col':
                self.data = np.uint16(data.T)
            elif axis_dir == 'row':
                self.data = np.uint16(data)
            else:
                print("\nInput to axis_dir=", axis_dir, " not recognized.")
                raise ValueError
        else:
            print("\ndata is not of type numpy.ndarray.")
            raise TypeError("\ndata is not of type numpy.ndarray.")

        self.dim  = self.data.shape
        self.max = self.data.max()
        self.min = self.data.min()
        self.dtype = self.data.dtype
        
        axes ={}
        if type(axesorder) == str and len(axesorder) == 3:
            for indx, axis in enumerate(axesorder):
                axes[axis] = indx
            self.axesorder = axesorder
            self.axes = axes
        else:
            print("\nInvalid input for axesorder.")
            raise ValueError
    
    
    @classmethod
    def NCData2TomoData(cls, ncobj): # Reaffirmed v 4.1
        return cls(ncobj.ncdata, axesorder='zyx')
    
    
    @classmethod
    def make_copy(cls, tomo_obj): # updated in 2.5 # Reaffirmed v 4.1
        tomo_copy = copy.deepcopy(tomo_obj)
        return tomo_copy
        
    @classmethod
    def read_ncfile(cls, filename, var='tomo'): # Reaffirmed v 4.1
        dataset0 = Dataset(filename)
        ncdim = dataset0.variables[var][:,:,:].shape
        ncformat = str(dataset0.data_model)
        ncdata = np.array(dataset0.variables[var][:,:,:])
        ncobj = NCData(ncdata, ncdim, ncformat)
        
        return cls(ncobj.ncdata, axesorder='zyx')
    
    
    @classmethod
    def read_ncfiles(cls, filepath, var='tomo'): # Reaffirmed v 4.1
        ncobjs, ncfiles = pydatarw.read_ncfiles(filepath)
        ncobj = NCData.combine_ncobjects(ncobjs) # combine to single data
        
        return cls(ncobj.ncdata, axesorder='zyx')
    
    
    
    def write_ncfile(self, filepath, var='tomo', ncformat='NETCDF3_CLASSIC'): # Reaffirmed v 4.1
        ncdim = None
        ncobj = NCData(self.data, ncdim, var, ncformat)
        ncobj.write_ncfile(filepath)
        
    
    def snr(self):
        return np.mean(self.data)/np.std(self.data)
    
    
    # Updated in v 4.2
    def view_histogram(self, bkground=None, data_range=(None, None), binwidth=500, density_=True, 
                       fig_axis=None, **kwargs):
        """View histogram of the tomogram data. Pass axis of existing plot to fig_axis
        to view the histogram in this plot."""
        gc.collect()
        hist_data = self.data.flatten()
        if bkground is not None: 
            hist_data = hist_data[hist_data>bkground]
            gc.collect()
        if data_range[0] is not None:   
            hist_data = hist_data[hist_data>data_range[0]]
            gc.collect()
        if data_range[1] is not None:   
            hist_data = hist_data[hist_data<data_range[1]]
            gc.collect()
        
        max_ = np.iinfo(self.dtype).max
        bins_ = np.arange(0, max_+ binwidth, binwidth)
        if fig_axis is None:
            plt.figure()
            data, nbins, _ = plt.hist(hist_data, bins=bins_, density=density_, **kwargs)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.show()
        else:
            data, nbins, _ = fig_axis.hist(hist_data, bins=bins_, density=density_, **kwargs)
            fig_axis.tick_params(axis='both', labelsize=7)
        del hist_data
        gc.collect()
        return data, nbins

    
    # Updated in v 4.3
    def histogram(self, bkground=None, data_range=(None, None), binwidth=500, density_=True):
        hist_data = self.data.flatten()
        if bkground is not None: hist_data = hist_data[hist_data>bkground]
        if data_range[0] is not None:   hist_data = hist_data[hist_data>data_range[0]]
        if data_range[1] is not None:   hist_data = hist_data[hist_data<data_range[1]]
        
        max_ = np.iinfo(self.dtype).max
        bins_ = np.arange(0, max_+binwidth, binwidth)

        return np.histogram(hist_data, bins=bins_, density=density_)
        
    
    def extract_orthoslice(self, sliceno=None, normal_axis='z'):
        if sliceno is None:
            sliceno = int(self.dim[self.axes[normal_axis]]/2)
        tomo_orhtoslc = self.data.take(sliceno, axis=self.axes[normal_axis])
        return tomo_orhtoslc
        

    
    def view_orthoslice(self, sliceno=None, normal_axis='z', cmap=None, fig_axis=None, **kwargs): # Updated in v 4.2
        '''View an orthogonal slice of the volume data.'''

        if sliceno is None:
            sliceno = int(self.dim[self.axes[normal_axis]]/2)
        
        if cmap is None: cmap = 'gray'
        if fig_axis is None:
            plt.figure()
            plt.imshow(self.data.take(sliceno, axis=self.axes[normal_axis]),cmap=cmap)    
            plt.colorbar()
            plt.title("Orthoslice: " + normal_axis + " = " + str(sliceno))
            ax = plt.gca()
            ax.set_aspect('equal', anchor='C')
            plt.plot()
        else:
            fig_axis.imshow(self.data.take(sliceno, axis=self.axes[normal_axis]), cmap=cmap)
            fig_axis.set_axis_off()
    


    def threshold(self, threshold_range=(None,None), pad_val=0, **kwargs):
        """Return filtered/ thresholded data by setting data values at all points
        with values outside threshold_range equal to pad_value (default zero)"""
        thresh = copy.deepcopy(self)
        low, high = threshold_range
        if type(low) is str:
            try:
                low = getattr(self, self.autothresh_list[low])(**kwargs)
            except:
                print("\n String input to lower threshold not recognized.\
                 Available options: {}".format(self.autothresh_list.keys()))
                raise

        if type(high) is str:
            try:
                high = getattr(self, self.autothresh_list[high])(**kwargs)
            except:
                print("\n String input to lower threshold not recognized.\
                 Available options: {}".format(self.autothresh_list.keys()))
                raise
        if low is not None: thresh.data[thresh.data < low] = pad_val
        if high is not None: thresh.data[thresh.data > high] = pad_val
        return thresh


    
    def quantile_threshold(self, threshold_range=(0.25, 0.75), bkground=None, pad_val=0):
        '''Return image thresholded between a low and high threshold values.
        Default values of low and high thresholds are the 0.25 and 0.75 quantiles
        of the image.'''
        
        if any(threshold_range) < 0 or any(threshold_range) > 1:
            raise ValueError("Threshold range must be between [0, 1].")
        else:
            data_trimmed = self.data.flatten()
            if bkground is not None:
                data_trimmed = data_trimmed[data_trimmed>bkground]
            
            if threshold_range[0] is not None: 
                low_thresh  = np.quantile(data_trimmed, threshold_range[0])
                print("\nLow treshold: ", low_thresh)
            if threshold_range[1] is not None: 
                high_thresh = np.quantile(data_trimmed, threshold_range[1])
                print("\t High threshold: ", high_thresh)
            del data_trimmed
            
            thresh = copy.deepcopy(self)        
        
            if threshold_range[0] is not None: thresh.data[thresh.data < low_thresh] = pad_val
            if threshold_range[1] is not None: thresh.data[thresh.data > high_thresh] = pad_val
        
        return thresh
    


    def hysteresis_threshold(self, threshold_range=(0.10, 0.75)):
        """
        Returns hysteresis threshold mask.
        threshold_range: tuple of ints
            (low, high) to perform hysteresis threshold.
        Return
        ------
        bool, hysteresis threshold mask same shape e as image.
        """
        
        thresh = copy.deepcopy(self) # no need of copy if apply_hysteresis_threshold used
        if any([thresh < 0 for thresh in threshold_range]):
            raise ValueError("Threshold range cannot have negative values.")
        if threshold_range[1] < threshold_range[0]:
            raise ValueError("Higher threshold must be greater than lower threshold.")
        if all([0 < thresh < 1 for thresh in threshold_range]):
            thresh_low = threshold_range[0] * self.max
            thresh_high = threshold_range[1] * self.max
        else:
            thresh_low, thresh_high = threshold_range

        hyst_mask = filters.apply_hysteresis_threshold(self.data, thresh_low, thresh_high)

        return hyst_mask

    
    # Automating thresholds - Otsu, added in ver=1.1
    
    def autothresh_Otsu(self, binwidth=500, bkground=None): # Updated in v 4.2
        '''Returns threshold value corresponding to Ostsu thresholding.
        Removes background data if bkground is specified.'''
        max_ = np.iinfo(self.dtype).max
        if binwidth < max_:
            no_of_bins = np.iinfo(self.dtype).max // binwidth
            data_trimmed = self.data.flatten()
            if bkground is not None:
                data_trimmed = data_trimmed[data_trimmed>bkground] # removing background pixels
    #            print(data_trimmed)
            return filters.threshold_otsu(data_trimmed, nbins=no_of_bins)
        else:
            raise ValueError("binwidth is greater than max grayscale value.")

    
    
    # Added in v 4.2
    def autothresh_minerror(self, binwidth=500,  bkground=None, cval=1e-9): 
        """
        Modified from:
        <script src="https://gist.github.com/al42and/c2d66f6704e024266108.js"></script>
        
        The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
        Original Matlab code:
        --------------------
        https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
        Paper:
        -----
        Kittler, J. & Illingworth, J. Minimum error thresholding. 
        Pattern Recognit. 19, 41–47 (1986).
        """
        
        max_ = np.iinfo(self.data.dtype).max
        no_of_bins = max_//binwidth
        data_trimmed = self.data.flatten()
        if bkground is not None:
            data_trimmed = data_trimmed[data_trimmed>bkground] # removing background pixels
            
        h,g = np.histogram(data_trimmed, bins=no_of_bins)
        h = h.astype(np.float)
        g = g.astype(np.float)
        g = 0.5*(g[1:] + g[:-1])
        
        c = np.cumsum(h)
        m = np.cumsum(h * g)
        s = np.cumsum(h * g**2)
        c[c==0] = cval
        sigma_f = np.sqrt(s/c - (m/c)**2)
        cb = c[-1] - c
        mb = m[-1] - m
        sb = s[-1] - s
        cb[cb==0] = cval
        sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
        p =  c / c[-1]
        v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
        v[~np.isfinite(v)] = np.inf
        idx = np.argmin(v)
        t = g[idx]
        
        return t
    
    
    
    def autothresh_knee(self, binwidth=500, tol=0.1,  bkground=None, visualize=False):
        
        def fit_least_sqr_line(points):
            '''Fitting a line of the form   y = b*x + c;
            the values of a and b are given by:
                b = Cov(x,y) / Var(x)
                a = E(y) - b*E(x).'''
            
            covmat = np.cov(points) 
            b = covmat[0, 1] / np.var(points[0, :])
            a = np.mean(points[1, :]) - b*np.mean(points[0, :])
            
            y_fit = points[0, :]*b + a
            residuals = np.abs(y_fit - points[1, :])
            
            return b, a, residuals
        
        
        def line_intersection(m1, c1, m2, c2):
            '''Finding point of intersection between 2 lines given by
            y = m1*x + c1  and  y = m2*x + c2.'''
            y = (m1*c2 - m2*c1) / (m1 - m2)
            x = (c2 - c1) / (m1 - m2)
            
            return x, y
        
        # Obtaining histogram with specified bin-width
        no_of_bins = 65535//binwidth
        data_trimmed = copy.deepcopy(self.data).flatten()
        if bkground is not None:
            data_trimmed = data_trimmed[data_trimmed>bkground] # removing background pixels
            
        hist_, bin_edges = np.histogram(data_trimmed, bins=no_of_bins)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        
        if visualize:
            plt.figure()
            plt.plot(bin_centers, hist_)
        
        
        ### Initialize (Calculations)
        n_left = 10
        n_right = 10
        indx_peak = np.argmax(hist_)
        indx_tail = -1 
        b_peak = bin_centers[indx_peak]
        b_tail = bin_centers[indx_tail]
        hist_peak = hist_[indx_peak]
        hist_tail = hist_[indx_tail]
        
        left = [[bin_centers[indx_peak + i], hist_[indx_peak + i]] for i in range(n_left)]
        left = np.array(left).T
        
        right = [[bin_centers[indx_tail - i], hist_[indx_tail - i]] for i in range(n_right)]
        right = np.array(right).T
        
        b_left, a_left, residuals_left = fit_least_sqr_line(left)
        b_right, a_right, residuals_right = fit_least_sqr_line(right)
        
        while np.max(np.abs(residuals_left)) > tol:
            if len(left) > 2:
                del_indx = np.argmax(residuals_left[1:]) # index corresponding to maximum error other than hist_peak
                np.delete(left, del_indx)
                b_left, a_left, residuals_left = fit_least_sqr_line(left)
            else:
                break
        
        while np.max(np.abs(residuals_right)) > tol:
            if len(right) > 2:
                del_indx = np.argmax(residuals_right[:-1]) # index corresponding to maximum error other than hist_tail
                np.delete(right, del_indx)
                b_right, a_right, residuals_right = fit_least_sqr_line(right)
            else:
                break
        
        x0, y0 = line_intersection(b_left, a_left, b_right, a_right)
        threshold_vals = []
        threshold_vals.append(x0)
        
        if visualize:
            plt.plot([b_peak, x0], [hist_peak, y0])
            plt.plot([b_tail, x0], [hist_tail, y0])
        
        
        ### Iterations

        rel_errors = [100]
        
        while rel_errors[-1] > tol:
            
            left = [[binc, hist_[i]] for i, binc in enumerate(bin_centers) if i>indx_peak and binc<threshold_vals[-1]]
            right = [[binc, hist_[-1-i]] for i, binc in enumerate(reversed(bin_centers)) if binc>threshold_vals[-1]]
            
            left.append([x0, y0])
            right.append([x0, y0])
            
            left = np.array(left).T
            right = np.array(right).T
            
            b_left, a_left, residuals_left = fit_least_sqr_line(left)
            b_right, a_right, residuals_right = fit_least_sqr_line(right)
            
            while np.max(np.abs(residuals_left)) > tol:
                if len(left) > 2:
                    del_indx = np.argmax(residuals_left[1:]) # index corresponding to maximum error other than hist_peak
                    np.delete(left, del_indx)
                    b_left, a_left, residuals_left = fit_least_sqr_line(left)
                else:
                    break
        
            while np.max(np.abs(residuals_right)) > tol:
                if len(right) > 2:
                    del_indx = np.argmax(residuals_right[:-1]) # index corresponding to maximum error other than hist_tail
                    np.delete(right, del_indx)
                    b_right, a_right, residuals_right = fit_least_sqr_line(right)
                else:
                    break
            
            x0, y0 = line_intersection(b_left, a_left, b_right, a_right)
            threshold_vals.append(x0)
            
            if visualize:
                plt.plot([b_peak, x0], [hist_peak, y0])
                plt.plot([b_tail, x0], [hist_tail, y0])
            
            rel_errors.append(100*abs((threshold_vals[-1] - threshold_vals[-2]) / threshold_vals[-2]))
        
        if visualize: plt.show()
        return threshold_vals[-1]



    def autothresh_triangle(self, binwidth=500, bkground=None):
        """
        Returns threshold value estimated using triangle method (similar to knee).

        binwidth: same dtype as image
            binwidth for histogram
        bkground: same dtype as image
            only values about this in the image will be considered
        return: float
            threshold value

        Based on:
        https://github.com/scikit-image/scikit-image/blob/feafe48094f62c06972bcb7ea87ad6ca27fbde9b/
        skimage/filters/thresholding.py#L764
        """
        image_data = self.data.flatten()
        image_data = image_data[image_data > bkground]
        hist, bin_edges = np.histogram(image_data, bins=500)
        resolution = bin_edges[1] - bin_edges[0]
        arg_peak_height = np.argmax(hist)
        peak_height = hist[arg_peak_height]

        arg_low_level, arg_high_level = np.where(hist > 0)[0][[0, -1]]
        flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height

        if flip:
            nbins = len(hist)
            hist = hist[::-1]
            arg_low_level = nbins - arg_high_level - 1
            arg_peak_height = nbins - arg_peak_height - 1

        # If flip == True, arg_high_level becomes incorrect
        # but we don't need it anymore.
        del (arg_high_level)

        # Set up the coordinate system.
        width = arg_peak_height - arg_low_level
        x1 = np.arange(width)
        y1 = hist[x1 + arg_low_level]

        # Normalize.
        norm = np.sqrt(peak_height ** 2 + width ** 2)
        peak_height /= norm
        width /= norm

        length = peak_height * x1 - width * y1
        arg_level = np.argmax(length) + arg_low_level

        if flip:
            arg_level = nbins - arg_level - 1

        return bin_edges[arg_level]
    
    
    def binarize(self):
        '''Return binarized data by setting all positive data values to unity and
        others to zero.'''
        binarized = copy.deepcopy(self)
        binarized.data[binarized.data > 0] = 1
        return binarized
    
    
    def invert(self): # updated: 2020-03-10 version=3.3
        '''Return inverted image: darker regions change to brighter and vise versa'''
        inverted = self.data.astype(np.float32)
#        max_ = inverted.max()
#        print(max_)
#        inverted /= max_ #normalizing the data
        inverted = np.ones(self.dim, dtype=np.uint16)*inverted.max() - inverted #inverting
#        inverted *= max_ #rescaling the data 
        return TomoData(inverted)
    
    
    def mask(self, mask_, copy_=False):
        if isinstance(mask_, TomoData):
            maskimg = mask_.data
        else:
            maskimg = mask_
        
        if copy_:
            img1 = np.copy(self.data)
            img1 = np.multiply(img1, maskimg)
            return TomoData(img1, axesorder=self.axesorder)
        else:
            self.data = np.multiply(self.data, maskimg)
            
    
    
    
    def sobel_3D(self, directional=False, sigma=0, **kwargs):
        """Calculates 3D sobel gradient of image.
        
        Parameters
        ----------
        directional: Accepts values False, 'gradients', 'angles'.
            Default value is False, the output is then only sobel gradient magnitude
            G at each pixel. 'gradients' output Gx, Gy, and Gz values in addition to 
            gradient magnitude G at each pixel. 'angles' output thetaxy = atan2(Gy/Gx), 
            thetayz = atan2(Gz/Gy), and thetazx = atan2(Gx/Gz) in addition to 
            gradient magnitude G at each pixel.
        
        apply_filter: Applies a gaussian filter if True, 
            ndimage.gaussian_filter(input, sigma, order=0, output=None, 
                                    mode='reflect', cval=0.0, truncate=4.0)
            Named input values to the filter is passed using **kwargs.
        
        sigma_: Standard deviation for gaussian filter. Default value is 0.
        
        Returns
        -------
        3D sobel gradient TomoData object. 
        Additionally numpy arrays Gx, Gy, Gz OR thetaxy, thetayz, thetazx.
        """
        
        tomo_sbl = copy.deepcopy(self)
        img = tomo_sbl.data
        
        ### Applying gaussian filter
        
        if sigma>0:
            if sigma == 1:
                warnings.warn("\nsigma value for Gaussian filter is 1.")
            
            img = ndimage.gaussian_filter(img, sigma, **kwargs)
            
        gradient = np.zeros(img.shape, dtype=np.float32)
        Gx = ndimage.sobel(img, axis=2, output=np.float32)
        Gy = ndimage.sobel(img, axis=1, output=np.float32)
        Gz = ndimage.sobel(img, axis=0, output=np.float32)
        gradient += Gz**2
        gradient += Gy**2
        gradient += Gx**2
        gradient **= 0.5
        gradient /= 8
        tomo_sbl.data = np.uint16(gradient)
        
        if directional:
            if directional == 'gradients':
                return tomo_sbl, Gz, Gy, Gx
            elif directional == 'angles':
                thetaxy = np.arctan2(Gy, Gx, dtype=np.float32)
                thetayz = np.arctan2(Gz, Gy, dtype=np.float32)
                thetazx = np.arctan2(Gx, Gz, dtype=np.float32)
                
                del Gx, Gy, Gz
                return tomo_sbl, thetaxy, thetayz, thetazx
        else:
            return tomo_sbl
    
    
    
    
    def compute_volume(self):
        return np.count_nonzero(self.data)
    
    
    def extract_locdata(self):
        '''All positive data points are located and returned as locdata.'''
        return np.array(np.where(self.data>0))
    
    
    def extract_volocdata(self):
        locdata = np.array(np.where(self.data>0))
        return LocData.VolLocData(locdata, axesorder=self.axesorder, axis_dir=self.axis_dir)


    
    def aniso_diff(self, copy_=False, niter=1, kappa=50, gamma=0.1, voxelspacing=None,
                   option=1, msg=False):
        '''
        Anisotropic Diffusion from medpy.
        
        Parameters:
        ----------
        copy_: bool
            Output as copy.        	
        niter : integer
            Number of iterations.
        
        kappa : integer       
            Conduction coefficient, e.g. 20-100. kappa controls conduction 
            as a function of the gradient. If kappa is low small intensity gradients 
            are able to block conduction and hence diffusion across steep edges. 
            A large value reduces the influence of intensity gradients on conduction.
        
        gamma : float        
            Controls the speed of diffusion. Pick a value <=.25 for stability.
        
        voxelspacing : tuple of floats or array_like
            The distance between adjacent pixels in all img.ndim directions
        
        option : {1, 2, 3}
            Whether to use the Perona Malik diffusion equation No. 1 or No. 2, 
            or Tukey’s biweight function. Equation 1 favours high contrast edges 
            over low contrast ones, while equation 2 favours wide regions over 
            smaller ones. See [1] for details. 
            Equation 3 preserves sharper boundaries than previous formulations 
            and improves the automatic stopping of the diffusion. See [2] for details.
        
        Returns
        -------
        Output image anisotropically diffused as per option.
        
        References:
        ----------
        
        [1] P. Perona and J. Malik. Scale-space and edge detection using ansotropic 
        diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        12(7):629-639, July 1990.
        [2]  M.J. Black, G. Sapiro, D. Marimont, D. Heeger Robust anisotropic diffusion. 
        IEEE Transactions on Image Processing, 7(3):421-432, March 1998.
        
        From: https://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html
        '''
        max_ = self.max

        imgout = anisotropic_diffusion(img_as_float(self.data / max_, force_copy=copy_),
                                       niter=niter,
                                       kappa=kappa,
                                       gamma=gamma,
                                       option=option
                                       )
        r = []
        if copy_:
            tomo_out = TomoData(imgout*max_)
            r.append(tomo_out)
        if msg:
            message = "\n\nAnisotropic Diffusion: "\
                      + " \n\t input: {0} \n\tniter: {1} \n\t kappa: {2} \n\t gamma: {3} \
                    \n\t option: {4} \n\tOutput: {5}".format('tomo_sfrc', niter, kappa,
                                                             gamma, option, 'tomo_out'
                                                             )
            r.append(message)
        return tuple(r)

   


    def unsharpmask(self, copy_=False, rad=1, amt=1, msg=False):
        '''
        Performs unsharp mask based on skimage.fileters.unsharp_mask
        Parameters
        ----------
        copy_ : bool, optional
            If True, output image is a separate copy. The default is False.
        
        rad : scalar or sequence of scalars, optional
            If a scalar is given, then its value is used for all dimensions. 
            If sequence is given, then there must be exactly one radius for each 
            dimension except the last dimension for multichannel images. 
            Note that 0 radius means no blurring, and negative values are not allowed.
        
        amt : scalar, optional
            The details will be amplified with this factor. The factor could be 0 
            or negative. Typically, it is a small positive number, e.g. 1.0.

        Returns
        -------
        unsharped tomo image.
        '''

        r = []
        if msg:
            message = "\n\nUnsharp Mask:" + " \n\t input: {0} \n\t radius: {1}\
                    \n\t amount: {2} \n\t output: {3}".format('tomo_out', rad, amt, 'tomo_unsh')

        if copy_: 
            image = np.copy(self.data)
            image = filters.unsharp_mask(image, radius=rad, amount=amt,
                                         preserve_range=True
                                         )
            tomo_out = TomoData(image)
            r.append(tomo_out)
            if msg:
                r.append(message)
            return tuple(r)
        else:
            self.data = filters.unsharp_mask(self.data, radius=rad, amount=amt,
                                             preserve_range=True
                                             )
            if msg:
                return message
            



    def snapshot(self, ax, title=None, sliceno=None, normal_axis='z', cmap=None, bkground=None,
                 data_range=(None, None), binwidth=500, density_=True):
        """
        Plots a snapshot of the data with 2 subplots on given axis:
            subplot1 : A slice of the data.
            subplot2 : Histogram of the data.

        Parameters
        ----------
        ax : matplotlib axes
            A matplotlib subplot axes with 2 rows.
            
        sliceno : int, optional
            Slice of the data to be displayed. The default is None.
            
        normal_axis : 'x', 'y', or 'z'., optional
            Axis along whihc the slice no is specidfied. The default is 'z'.
            
        cmap : TYPE, optional
            cmap for data to be plotted as image. The default is None which
            renders cmap='gray'.
            
        bkground : uint8 or uint16, optional
            The grayscale value of background, not included in the histogram plotted. 
            The default is None.
            
        data_range : (low, high), optional
            low and high could be uint8, uint16, or None. Together the tuple
            specifies the data range to be included in the histogram plot.
            The default is (None, None) which means entire data range to be plotted.
            
        binwidth : int, optional
            uniform bin width for histogram. The default is 500 (assuming data is
                                                                 of type uint16).
            
        density_ : Bool, optional
            If Ture, converts histogram to a density plot. The default is True.

        Returns
        -------
        None.

        """
        self.view_orthoslice(sliceno=sliceno, normal_axis=normal_axis, cmap=cmap, 
                             fig_axis=ax[0]
                             )
        if title:
            ax[0].set_title(title, fontsize=7, fontweight='bold')
        self.view_histogram(bkground=bkground, data_range=data_range, 
                            binwidth=binwidth, density_=density_, fig_axis=ax[1]
                            )
        ax[1].set_yticklabels([])



    def save_snapshot(self, fig_fpath, title=None, sliceno=None, normal_axis='z', cmap=None, bkground=None,
                 data_range=(None, None), binwidth=500, density_=True):
        """
        Plots a snapshot of the data with 2 subplots on given axis:
            subplot1 : A slice of the data.
            subplot2 : Histogram of the data.

        Parameters
        ----------
        fig_fpath : str or os.path
            Output figure filepath

        sliceno : int, optional
            Slice of the data to be displayed. The default is None.

        normal_axis : 'x', 'y', or 'z'., optional
            Axis along whihc the slice no is specidfied. The default is 'z'.

        cmap : TYPE, optional
            cmap for data to be plotted as image. The default is None which
            renders cmap='gray'.

        bkground : uint8 or uint16, optional
            The grayscale value of background, not included in the histogram plotted.
            The default is None.

        data_range : (low, high), optional
            low and high could be uint8, uint16, or None. Together the tuple
            specifies the data range to be included in the histogram plot.
            The default is (None, None) which means entire data range to be plotted.

        binwidth : int, optional
            uniform bin width for histogram. The default is 500 (assuming data is
                                                                 of type uint16).

        density_ : Bool, optional
            If Ture, converts histogram to a density plot. The default is True.

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        ax = ax.flatten()
        self.view_orthoslice(sliceno=sliceno, normal_axis=normal_axis, cmap=cmap,
                             fig_axis=ax[0]
                             )
        if title:
            ax[0].set_title(title, fontsize=7, fontweight='bold')
        self.view_histogram(bkground=bkground, data_range=data_range,
                            binwidth=binwidth, density_=density_, fig_axis=ax[1]
                            )
        ax[1].set_yticklabels([])
        fig.savefig(fig_fpath, dpi=600)


                                      

#------------------- Under Development -------------------#     

# =============================================================================
#     def render_3D(self): # edit this
#         ipy.magic("gui qt")
#         #mlab.volume_slice(self.data)
#         volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data), vmin=0.2, vmax=1.0)
#         c = ctf.save_ctfs(volume._volume_property)
#         values = np.linspace(0., 1., 256)
#         c['rgb']=cm.get_cmap('gray_r')(values.copy())
#         ctf.load_ctfs(c, volume._volume_property)
#         volume.update_ctf = True
# =============================================================================
    


    
#------------------------------------------------------------------------------
### LIST OF UPDATES
        
# directional sobel