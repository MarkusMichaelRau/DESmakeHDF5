""" This script converts the PDF input files and the
point prediction files from numpy data files into
HDF5 files. It also compressed the PDF using
Spline compression.

USAGE: python create.py PDF_file PointPred_file output_fname list_of_additional_names

PDF_file: filename of the file that stores the PDF values.
          Format: z_1      z_2     ... z_m
                  pdf1_d1  pdf1_d2 ... pdf1_dm
                                   ...
                  pdfn_d1 pdfn_d2  ... pdfn_dm
PointPred_file: filename of the file that stores point predictions.
          Format: 'COADD_ID' 'mean', 'median', 'mode', 'sample'

NOTE: PDFs have to be in the same order as specified in the point prediction file
output_fname: filename for the output

Author: MMRAU
Email: markusmichael.rau@googlemail.com
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spl
import pandas as pd
import sys

#Tuning parameters for the spline compression
#please adapt this to your data
zmin = 0.0
zmax = 5.0
nnum = 80
thresh = 0.001


class SplCompr(object):
    """ Class to implement the Spline Compression

    """

    def __init__(self, zmin=0.0, zmax=1.5, nnum=35, thresh=0.001):
        self.zmin = zmin
        self.zmax = zmax
        self.nnum = nnum
        self.zlim = np.linspace(start=zmin, stop=zmax, num=nnum)
        self.thresh = thresh

    def compress_pdf(self, data):
        """ Compress the PDF

        """
        simple_spline = np.array([], dtype=np.float16)
        index_array = np.array([], dtype=np.uint8)
        for i in xrange(1, data.shape[1]):
            spline = spl(data[:, 0], data[:, i], ext=1)
            int_zw = spline(self.zlim)
            index_over = np.where(int_zw > self.thresh)[0]

            if len(index_over) < 2:
                print len(index_over)
                raise ValueError('The grid spacing is too big. This can happen for instance if the PDF resembles a delta function.')

            try:
                index_array = np.append(index_array,
                                        np.array([index_over[0], index_over[-1]], dtype=np.float16))
            except IndexError as e:
                print "Decrease the grid spacing! This can happen if the PDFs are a bit undersmoothed."
                print e.message
            #which points are above the threshold?
            data_over = np.array(int_zw[index_over[0]:index_over[-1]], dtype=np.float16)
            simple_spline = np.append(simple_spline, data_over)
        return simple_spline, index_array

    def uncompress(self, spline_data, index_data):
        """ Uncompress the sparse representation
        """
        lower_counter = 0.0
        higher_counter = 0.0
        output = np.zeros((self.nnum, len(index_data)/2))
        for i in np.arange(0, len(index_data), step=2):
            curr_pdf = np.zeros((self.nnum,))
            if i == 0:
                higher_counter += (index_data[i+1] - index_data[i])
            else:
                lower_counter = higher_counter
                higher_counter += (index_data[i+1] - index_data[i])
            curr_pdf[index_data[i]:index_data[i+1]] = spline_data[lower_counter:higher_counter]
            output[:, i/2] = curr_pdf
        return output

    def get_zrange_weights(self, uncompress, zrange, z=None):
        """Calculate the weights by integrating in
        the predefined zrange

        """
        if z is None:
            z = self.zlim
        zrange_weights = np.ones((uncompress.shape[1],))
        for i in xrange(len(zrange_weights)):
            model = spl(z, uncompress[:, i], ext=1)
            zrange_weights[i] = model.integral(zrange[0], zrange[1])

        return zrange_weights

    def stack_pdf(self, zvalues, compressed_pdfs, zrange=None, weights=None):
        """Stack the PDFs with weights
        zlims: integrate PDF in the ranges
        weights: additional weights
        """

        uncompr = self.uncompress(compressed_pdfs[0],
                                  compressed_pdfs[1])
        if zrange is not None:
            zrange_weights = self.get_zrange_weights(uncompr, zrange)
            if weights is None:
                weights = zrange_weights
            else:
                weights *= zrange_weights

        data_stack = np.average(uncompr, axis=1, weights=weights)

        model = spl(self.zlim, data_stack, ext=1)
        model = spl(self.zlim, data_stack/model.integral(self.zmin, self.zmax), ext=1)
        outputpdf = model(zvalues)
        return np.column_stack((zvalues, outputpdf))

    def store_pdf(self, data, out_filename):
        """Store the compressed files

        """

        compr = self.compress_pdf(data)
        compr[0].astype('float16').tofile('PDF'+out_filename)
        compr[1].astype('uint8').tofile('IDX'+out_filename)
        print "Your PDF has been stored!"

    def read_pdf(self, out_filename):
        """Read the compressed files

        """

        pdfoutput = np.fromfile('PDF'+out_filename, dtype=np.float16)
        indexoutput = np.fromfile('IDX'+out_filename, dtype=np.uint8)

        return pdfoutput, indexoutput


if __name__ == '__main__':
    #get system args
    args = sys.argv[1:]

    if len(args) != 3:
        print "Wrong number of command line arguments"
        print "Usage like %>python create.py PDF_file PointPred_file output_fname"
        sys.exit(1)

    #read in command line arguments:

    pdf_file = args[0]
    point_pred = args[1]
    output_fname = args[2]
    rest_headernames = args[3:]
    #iteratively add point pred to pandas data_frame

    store = pd.HDFStore(output_fname, mode='a')

    #read in the point predictions
    print ['COADDED_OBJECTS_ID', 'mean', 'median', 'mode', 'sample'] + rest_headernames

    try:
        point_pred = pd.read_table(point_pred, sep=' ', names=['COADDED_OBJECTS_ID', 'mean', 'median', 'mode', 'sample'] + rest_headernames, header=None)
    except pd.parser.CParserError as e:
        print "The number of additional column names you specified does not coincide with the table format."
        print e.message
    except Exception, e:
        import traceback
        print traceback.format_exc()

    #add point predictions to the output
    store.append('point_pred', point_pred, complevel=5, complib='blosc')
    pdf_df = np.loadtxt(pdf_file)
    #extract the
    complete_pdf = pdf_df
    pdf_complete_df = pd.DataFrame(complete_pdf[1:, :])
    pdf_complete_df.columns = ['pdf_' + str(j) for j in complete_pdf[0, :]]
    store.append('pdf', pdf_complete_df, complevel=5, complib='blosc')

    #compress the PDFS
    spl_comp = SplCompr(zmin, zmax, nnum, thresh)
    #print type(pdf_df.columns)
    #print pdf_df.columns
    compressed_pdf = spl_comp.compress_pdf(pdf_df.T)
    dat_compressed = np.column_stack((spl_comp.zlim, spl_comp.uncompress(compressed_pdf[0], compressed_pdf[1]))).T

    compressed_df = pd.DataFrame(dat_compressed[1:, :])
    compressed_df.columns = ['pdf_' + str(j) for j in dat_compressed[0, :]]

    store.append('pdf_compressed', compressed_df, complevel=5, complib='blosc')
    store.close()

