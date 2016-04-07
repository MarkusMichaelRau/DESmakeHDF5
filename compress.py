

""" This module implements the compression functionality
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
nnum = 50
thresh = np.float16(0.001)
zlim = np.linspace(start=zmin, stop=zmax, num=nnum, dtype=np.float16)


def compress_pdf(data):
    """ Compress the PDF

    """
    simple_spline = np.array([], dtype=np.float16)
    index_array = np.array([], dtype=np.float16)
    for i in xrange(1, data.shape[1]):
        spline = spl(data[:, 0], data[:, i], ext=1)
        int_zw = spline(zlim)
        index_over = np.where(int_zw > thresh)[0]

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


def uncompress_dict(spline_data, index_data, coadd_id):

    """ Uncompress the sparse representation
    """
    lower_counter = 0.0
    higher_counter = 0.0
    dictlist = []
    #first dictionary specifies the configurations
    zlim_store = [float(el) for el in zlim]
    config = [zlim_store, float(thresh)]
    dictlist.append(config)
    for i in np.arange(0, len(index_data), step=2):
        if i == 0:
            higher_counter += (index_data[i+1] - index_data[i])
        else:
            lower_counter = higher_counter
            higher_counter += (index_data[i+1] - index_data[i])
        #curr_pdf[index_data[i]:index_data[i+1]] = spline_data[lower_counter:higher_counter]
        curr_list = []
        curr_list.append(float(coadd_id[i/2]))
        zw_list = spline_data[lower_counter:higher_counter].tolist()
        zw_list_store = [float(el) for el in zw_list]
        curr_list.append(zw_list_store)
        #curr_list.append(spline_data[lower_counter:higher_counter].tolist())
        curr_list.append([float(index_data[i]), float(index_data[i+1])])

        dictlist.append(curr_list)

    return dictlist

#unit test
if __name__ == '__main__':
    print 'Test if compression is working'
    from read_in import read_file
    from matplotlib import pyplot as plt
    isOk, z_pdf, coadd_id = read_file('Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf.hdf5')

    #compress the PDF

    compressed_data = compress_pdf(z_pdf)

    #uncompress the PDF
    reconstructed = uncompress(*compressed_data)

    spline_model = spl(zlim, reconstructed[:, 1])
    plt.plot(z_pdf[:, 0], spline_model(z_pdf[:, 0]))
    plt.plot(z_pdf[:, 0], z_pdf[:, 2])
    plt.show()
