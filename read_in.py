"""
Modified version from Ben's routine
Author MMRAU
"""

import pandas as pd
import numpy as np
import os

#defined binning
#fixed for this module
binning = ['pdf_0.03', 'pdf_0.08', 'pdf_0.12', 'pdf_0.17', 'pdf_0.23', 'pdf_0.28', 'pdf_0.33',
           'pdf_0.38', 'pdf_0.43', 'pdf_0.48', 'pdf_0.53', 'pdf_0.58', 'pdf_0.62', 'pdf_0.68',
           'pdf_0.72', 'pdf_0.78', 'pdf_0.83', 'pdf_0.88', 'pdf_0.93', 'pdf_0.97', 'pdf_1.02',
           'pdf_1.07', 'pdf_1.12', 'pdf_1.17', 'pdf_1.22', 'pdf_1.27', 'pdf_1.32', 'pdf_1.38',
           'pdf_1.42', 'pdf_1.47', 'pdf_1.52', 'pdf_1.57', 'pdf_1.62', 'pdf_1.67', 'pdf_1.72',
           'pdf_1.77', 'pdf_1.82', 'pdf_1.88', 'pdf_1.92', 'pdf_1.97', 'pdf_2.02', 'pdf_2.07',
           'pdf_2.12', 'pdf_2.17', 'pdf_2.23', 'pdf_2.27', 'pdf_2.32', 'pdf_2.38', 'pdf_2.42',
           'pdf_2.48']


def _read_hdf(filename, cols):
    """ Checks that the hdf file is a valid file for our purposes"
    """
    #is args set?
    if cols is None:
        return False, 'you must have at least some defined cols'

    #does the file exist
    if os.path.exists(filename) is False:
        return False, 'file does not exist'

    #can I read the file into a buffer
    #try:

    df = pd.read_hdf(filename, key='pdf')
   # df2 = pd.read_hdf(filename, key='point_predictions')

    #is the object a pandas dataframe?
    if type(df) != pd.core.frame.DataFrame:
        return False, 'pandas dataframe not standard'

    #does it have the required columns?
    for i in cols:
        if i not in df:
            return False, 'missing column ' + i

    return True, df


def _read_file(filename, cols):

    if ('.fit' in filename[-7:]) or ('.csv' in filename[-7:]):
        raise NotImplementedError("Fits files are not supported currently.")
    if '.hdf5' in filename[-5:]:
        return _read_hdf(filename, cols)
    return False, 'currently unable to read file'


def read_file(filename):
    """ Read in the pdf file from disc

    """

    isOkay, d = _read_file('Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf.hdf5',
                           binning)

    zcols = [c for c in d.keys() if 'pdf_' in c]
    pdf_z_center = np.array([float(c.split('f_')[-1]) for c in zcols])
    return isOkay, np.column_stack((pdf_z_center, d[zcols].T)), np.array(d['COADD_OBJECTS_ID'])


#unit test
if __name__ == '__main__':
    isOk, z_pdf, coadd_obj_id = read_file('Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf.hdf5')
    if isOk is True:
        print "Test passed"
    else:
        print 'Test failed'
