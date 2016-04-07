""" Implements test cases to validate the correctness of the software
Author: MMRAU, markusmichael.rau@googmail.com
"""
from create import read_in

testfile = 'Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf'

def test_read_in():
    test_read = read_in(testfile)
    for idx, el in enumerate(test_read):

