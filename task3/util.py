#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:15:30 2017

Utilities
"""
import zipfile

class Utilities:
    
    def extract_zip(source, dest):
        zip_ref = zipfile.ZipFile(source, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()