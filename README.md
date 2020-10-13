# Materials-Fingerprinting
 Data and python files to reproduce classification algorithm in Materials Fingerprinting manuscript.

 To make the c-version of the dpc distance, a makefile is provided, simply type
    make libdpc

The paths to the data files need to be specified, they are in the file
classify_utils.py on lines 103 and 106. Once they have been specified,
the classification may be run by typing:
    python3 tda_classify.py
