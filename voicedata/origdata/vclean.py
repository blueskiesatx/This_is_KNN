import pandas as pd
import numpy as np
import os
from random import sample
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

metadata = 'vox1_meta.csv'
wavdir = 'vox1_dev_wav'

dupdf = pd.read_csv('duplicates.txt', sep='/')

# define useful functions
def dfheader_remove_upper_space(df):
    '''
    change column names of dataframe to lowercase with underscores
    '''
    # get list of column names
    oldcol = list(df.columns.values)
    # convert each column name to lowercase and replace space with underscore
    newcol = [s.lower().replace(' ', '_') for s in oldcol]
    # create a rename dict with the above
    rendict = dict(zip(oldcol, newcol))
    # return renamed dataframe
    return(df.rename(columns=rendict))


def dedup(row):
    '''
    function to load files and extract features
    '''
    # path to voice directory
    vpath = os.path.join(os.path.abspath(wavdir), str(row.celeb_id))
    for vdir in os.listdir(os.fsencode(vpath)):
        # loop through subdirectories and count number of wav files

        vdirname = os.fsdecode(vdir)
        if vdirname in dupdf.fdir.values:
            delpath = os.path.join(vpath, vdirname)
            print(f'DELETING DIRECTORY: {delpath}')

            vfiledirpath = os.path.join(vpath, vdirname)
            for file in os.listdir(os.fsencode(vfiledirpath)):
                delfile = os.path.join(vfiledirpath, os.fsdecode(file))
                print(f'FILE: {delfile}')
                os.remove(delfile)

            os.rmdir(delpath)


def cleanup():
    '''
    Function to count number of wav files per celeb
    '''
    # read data file
    df = dfheader_remove_upper_space(pd.read_csv(metadata, sep='\t'))
    # remove duplicate directories
    df.apply(dedup, axis=1)


cleanup()

