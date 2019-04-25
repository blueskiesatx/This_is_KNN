import pandas as pd
import numpy as np
import os
from random import sample
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

metadata = 'vox1_meta.csv'
inputcsv = 'vmeta.csv'
numceleb = 200
vtrainfile = '../vtrain_200'
vtestfile = '../vtest_200'
wavdir = 'vox1_dev_wav'
maxsamples = 45
maxmfcc = 40

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

def featcompress(featlist, r, dlist):
    # print(f'COMPRESSING: {len(featlist)}')
    feature = np.mean(np.array(featlist).T, axis=1)
    # create dictionary
    rdict = {
        'feature': feature,
        'identity': r.celeb_name,
        'gender': r.gender,
        'nationality': r.nationality}
    dlist.append(rdict)
    return dlist


def audioparsermod(row):
    '''
    function to load files and extract features
    '''
    print(f'{row.name} PROCESSING: {row.celeb_name}')
    datalist = []
    dlist = row.dirlist
    compnum = row.filecount//maxsamples

    for vdpath in dlist:
        cfeatlist = []
        print(f'DIRECTORY: {vdpath}, compnum: {compnum}, fcount: {row.filecount}')
        fcount = 0
        fpathlist = [os.path.join(vdpath, os.fsdecode(file)) for file in os.listdir(os.fsencode(vdpath))]
        # loop through selected directories
        for fpath in fpathlist:
            fcount += 1
            # here kaiser_fast is a technique used for faster extraction
            # here kaiser_best is a technique used for higher quality
            X, sample_rate = librosa.load(fpath, res_type='kaiser_best')
            # extract mfcc feature from data and average across frames
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=maxmfcc)
            f = np.mean(mfcc.T, axis=0)

            cfeatlist.append(list(f))
            # average across files
            if fcount >= compnum:
                datalist = featcompress(cfeatlist, row, datalist)
                fcount = 0
                cfeatlist = []

        if fcount > 0:
            datalist = featcompress(cfeatlist, row, datalist)

    # select a random subset
    datalist = sample(datalist, maxsamples)
    print(f'DONE Name: {row.celeb_name}, Files: {len(datalist)}')
    # build and return dataframe
    return pd.DataFrame(datalist)


def audioparser(row):
    '''
    function to load files and extract features
    selects a random subset of files for each celebrity
    '''
    print(f'{row.name} PROCESSING: {row.celeb_name}')
    # select a random subset of files
    datalist = []
    filelist = sample(row.filelist, maxsamples)

    for fpath in filelist:
        # loop through selected wav files

        # here kaiser_fast is a technique used for faster extraction
        # here kaiser_best is a technique used for higher quality
        X, sample_rate = librosa.load(fpath, res_type='kaiser_best')
        # we extract mfcc feature from data
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=maxmfcc)
        feature = np.mean(mfcc.T, axis=0)
        # append to list of dictionaries
        rdict = {
            'feature': feature,
            'identity': row.celeb_name,
            'gender': row.gender,
            'nationality': row.nationality}
        datalist.append(rdict)

    print(f'DONE Name: {row.celeb_name}, Files: {len(datalist)}')
    # build and return dataframe
    return pd.DataFrame(datalist)


def filecount(row):
    '''
    function to load files and extract features
    '''
    # path to voice directory
    vpath = os.path.join(os.path.abspath(wavdir), str(row.celeb_id))
    dl = os.listdir(os.fsencode(vpath))
    dcount = len(dl)
    dlist = [os.path.join(vpath, os.fsdecode(vdir)) for vdir in dl]
    fcount = 0
    filelist = []
    fclist = []

    for vdpath in dlist:
        # loop through subdirectories and count number of wav files
        fl = os.listdir(os.fsencode(vdpath))
        fcount += len(fl)
        fclist.append(len(fl))
        filelist += [os.path.join(vdpath, os.fsdecode(file)) for file in fl]

    # print which celebs have least number of files
    # if fcount <= maxsamples:
        # print(f'{row.name} - Name: {row.celeb_name}, Nationality: {row.nationality}, dcount: {dcount}, fcount: {fcount}, compnum: {compnum}, fclist: {fclist}, cfcount: {cfcount}, cfclist: {cfclist}')

    return [dcount, dlist, fcount, filelist]


def inspect_data():
    '''
    Function to count number of wav files per celeb
    '''
    # read data file
    df = dfheader_remove_upper_space(pd.read_csv(inputcsv, sep='\t'))

    # get the count and list of files
    df['dircount'], df['dirlist'], df['filecount'], df['filelist'] = zip(*df.apply(filecount, axis=1))
    # drop unwanted columns
    df = df.drop(['celeb_id', 'set'], axis=1)
    print(df.describe())
    return df


def load_voice():
    '''
    Function to load all wav files, extract features and save as csv
    '''
    # read data file
    df = inspect_data()
    clist = []

    for _, row in df.iterrows():
        # loop through each celeb and build list of dataframes with name, gender and features
        clist.append(audioparser(row))

    # concatenate into single dataframe
    df = pd.concat(clist)
    return df


def data_preprocess(vdf):
    '''
    Function to do some pre-processing
    - split train vs test
    - scale the input features
    '''
    # extract feature, gender and name to numpy arrays
    X = np.vstack(vdf.feature.to_numpy())
    g = vdf.gender.to_numpy()
    n = vdf.nationality.to_numpy()
    i = vdf.identity.to_numpy()
    
    # split into train and test
    X_train, X_test, g_train, g_test, n_train, n_test, i_train, i_test = train_test_split(X, g, n, i, random_state=1, stratify=i)

    # apply standard scaler
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # build train and test dataframes
    traindf = pd.DataFrame({
        'feature': X_train_scaled.tolist(),
        'gender': g_train,
        'nationality': n_train,
        'identity': i_train})
    
    testdf = pd.DataFrame({
        'feature': X_test_scaled.tolist(),
        'gender': g_test,
        'nationality': n_test,
        'identity': i_test})

    traindf.to_pickle(vtrainfile)
    testdf.to_pickle(vtestfile)

# df = dfheader_remove_upper_space(pd.read_csv(metadata, sep='\t')).sample(n=numceleb).reset_index(drop=True)
# df.to_csv(inputcsv, sep='\t', index=False)

# inspect_data().to_csv('tmp.csv', index=False)
voice = load_voice()
print(voice.head(10))
data_preprocess(voice)

