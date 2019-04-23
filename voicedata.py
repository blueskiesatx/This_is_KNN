import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

vtrainfile = '../voicedata/vtrain'
vtestfile  = '../voicedata/vtest'

def data_encode(gtrain, gtest, ntrain, ntest, itrain, itest):
    # Label-encode gender
    genc = LabelEncoder()
    genc.fit(gtrain)

    # Convert gender labels to encoded values
    enc_gtrain = genc.transform(gtrain)
    enc_gtest  = genc.transform(gtest)

    # Convert encoded gender values to one-hot-encoding
    cat_gtrain = to_categorical(enc_gtrain)
    cat_gtest  = to_categorical(enc_gtest)

    # Label-encode nationality
    nenc = LabelEncoder()
    nenc.fit(ntrain)

    # Convert gender labels to encoded values
    enc_ntrain = nenc.transform(ntrain)
    enc_ntest  = nenc.transform(ntest)

    # Convert encoded gender values to one-hot-encoding
    cat_ntrain = to_categorical(enc_ntrain)
    cat_ntest  = to_categorical(enc_ntest)

    # Label-encode identity
    ienc = LabelEncoder()
    ienc.fit(itrain)

    # Convert gender labels to encoded values
    enc_itrain = ienc.transform(itrain)
    enc_itest  = ienc.transform(itest)

    # Convert encoded gender values to one-hot-encoding
    cat_itrain = to_categorical(enc_itrain)
    cat_itest  = to_categorical(enc_itest)

    return (
        genc, enc_gtrain, enc_gtest, cat_gtrain, cat_gtest,
        nenc, enc_ntrain, enc_ntest, cat_ntrain, cat_ntest,
        ienc, enc_itrain, enc_itest, cat_itrain, cat_itest)

def voicedata():
    '''
    Function to load the train and test data and apply encoding preprocessing
    on gender, nationality and identity
    '''
    # read pickle files
    traindf = pd.read_pickle(vtrainfile)
    testdf = pd.read_pickle(vtestfile)

    # extract appropriate columns to numpy arrays
    X_train = np.vstack(traindf.feature.to_numpy())
    g_train = traindf.gender.to_numpy()
    n_train = traindf.nationality.to_numpy()
    i_train = traindf.identity.to_numpy()

    X_test = np.vstack(testdf.feature.to_numpy())
    g_test = testdf.gender.to_numpy()
    n_test = testdf.nationality.to_numpy()
    i_test = testdf.identity.to_numpy()

    print(traindf.shape)
    print(testdf.shape)

    # get encoded and categorical values
    (genc, g_train_enc, g_test_enc, g_train_cat, g_test_cat,
     nenc, n_train_enc, n_test_enc, n_train_cat, n_test_cat,
     ienc, i_train_enc, i_test_enc, i_train_cat, i_test_cat) = data_encode(g_train, g_test, n_train, n_test, i_train, i_test)

    return (X_train, X_test,
            genc, g_train, g_train_enc, g_train_cat, g_test, g_test_enc, g_test_cat,
            nenc, n_train, n_train_enc, n_train_cat, n_test, n_test_enc, n_test_cat,
            ienc, i_train, i_train_enc, i_train_cat, i_test, i_test_enc, i_test_cat)

