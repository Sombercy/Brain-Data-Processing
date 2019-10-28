from scipy.io import loadmat
import pandas as pd

def load_data(filename, header = None, index_col = None, binarize = False):
    if filename[-3:] == 'csv': 
        df = pd.read_csv(filename, header = header, index_col = index_col)
    elif filename[-3:] == 'mat':
        df = loadmat(filename, squeeze_me=True)
    if (header != None):
        X = df.drop(['y'], axis = 1).values
        y = df['y'].values
        if binarize:
            y[y!=1] = 0
            df['y'] = y
    else:
        X = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values
        if binarize:
            y[y!=1] = 0
            df[df.columns[-1]] = y
    return df, X, y
