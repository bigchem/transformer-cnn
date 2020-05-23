#Converts Transformer CNN output model to a pickle 
#file to be used in standalone module.
#Pavel Karpov, carpovpv@gmail.com

import h5py
import numpy as np 
import pickle
import sys 
import struct
import os 

if len(sys.argv) != 6:
   print("Usage: python3 %s model-file-h5 regression|classification property-name eval units" % (sys.argv[0],));
   print("\nFor example, your model file (solubility.h5) contains a regression model for prognosing water solubility.\n"
         "Then to convert your model, use:"
         "\n\tpython3 conv.py solubility.h5 regression Solubility result.\n"
         "This command will create a hERG.pickle file compatible with standalone ochem.py script.\n"
         "The eval argument may contain any python eval transformation for the raw output of the model.\n"
         "Two variables can be used here: result - the output of the model, MolWt - molecular weight.\n"
         "For solbulity model one can use: \"math.pow(10.0, (result - 0.9) * 16.829999905824657 + 1.7120000410079956) * MolWt\"\n"
         "to convert the value to g/L.");
   sys.exit(0);

info = [sys.argv[3], sys.argv[2], sys.argv[4], sys.argv[5]];

DD = [];

d = np.load("embeddings.npy", allow_pickle = True);
for q in d:
   DD.append(q);

f = h5py.File(sys.argv[1], "r");

w = f["conv1d_6"]["conv1d_6"]["kernel:0"][:][0];
b = f["conv1d_6"]["conv1d_6"]["bias:0"][:];

DD.append(w);
DD.append(b);

#usual valid convolutions
w = f["conv1d_7"]["conv1d_7"]["kernel:0"][:];
b = f["conv1d_7"]["conv1d_7"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_8"]["conv1d_8"]["kernel:0"][:];
b = f["conv1d_8"]["conv1d_8"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_9"]["conv1d_9"]["kernel:0"][:];
b = f["conv1d_9"]["conv1d_9"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_10"]["conv1d_10"]["kernel:0"][:];
b = f["conv1d_10"]["conv1d_10"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_11"]["conv1d_11"]["kernel:0"][:];
b = f["conv1d_11"]["conv1d_11"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_12"]["conv1d_12"]["kernel:0"][:];
b = f["conv1d_12"]["conv1d_12"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_13"]["conv1d_13"]["kernel:0"][:];
b = f["conv1d_13"]["conv1d_13"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_14"]["conv1d_14"]["kernel:0"][:];
b = f["conv1d_14"]["conv1d_14"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_15"]["conv1d_15"]["kernel:0"][:];
b = f["conv1d_15"]["conv1d_15"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_16"]["conv1d_16"]["kernel:0"][:];
b = f["conv1d_16"]["conv1d_16"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["conv1d_17"]["conv1d_17"]["kernel:0"][:];
b = f["conv1d_17"]["conv1d_17"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["dense_3"]["dense_3"]["kernel:0"][:];
b = f["dense_3"]["dense_3"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["dense_4"]["dense_4"]["kernel:0"][:];
b = f["dense_4"]["dense_4"]["bias:0"][:];

DD.append(w);
DD.append(b);

w = f["dense_5"]["dense_5"]["kernel:0"][:];
b = f["dense_5"]["dense_5"]["bias:0"][:];

DD.append(w);
DD.append(b);

modeltype = sys.argv[2].lower().capitalize() + "-property";

w = f[modeltype][modeltype]["kernel:0"][:];
b = f[modeltype][modeltype]["bias:0"][:];

DD.append(w);
DD.append(b);

DD = [info, DD];

pickle.dump(DD, open(sys.argv[3] + ".pickle", "wb"));


