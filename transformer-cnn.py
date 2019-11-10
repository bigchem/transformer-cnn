import sys
import os
import time
import pickle
import math
import configparser
import numpy as np
import csv
import h5py
import tarfile
import shutil
import math

from rdkit import Chem
from rdkit.Chem import SaltRemover
from layers import PositionLayer, MaskLayerLeft, \
                   MaskLayerRight, MaskLayerTriangular, \
                   SelfLayer, LayerNormalization

version = 3;
print("Version: ", version);

if(len (sys.argv) != 2):
    print("Usage: ", sys.argv[0], "config.cfg");
    sys.exit(0);

print("Load config file: ", sys.argv[1]);

config = configparser.ConfigParser();
config.read(sys.argv[1]);

def getConfig(section, attribute, default=""):
    try:
        return config[section][attribute];
    except:
        return default;

TRAIN = getConfig("Task","train_mode");
MODEL_FILE = getConfig("Task","model_file");
TRAIN_FILE = getConfig("Task","train_data_file");
APPLY_FILE = getConfig("Task","apply_data_file", "train.csv");
RESULT_FILE = getConfig("Task","result_file", "results.csv");
NUM_EPOCHS = int(getConfig("Details","n_epochs", "100"));
BATCH_SIZE = int(getConfig("Details","batch_size", "32"));
SEED = int(getConfig("Details","seed", "657488"));
CANONIZE = getConfig("Details","canonize");
DEVICE = getConfig("Details","gpu");
EARLY_STOPPING = float(getConfig("Details", "early-sopping", "0.9"));
AVERAGING = int(getConfig("Details", "averaging", "5"));
CONV_OFFSET = int(getConfig("Details", "conv-offset", 60));
FIXED_LEARNING_RATE = getConfig("Details", "fixed-learning-rate", "False");

FIRST_LINE = True

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE;

N_HIDDEN = 512;
N_HIDDEN_CNN = 512;
EMBEDDING_SIZE = 64;
KEY_SIZE = EMBEDDING_SIZE;

#our vocabulary
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$";
g_chars = set(chars);
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print("Using: ", DEVICE);
print("Set seed to ", SEED);

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False);
config.gpu_options.allow_growth = True;
tf.logging.set_verbosity(tf.logging.ERROR);
K.set_session(tf.Session(config=config));

try:
    tf.random.set_random_seed(SEED);
except:
    print ("not supported tf.random.set_random_seed(SEED)")

np.random.seed(SEED);

props = {};

def calcStatistics(p,y):

   r2 = np.corrcoef(p, y)[0, 1];
   r2 = r2 * r2;

   press = 0.0;
   for i in range(len(y)):
      press += ( p[i] - y[i] ) * ( p[i] - y[i] );

   rmsep = math.sqrt(press / len(y));
   return r2, rmsep;

def spc_step(data, threshold):

   tn = 0.0;
   tp = 0.0;
   fp = 0.0;
   fn = 0.0;

   for point in data:
      if(point[1] > 1.0e-3):
         if(point[0] >= threshold):
            tp += 1;
         else:
            fn += 1;
      else:
         if(point[0] < threshold):
            tn += 1;
         else:
            fp += 1;

   if(fp + tn == 0):
      fpr = 0.0;
   else:
      fpr = fp / (fp + tn);

   tpr = tp / (tp + fn);
   return fpr, tpr;

def spc(data):

   maxh = -sys.float_info.max;
   minh = +sys.float_info.max;

   x = [];
   for point in data:
      x.append(point[0]);
      if(point[0] > maxh):
         maxh = point[0];
      if(point[0] < minh):
         minh = point[0];


   step =  (maxh - minh) / 1000.0;

   h = minh - 0.1 * step + 1e-5;

   spc_data = [];

   prev_fpr = -1.0;
   prev_tpr = -1.0;

   while (h <= maxh + 0.1* step):
      fpr, tpr = spc_step(data, h);
      if(fpr != prev_fpr):
          spc_data.append((fpr, tpr, h));

      prev_fpr = fpr;
      prev_tpr = tpr;

      h += step;

   spc_data = sorted(spc_data, key= lambda tup: tup[0]);
   return spc_data;

def auc(data):
   S = 0.0;

   x1 = data[0][0];
   y1 = data[0][1];

   for i in range(1, len(data)):
      x2 = data[i][0];
      y2 = data[i][1];

      S += (x2 - x1) * (y1 + y2);
      x1 = x2;
      y1 = y2;

   S *= 0.5;
   return round(S,4);

def optimal_threshold(data):

   ot1 = 0.0;
   ot2 = 0.0;

   m1 = 10;
   m2 = -10;

   for point in data:
      r = point[1] - (1.0 - point[0]);
      if (r>=0):
         if(r <= m1):
            m1 = r;
            ot1 = point[2];
      else:
         if(r > m2):
            m2 = r;
            ot2 = point[2];


   ot = (ot1 + ot2) / 2.0;
   return round(ot,5);

def tnt(data, threshold):

   tn = 0;
   tp = 0;
   fp = 0;
   fn = 0;

   for point in data:
      if(point[1] > 1.0e-3):
         if(point[0] >= threshold):
            tp += 1;
         else:
            fn += 1;
      else:
         if(point[0] < threshold):
            tn += 1;
         else:
            fp += 1;

   return [tp, fp, tn, fn, (tp + tn) / (tp +tn + fp +fn)];

def auc_acc(data):
   spc_data = spc(data);
   v_auc = auc(spc_data);
   v_ot = optimal_threshold(spc_data);
   v_tnt = tnt(data, v_ot);

   return v_auc, v_tnt[4], v_ot;

class suppress_stderr(object):
   def __init__(self):
       self.null_fds = [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]
   def __enter__(self):
       os.dup2(self.null_fds[0],2)
   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)
       for fd in self.null_fds + self.save_fds:
          os.close(fd)

def findBoundaries(DS):

    for prop in props:

       x = [];
       for i in range(len(DS)):
         if DS[i][2][prop] == 1:
            x.append(DS[i][1][prop]);

       hist= np.histogram(x)[0];
       if np.count_nonzero(hist) > 2:
          y_min = np.min(x);
          y_max = np.max(x);

          add = 0.01 * (y_max - y_min);
          y_max = y_max + add;
          y_min = y_min - add;

          print(props[prop][1], "regression:", y_min, "to", y_max, "scaling...");

          for i in range(len(DS)):
             if DS[i][2][prop] == 1:
                DS[i][1][prop] = 0.9 + 0.8 * (DS[i][1][prop] - y_max) / (y_max - y_min);

          props[prop].extend(["regression", y_min, y_max]);

       else:
          print(props[prop][1], "classification");
          props[prop].extend(["classification"]);


def analyzeDescrFile(fname):

    first_row = FIRST_LINE;

    DS = [];
    ind_mol = 0;

    remover = SaltRemover.SaltRemover();

    line = 0;
    for row in csv.reader(open(fname, "r")):

       #if line > 100: break;
       #line = line + 1;

       if first_row:
          first_row = False;
          j = 0;
          for i, prop in enumerate(row):
             if prop == "mol": 
                ind_mol = i;
                continue;
             props[j] = [i, prop];
             j = j + 1;          
          continue;

       mol = row[ind_mol].strip();

       #remove molecules with symbols not in our vocabulary
       g_mol = set(mol);
       g_left = g_mol - g_chars;
       if len(g_left) > 0: continue;

       arr = [];
       try:
          if CANONIZE == 'True':
             with suppress_stderr():
                m = Chem.MolFromSmiles(mol);
                m = remover.StripMol(m);
                if m is not None and m.GetNumAtoms() > 0:
                   for step in range(10):
                      arr.append(Chem.MolToSmiles(m, rootedAtAtom = np.random.randint(0, m.GetNumAtoms()), canonical = False));
                else:
                   arr.append(mol);
          else:
             arr.append(mol);
       except:
          arr.append(mol);

       vals = np.zeros( len(props), dtype=np.float32 );
       mask = np.zeros( len(props), dtype =np.int8 );

       for prop in props:
           idx = prop;
           icsv = props[prop][0];
           s = row[icsv].strip();
           if s == '':
              val = 0;              
           else:
              val = float(s);
              mask[idx] = 1;
           vals[idx] = val;      

       arr = list(set(arr));
       for step in range(len(arr)):
          DS.append( [ arr[step], np.copy(vals), mask ]);
  
    findBoundaries(DS);

    return DS;

def gen_data(data):

    batch_size = len(data);

    #search for max lengths
    nl = len(data[0][0]);
    for i in range(1, batch_size, 1):
        nl_a = len(data[i][0]);
        if nl_a > nl:
            nl = nl_a;

    nl = nl + CONV_OFFSET;

    x = np.zeros((batch_size, nl), np.int8);
    mx = np.zeros((batch_size, nl), np.int8);

    z = [];
    ym = [];
 
    for i in range(len(props)):
       z.append(np.zeros((batch_size, 1), np.float32));
       ym.append(np.zeros((batch_size, 1), np.int8));

    for cnt in range(batch_size):

        n = len(data[cnt][0]);
        for i in range(n):
           x[cnt, i] = char_to_ix[ data[cnt][0][i]] ;
        mx[cnt, :i+1] = 1;

        for i in range(len(props)):    
           z[i][cnt] = data[cnt][1][i];
           ym[i][cnt ] = data[cnt][2][i]; 

    d = [x, mx];
  
    for i in range(len(props)):
       d.extend([ym[i]]);

    return d, z;

def data_generator(ds):

   data = [];
   while True:
      for i in range(len(ds)):
         data.append( ds[i] );
         if len(data) == BATCH_SIZE:
            yield gen_data(data);
            data = [];
      if len(data) > 0:
         yield gen_data(data);
         data = [];
      raise StopIteration();

def buildNetwork():

    unfreeze = False;
    n_block, n_self = 3, 10;

    l_in = layers.Input( shape= (None,));
    l_mask = layers.Input( shape= (None,));
 
    l_ymask = [];
    for i in range(len(props)):
       l_ymask.append( layers.Input( shape=(1, )));

    #transformer part
    #positional encodings for product and reagents, respectively
    l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask);
    l_left_mask = MaskLayerLeft()(l_mask);

    #encoder
    l_voc = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None, trainable = unfreeze);
    l_embed = layers.Add()([ l_voc(l_in), l_pos]);

    for layer in range(n_block):

       #self attention
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE, trainable= unfreeze) ([l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE, trainable = unfreeze), trainable = unfreeze) (l_con);
       if unfreeze == True: l_dense = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_dense, l_embed]);
       l_att = LayerNormalization(trainable = unfreeze)(l_add);

       #position-wise
       l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu', trainable = unfreeze)(l_att);
       l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1, trainable = unfreeze)(l_c1);
       if unfreeze == True: l_c2 = layers.Dropout(rate=0.1)(l_c2);
       l_ff = layers.Add()([l_att, l_c2]);
       l_embed = LayerNormalization(trainable = unfreeze)(l_ff);

    #end of Transformer's part
    l_encoder = l_embed;

    #text-cnn part
    #https://github.com/deepchem/deepchem/blob/b7a6d3d759145d238eb8abaf76183e9dbd7b683c/deepchem/models/tensorgraph/models/text_cnn.py

    l_in2 =  layers.Input( shape= (None,EMBEDDING_SIZE));

    kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
    num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160];

    l_pool = [];
    for i in range(len(kernel_sizes)):
       l_conv = layers.Conv1D(num_filters[i], kernel_size=kernel_sizes[i], padding='valid',
                              kernel_initializer='normal', activation='relu')(l_in2);
       l_maxpool = layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(l_conv);
       l_pool.append(l_maxpool);

    l_cnn = layers.Concatenate(axis=1)(l_pool);
    l_cnn_drop = layers.Dropout(rate = 0.25)(l_cnn);

    #dense part
    l_dense =layers.Dense(N_HIDDEN_CNN, activation='relu') (l_cnn_drop);

    #https://github.com/ParikhKadam/Highway-Layer-Keras
    transform_gate = layers.Dense(units= N_HIDDEN_CNN, activation="sigmoid",
                     bias_initializer=tf.keras.initializers.Constant(-1))(l_dense);

    carry_gate = layers.Lambda(lambda x: 1.0 - x, output_shape=(N_HIDDEN_CNN,))(transform_gate);
    transformed_data = layers.Dense(units= N_HIDDEN_CNN, activation="relu")(l_dense);
    transformed_gated = layers.Multiply()([transform_gate, transformed_data]);
    identity_gated = layers.Multiply()([carry_gate, l_dense]);

    l_highway = layers.Add()([transformed_gated, identity_gated]);

    #Because of multitask we have here a few different outputs and a custom loss.

    def mse_loss(prop):
       def loss(y_true, y_pred):
          y2 = y_true * l_ymask[prop] + y_pred * (1 - l_ymask[prop]);
          return tf.keras.losses.mse(y2, y_pred);
       return loss;

    def binary_loss(prop):
       def loss(y_true, y_pred):
          y2 = K.max(y_pred, 0) - y_pred * y_true + tf.log(1. + tf.exp(-K.abs(y_pred)))
          return tf.reduce_mean( y2 * l_ymask[prop]);
       return loss;


    l_out = [];
    losses = [];
    for prop in props:
       if props[prop][2] == "regression":
          l_out.append(layers.Dense(1, activation='linear', name="Regression-" + props[prop][1]) (l_highway));
          losses.append(mse_loss(prop));
       else:
          l_out.append(layers.Dense(1, activation='sigmoid', name="Classification-" + props[prop][1]) (l_highway));
          losses.append(binary_loss(prop));
 
    l_input = [l_in2];
    l_input.extend(l_ymask);
 
    mdl = tf.keras.Model(l_input, l_out);
    mdl.compile (optimizer = 'adam', loss = losses);

    mdl.summary();

    K.set_value(mdl.optimizer.lr, 1.0e-4);

    #so far we do not train the encoder part of the model.
    encoder = tf.keras.Model([l_in, l_mask], l_encoder);
    encoder.compile(optimizer = 'adam', loss = 'mse');
    encoder.set_weights(np.load("embeddings.npy", allow_pickle = True));

    return mdl, encoder;


if __name__ == "__main__":

    device_str = "GPU" + str(DEVICE);

    if TRAIN == "True":
        print("Analyze training file...");

        DS = analyzeDescrFile(TRAIN_FILE);
        mdl, encoder = buildNetwork();
 
        nall = len(DS);
        print("Number of all points: ", nall);

        inds = np.arange(nall);

        def data_generator2(dsc):
           while True:
              for i in range(len(dsc)):
                 yield dsc[i][0], dsc[i][1];

        if EARLY_STOPPING == 0:
           np.random.shuffle(inds);

           d = [DS[x] for x in inds];
           all_generator = data_generator(d);

           DSC_ALL = [];
           for x, y in all_generator:
               z = encoder.predict(x);
               DSC_ALL.append((z, y));

           all_generator = data_generator2(DSC_ALL);

        else:
           np.random.shuffle(inds);
           ntrain = int(EARLY_STOPPING * nall);

           print("Trainig samples:", ntrain, "validation:", nall - ntrain);
           inds_train = inds[:ntrain];
           inds_valid = inds[ntrain:];

           DS_train = [DS[x] for x in inds_train];
           DS_valid = [DS[x] for x in inds_valid];

           train_generator = data_generator(DS_train);
           valid_generator = data_generator(DS_valid);

           DSC_TRAIN = [];
           DSC_VALID = [];

           #calculate "descriptors"
           for x, y in train_generator:             
              z = encoder.predict([x[0], x[1]]);
              d = [z];
              for i in range(len(props)):
                 d.extend([x[i+2]]);
              DSC_TRAIN.append((d, y));

           for x, y in valid_generator:
              z = encoder.predict([x[0], x[1]]);
              d = [z];
              for i in range(len(props)):
                 d.extend([x[i+2]]);
              DSC_VALID.append((d, y));

           train_generator = data_generator2(DSC_TRAIN);
           valid_generator = data_generator2(DSC_VALID);

        class MessagerCallback(tf.keras.callbacks.Callback):

           def __init__(self, **kwargs):
              self.steps = 0;
              self.warm = 64;

              self.early_max = 0.2 * NUM_EPOCHS if EARLY_STOPPING > 0 else NUM_EPOCHS;
              self.early_best = 0.0;
              self.early_count = 0;

              if EARLY_STOPPING > 0:
                 self.valid_gen = data_generator2(DSC_VALID);
                 self.train_gen = data_generator2(DSC_TRAIN);
              else:
                 self.train_gen = data_generator2(DSC_ALL);

           def on_batch_begin(self, batch, logs={}):
              self.steps += 1;
              if FIXED_LEARNING_RATE == "False":
                 lr = 1.0 * min(1.0, self.steps / self.warm) / max(self.steps, self.warm);
                 if lr < 1e-4 : lr = 1e-4;
                 K.set_value(self.model.optimizer.lr, lr);

           def on_epoch_end(self, epoch, logs={}):
              if EARLY_STOPPING > 0:
                 print("MESSAGE: train score: {} / validation score: {} / at epoch: {} {} ".format(round(float(logs["loss"]), 7),
                       round(float(logs["val_loss"]), 7), epoch +1, device_str));
                 early = float(logs["val_loss"]);
                 if(epoch == 0):
                    self.early_best = early;
                 else:
                    if early < self.early_best :
                       self.early_count = 0;
                       self.early_best = early;
                    else:
                       self.early_count += 1;
                       if self.early_count > self.early_max:
                          self.model.stop_training = True;
                       return;

              else:
                 print("MESSAGE: train score: {} / at epoch: {} {} ".format(round(float(logs["loss"]), 5),
                       epoch +1, device_str));

              if EARLY_STOPPING == 0 and epoch >= (NUM_EPOCHS - AVERAGING-1):
                  self.model.save_weights("e-" + str(epoch) + ".h5");

              if os.path.exists("stop"):
                 self.model.stop_training = True;
              return;

        if EARLY_STOPPING > 0:
           history = mdl.fit_generator( generator = train_generator,
                     steps_per_epoch = len(DSC_TRAIN),
                     epochs = NUM_EPOCHS,
                     validation_data = valid_generator,
                     validation_steps = len(DSC_VALID),
                     use_multiprocessing=False,
                     shuffle = True,
                     verbose = 0,
                     callbacks = [ ModelCheckpoint("model/", monitor='val_loss',
                                        save_best_only= True, save_weights_only= True,
                                        mode='auto', period=1),
                                   MessagerCallback()]);

           mdl.load_weights("model/");   #restoring best saved model
           mdl.save_weights("model.h5");

        else:
           history = mdl.fit_generator( generator = all_generator,
                     steps_per_epoch = len(DSC_ALL),
                     epochs = NUM_EPOCHS,
                     use_multiprocessing=False,
                     shuffle = True,
                     verbose = 0,
                     callbacks = [MessagerCallback()]);

           print("Averaging weights");
           f = [];

           for i in range(NUM_EPOCHS - AVERAGING-1, NUM_EPOCHS):
              f.append(h5py.File("e-" + str(i) + ".h5", "r+"));

           keys = list(f[0].keys());
           for key in keys:
              groups = list(f[0][key]);
              if len(groups):
                 for group in groups:
                    items = list(f[0][key][group].keys());
                    for item in items:
                       data = [];
                       for i in range(len(f)):
                          data.append(f[i][key][group][item]);
                       avg = np.mean(data, axis = 0);
                       del f[0][key][group][item];
                       f[0][key][group].create_dataset(item, data=avg);
           for fp in f:
              fp.close();

           for i in range(NUM_EPOCHS - AVERAGING, NUM_EPOCHS):
              os.remove("e-" + str(i) + ".h5");
           os.rename("e-" + str(NUM_EPOCHS - AVERAGING-1) + ".h5", "model.h5");

        with open('model.pkl', 'wb') as f:
           pickle.dump(props, f);

        tar = tarfile.open(MODEL_FILE, "w:gz");
        tar.add("model.pkl");
        tar.add("model.h5");
        tar.close();

        if EARLY_STOPPING > 0:
           shutil.rmtree("model/");

        os.remove("model.pkl");
        os.remove("model.h5");

    elif TRAIN == "False":

       tar = tarfile.open(MODEL_FILE);
       tar.extractall();
       tar.close();

       props = pickle.load( open( "model.pkl", "rb" ));

       mdl, encoder = buildNetwork();
       mdl.load_weights("model.h5");

       first_row = FIRST_LINE;
       DS = [];

       fp = open(RESULT_FILE, "w");
       for prop in props:
          print(props[prop][1], end=",", file=fp);
       print("", file=fp);
 
       ind_mol = 0;

       if CANONIZE == 'True':
          remover = SaltRemover.SaltRemover();
          for row in csv.reader(open(APPLY_FILE, "r")):
             if first_row:
                first_row = False;
                continue;

             mol = row[ind_mol];
             g_mol = set(mol);
             g_left = g_mol - g_chars;
 
             arr = [];
             if len(g_left) == 0:
                try:
                   with suppress_stderr():
                      m = Chem.MolFromSmiles(mol);
                      m = remover.StripMol(m);
                      if m is not None and m.GetNumAtoms() > 0:
                         for step in range(10):
                            arr.append(Chem.MolToSmiles(m, rootedAtAtom = np.random.randint(0, m.GetNumAtoms()), canonical = False));
                      else:
                         arr.append(mol);
                except:
                    arr.append(mol);
             else:
                for i in range(len(props)):
                   print("error", end=",", file=fp);

             z = np.zeros(len(props), dtype=np.float32);
             ymask = np.ones(len(props), dtype=np.int8);

             d= [];
             for i in range(len(arr)):
                d.append( [arr[i], z, ymask]);
               
             x, y = gen_data(d);            
             internal = encoder.predict( [x[0], x[1]]); 

             p = [internal];
             for i in range(len(props)):
                 p.extend([x[i+2]]);

             y = mdl.predict( p );
             res = np.zeros( len(props)); 

             for prop in props:
                res[prop] = np.mean(y[prop]);               
                if props[prop][2] == "regression":
                   res[prop] = (res[prop] - 0.9) / 0.8 * (props[prop][4] - props[prop][3]) + props[prop][4];
                print(res[prop], end=",", file=fp);
             print("", file=fp);
                          
       else:

          arr = [];
          e = [];

          for row in csv.reader(open(APPLY_FILE, "r")):
             if first_row:
                first_row = False;
                continue;

             mol = row[ind_mol];
             g_mol = set(mol);
             g_left = g_mol - g_chars;

             if len(g_left) == 0:
                e.append(1);
                arr.append(mol);
             else:
                e.append(0);
                arr.append("CC");

             if (len(arr) == BATCH_SIZE):
      
                z = np.zeros(len(props), dtype=np.float32);
                ymask = np.ones(len(props), dtype=np.int8);

                d= [];
                for i in range(len(arr)):
                   d.append( [arr[i], z, ymask]);
               
                x, y = gen_data(d);            
                internal = encoder.predict( [x[0], x[1]]); 

                p = [internal];
                for i in range(len(props)):
                   p.extend([x[i+2]]);

                y = mdl.predict( p );
                res = np.zeros( len(props)); 

                for i in range(len(arr)):
                   if e[i] == 0:
                      for prop in props:
                          print("error", end=",", file=fp);
                      print("", file=fp);
                      continue;

                   for prop in props:
                      res = y[prop][i][0];
                      if props[prop][2] == "regression":
                         res = (res - 0.9) / 0.8 * (props[prop][4] - props[prop][3]) + props[prop][4];
                      print(res, end=",", file=fp);
                   print("", file=fp);
 
                arr = [];
                e = [];

          if len(arr):

             z = np.zeros(len(props), dtype=np.float32);
             ymask = np.ones(len(props), dtype=np.int8);

             d= [];
             for i in range(len(arr)):
                d.append( [arr[i], z, ymask]);
               
             x, y = gen_data(d);            
             internal = encoder.predict( [x[0], x[1]]); 

             p = [internal];
             for i in range(len(props)):
                p.extend([x[i+2]]);

             y = mdl.predict( p );
             res = np.zeros( len(props)); 

             for i in range(len(arr)):
                if e[i] == 0:
                   for prop in props:
                      print("error", end=",", file=fp);
                   print("", file=fp);
                   continue;

                for prop in props:
                   res = y[prop][i][0];
                   if props[prop][2] == "regression":
                      res = (res - 0.9) / 0.8 * (props[prop][4] - props[prop][3]) + props[prop][4];
                   print(res, end=",", file=fp);
                print("", file=fp);

       fp.close();

       os.remove("model.pkl");
       os.remove("model.h5");


    print("Relax!");
