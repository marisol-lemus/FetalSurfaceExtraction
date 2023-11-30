import numpy as np
import nibabel as nib
from tqdm import tqdm
import os, glob, sys, time, pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import tensorflow as tf
import cv2
import tensorflow as tf
import numpy as np
import os
#from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D, MaxPooling2D, Input, concatenate, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import log_cosh
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, ResNet50V2
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser('   ==========   Fetal brain age prediction, made by Team Rocket 2023.11.03 ver.1)   ==========   ')
parser.add_argument('-train_csv',action='store',dest='train',type=str, default='BAST_train_all_files.csv', help='input csv table')
parser.add_argument('-val_csv',action='store',dest='valid',type=str, default='BAST_valid_all_files.csv', help='input csv table')
parser.add_argument('-batch_size',action='store',default=32,dest='num_batch',type=int, help='Number of batch')
parser.add_argument('-epochs',action='store',default=1000,dest='epochs',type=int, help='Number of epochs')
parser.add_argument('-learning_rate',action='store',dest='learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('-n_slice',action='store',dest='num_slice',default=3,type=int, required=True, help='Number of training slice from a volume')
parser.add_argument('-slice_mode',action='store',dest='slice_mode',default=0,type=int, required=True, help='0: multi-slice training, 1: multi-channel training')
parser.add_argument('-tta',action='store',dest='num_tta',default=20, type=int, help='Number of tta')
parser.add_argument('-d_huber', action='store',dest='delta_huber', default=1.0, type=float, help='delta value of huber loss')
parser.add_argument('-gpu',action='store',dest='num_gpu',default='0', type=str, help='GPU selection')
parser.add_argument('-rl', '--result_save_location', action='store',default='./', dest='result_loc', required=True, type=str, help='Output folder name, default: ./')
parser.add_argument('-wl', '--weight_save_location', action='store',default='./', dest='weight_loc', required=True, type=str, help='Output folder name, default: ./')
parser.add_argument('-hl', '--history_save_location', action='store',default='./', dest='hist_loc', required=True, type=str, help='Output folder name, default: ./')
parser.add_argument('-output_csv',action='store',dest='output',type=str, default='output', help='name for csv logger')
args = parser.parse_args()

result_loc=args.result_loc
weight_loc=args.weight_loc
hist_loc=args.hist_loc
output_file=args.output


if os.path.exists(result_loc)==False:
    os.makedirs(result_loc,exist_ok=True)
if os.path.exists(weight_loc)==False:
    os.makedirs(weight_loc, exist_ok=True)
if os.path.exists(hist_loc)==False:
    os.makedirs(hist_loc, exist_ok=True)
if os.path.exists(output_file)==False:
    os.makedirs(output_file, exist_ok=True)

print('\n\n')
print('\t\t Prediction result save location: \t\t\t'+os.path.realpath(result_loc))
print('\t\t Prediction weights save location: \t\t\t'+os.path.realpath(weight_loc))
print('\t\t Prediction history save location: \t\t\t'+os.path.realpath(hist_loc))
print('\t\t number training slice: \t\t\t\t'+str(args.num_slice))
print('\t\t Slice mode: \t\t\t\t\t\t'+str(args.slice_mode))
print('\t\t TTA times: \t\t\t\t\t\t'+str(args.num_tta))
print('\t\t batch_size: \t\t\t\t\t\t'+str(args.num_batch))
print('\t\t learning_rate: \t\t\t\t\t'+str(args.learning_rate))
print('\t\t delta of Huber loss: \t\t\t\t\t'+str(args.delta_huber))
print('\t\t GPU number: \t\t\t\t\t\t'+str(args.num_gpu))
print('\n\n')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.num_gpu

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

batch_size = args.num_batch
num_slice = args.num_slice
epochs = args.epochs

def huber_loss(y_true, y_pred, delta=args.delta_huber ): #args.delta_huber
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear


def crop_pad_ND(img, target_shape):
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result



def make_dic(img_list, num_slice, slice_mode=0, desc=''):
    max_size = [290, 290, 1]
    target_size = (224,224)
    if slice_mode:
        dic = np.zeros([len(img_list), max_size[1], max_size[0], num_slice],dtype=np.float16)
    else:
        dic = np.zeros([len(img_list)*num_slice, 224, 224, 1],dtype=np.float16)
    for i in tqdm(range(0, len(img_list)),desc=desc):
        try:
            img = np.squeeze(nib.load(img_list[i]).get_fdata())
            img = np.squeeze(nib.load(img_list[i]).get_fdata())
            #edite aqui
            assert not np.isnan(img).any()
            img = ((img - float(img.min())) / float(img.max() +1E-7))*float(255)
            img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)), axis=0))
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img = img / 255.0
            #deje de editar aqui
            #img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)),axis=0))
            # img = (img-np.mean(img))/np.std(img)
            if slice_mode:
                dic[i,:,:,:]=np.swapaxes(img[:,:,int(img.shape[-1]/2-1-np.int(num_slice/2)):int(img.shape[-1]/2+np.int(num_slice/2))],0,1)
            else:
                dic[i*num_slice:i*num_slice+num_slice,:,:,0]=np.swapaxes(img[:,:,int(img.shape[-1]/2-1-np.int(num_slice/2)):int(img.shape[-1]/2+np.int(num_slice/2))],0,2)
                # dic[i*num_slice:i*num_slice+num_slice,:,:,1]=dic[i*num_slice:i*num_slice+num_slice,:,:,0]
                # dic[i*num_slice:i*num_slice+num_slice,:,:,2]=dic[i*num_slice:i*num_slice+num_slice,:,:,0]
        except Exception as e:
            print(f"Error loading file {img_list[i]}: {e}")
    return dic


datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=360,
    width_shift_range=0.15,
    height_shift_range=0.1,
    #brightness_range=[0.5,1],
    vertical_flip=True,
    horizontal_flip=True)
    


def tta_prediction(datagen, model, dic, n_example):
    preds=np.zeros([len(dic),])
    for i in range(len(dic)):
        image = np.expand_dims(dic[i],0)
        pred = model.predict_generator(datagen.flow(image, batch_size=n_example),workers=4,steps=n_example, verbose=0)
        preds[i]=np.mean(pred)
    return preds



def age_predic_network(img_shape):
    model = ResNet50V2(input_shape=img_shape,include_top=False, weights=None, pooling='avg')
    o = Dropout(0.3)(model.layers[-1].output)
    o = Dense(1,activation='linear')(o)
    model = Model(model.layers[0].output, o)
    model.compile(optimizer=Adam(learning_rate=args.learning_rate,decay=0.001), loss=huber_loss, metrics=['mae','mean_absolute_percentage_error'])
    return model

model = age_predic_network([224,224,1])


callbacks = [EarlyStopping(monitor='val_mae', patience=300, verbose=1, mode='min'),
                ModelCheckpoint(filepath=weight_loc+'/best_fold'+'_rsl.h5', monitor='val_mae', 
                save_best_only=True, mode='min', save_weights_only=True, verbose=0),
                CSVLogger(str(output_file)+'Log.csv', separator=",", append=True)]


train_df = pd.read_csv(args.train)
valid_df = pd.read_csv(args.valid)

train_dic = make_dic(train_df.MR.values, num_slice, slice_mode=0, desc='make train dic')
val_dic = make_dic(valid_df.MR.values, num_slice, slice_mode=0, desc='make val dic')

train_GW = train_df.GW.values
b_train_GW = np.zeros([len(train_GW)*num_slice,])

for tt in range(len(train_GW)):
    b_train_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(train_GW[tt],(num_slice,))

val_GW = valid_df.GW.values
b_val_GW = np.zeros([len(val_GW)*num_slice,])

for tt in range(len(val_GW)):
    b_val_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(val_GW[tt],(num_slice,))


histo = model.fit(datagen.flow(train_dic,b_train_GW,batch_size=batch_size,shuffle=True),steps_per_epoch=len(train_dic)/batch_size,epochs=epochs, validation_data=datagen.flow(val_dic, b_val_GW, batch_size=batch_size,shuffle=True),validation_steps=len(val_dic),workers=8,callbacks=callbacks, verbose=2)

with open(hist_loc+'/history_fold_rsl.pkl', 'wb') as file_pi:
        pickle.dump(histo.history, file_pi)
model.load_weights(weight_loc+'/best_fold_rsl.h5')

del model, histo

K.clear_session()
tf.compat.v1.reset_default_graph()
