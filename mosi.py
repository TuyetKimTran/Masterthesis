import numpy as np, json
import pickle, sys, argparse
from keras.models import Model
from keras import backend as K
from keras import initializers
#from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import RMSprop
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from keras.utils.layer_utils import get_source_inputs
from kutilities.layers import MeanOverTime
# from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
global seed
seed = 1337
np.random.seed(seed)
import gc
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import itertools
import h5py

#@title
#=============================================================
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#=============================================================
import tensorflow as tf
from keras.backend import set_session

config = tf.compat.v1.ConfigProto()
set_session(tf.compat.v1.Session(config=config))
#==============================================================
# test
def calculate_accuracy(prediction, test_label, print_detailed_results=False):

    true_label=[]
    predicted_label=[]

    for i in range(prediction.shape[0]):
        true_label.append(np.argmax(test_label[i] ))
        predicted_label.append(np.argmax(prediction[i] ))

    if print_detailed_results:
        print ("Confusion Matrix :")
        print (confusion_matrix(true_label, predicted_label))
        print ("Classification Report :")
        print (classification_report(true_label, predicted_label))

    return accuracy_score(true_label, predicted_label)

def CAM(att_type, x, y):

    if att_type == 'simple':
        m_dash = dot([x, y], axes=[2,2])
        m = Activation('softmax')(m_dash)
        h_dash = dot([m, y], axes=[2,1])
        return multiply([h_dash, x])

def featuresExtraction(dataset):
    global train_text, train_audio, train_video, train_label
    global valid_text, valid_audio, valid_video, valid_label
    global test_text, test_audio, test_video, test_label
    global max_segment_len

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/X_train.h5','r')
    X_train = hf['data'][:]
    split_into_features_train = np.split(X_train, [300, 305, 325], axis=2)
    train_text = split_into_features_train[0]
    train_audio = split_into_features_train[1]
    train_video = split_into_features_train[2]

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/X_test.h5','r')
    X_test = hf['data'][:]
    split_into_features_test = np.split(X_test, [300, 305, 325], axis=2)
    test_text = split_into_features_test[0]
    test_audio = split_into_features_test[1]
    test_video = split_into_features_test[2]

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/X_valid.h5','r')
    X_valid = hf['data'][:]
    split_into_features_valid = np.split(X_valid, [300, 305, 325], axis=2)
    valid_text = split_into_features_valid[0]
    valid_audio = split_into_features_valid[1]
    valid_video = split_into_features_valid[2]

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/y_train.h5','r')
    y_train = hf['data'][:]

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/y_test.h5','r')
    y_test = hf['data'][:]

    hf = h5py.File('/Users/tuyet/Desktop/Masterthesis/datasets/MOSI/y_valid.h5','r')
    y_valid = hf['data'][:]


    max_segment_len = train_text.shape[1]


    # 2 Classes
    # test_label = [[1, 0] if val < 0 else [0, 1] for val in y_test]
    # test_label = np.array(test_label)

    # train_label = [[1, 0] if val < 0 else [0, 1] for val in y_train]
    # train_label = np.array(train_label)

    # valid_label = [[1, 0] if val < 0 else [0, 1] for val in y_valid]
    # valid_label = np.array(valid_label)

    # 7 Classes

    y_train = np.round(y_train)
    train_label = to_categorical(y_train - y_train.min())

    y_test = np.round(y_test)
    test_label = to_categorical(y_test - y_test.min())

    y_valid = np.round(y_valid)
    valid_label = to_categorical(y_valid - y_valid.min())

# ================================================== simple 3 =========================================
def CIA_model(mode, filePath, dataset, attn_type='mmmu', drops=[0.7, 0.5, 0.5], r_units=300, td_units=100):

    runs = 1
    best_accuracy = 0

    for run in range(runs):

        drop0  = drops[0]
        drop1  = drops[1]
        r_drop = drops[2]
        in_test_label   = []

        # =============================================================================================
        # ============================================== IIM ==========================================

        in_text_audio      = Input(shape=(train_text.shape[1], train_text.shape[2]))
        encoded_text_audio = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_text_audio))
        encoded_text_audio = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_audio))
        encoded_text_audio = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_text_audio')(encoded_text_audio))
        decoded_text_audio = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_audio))
        decoded_text_audio = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_text_audio))
        output_text_audio  = TimeDistributed(Dense(train_audio.shape[2], activation='tanh'),name='output_text_audio')(decoded_text_audio)

        in_text_video      = Input(shape=(train_text.shape[1], train_text.shape[2]))
        encoded_text_video = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_text_video))
        encoded_text_video = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_video))
        encoded_text_video = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_text_video')(encoded_text_video))
        decoded_text_video = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_video))
        decoded_text_video = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_text_video))
        output_text_video  = TimeDistributed(Dense(train_video.shape[2], activation='tanh'),name='output_text_video')(decoded_text_video)

        in_audio_text      = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        encoded_audio_text = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_audio_text))
        encoded_audio_text = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_text))
        encoded_audio_text = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_audio_text')(encoded_audio_text))
        decoded_audio_text = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_text))
        decoded_audio_text = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_audio_text))
        output_audio_text  = TimeDistributed(Dense(train_text.shape[2], activation='tanh'),name='output_audio_text')(decoded_audio_text)

        in_audio_video      = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        encoded_audio_video = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_audio_video))
        encoded_audio_video = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_video))
        encoded_audio_video = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_audio_video')(encoded_audio_video))
        decoded_audio_video = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_video))
        decoded_audio_video = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_audio_video))
        output_audio_video  = TimeDistributed(Dense(train_video.shape[2], activation='tanh'),name='output_audio_video')(decoded_audio_video)

        in_video_text      = Input(shape=(train_video.shape[1], train_video.shape[2]))
        encoded_video_text = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_video_text))
        encoded_video_text = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_text))
        encoded_video_text = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_video_text')(encoded_video_text))
        decoded_video_text = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_text))
        decoded_video_text = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_video_text))
        output_video_text  = TimeDistributed(Dense(train_text.shape[2], activation='tanh'),name='output_video_text')(decoded_video_text)

        in_video_audio      = Input(shape=(train_video.shape[1], train_video.shape[2]))
        encoded_video_audio = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(in_video_audio))
        encoded_video_audio = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_audio))
        encoded_video_audio = Dropout(drop1)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_video_audio')(encoded_video_audio))
        decoded_video_audio = Dropout(drop1)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_audio))
        decoded_video_audio = Dropout(drop1)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_video_audio))
        output_video_audio  = TimeDistributed(Dense(train_audio.shape[2], activation='tanh'),name='output_video_audio')(decoded_video_audio)

        rnn_text_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_text_audio)
        rnn_text_audio    = Dropout(drop0)(rnn_text_audio)
        td_text_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text_audio))

        rnn_text_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_text_video)
        rnn_text_video    = Dropout(drop0)(rnn_text_video)
        td_text_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text_video))

        rnn_audio_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_audio_text)
        rnn_audio_text    = Dropout(drop0)(rnn_audio_text)
        td_audio_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio_text))

        rnn_audio_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_audio_video)
        rnn_audio_video    = Dropout(drop0)(rnn_audio_video)
        td_audio_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio_video))

        rnn_video_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_video_text)
        rnn_video_text    = Dropout(drop0)(rnn_video_text)
        td_video_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video_text))

        rnn_video_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_video_audio)
        rnn_video_audio    = Dropout(drop0)(rnn_video_audio)
        td_video_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video_audio))

        # =============================================================================================
        # ================================== single modality ==========================================

        in_text     = Input(shape=(train_text.shape[1], train_text.shape[2]))
        rnn_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_text)
        rnn_text    = Dropout(drop0)(rnn_text)
        td_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text))

        in_audio     = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        rnn_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_audio)
        rnn_audio    = Dropout(drop0)(rnn_audio)
        td_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio))

        in_video     = Input(shape=(train_video.shape[1], train_video.shape[2]))
        rnn_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_video)
        rnn_video    = Dropout(drop0)(rnn_video)
        td_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video))

        # =============================================================================================
        # ======================================= Mean and CAM ========================================

        original_concat_text_audio = concatenate([td_text, td_audio])
        original_concat_td_text_audio = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_text_audio))
        modified_avg_text_audio = Average()([td_text_audio,td_audio_text])
        original_modified_ta_att = CAM('simple', original_concat_td_text_audio, modified_avg_text_audio)
        modified_original_ta_att = CAM('simple', modified_avg_text_audio, original_concat_td_text_audio)

        original_concat_audio_video = concatenate([td_audio, td_video])
        original_concat_td_audio_video = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_audio_video))
        modified_avg_audio_video = Average()([td_audio_video,td_video_audio])
        original_modified_av_att = CAM('simple', original_concat_td_audio_video, modified_avg_audio_video)
        modified_original_av_att = CAM('simple', modified_avg_audio_video, original_concat_td_audio_video)

        original_concat_video_text = concatenate([td_video, td_text])
        original_concat_td_video_text = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_video_text))
        modified_avg_video_text = Average()([td_video_text,td_text_video])
        original_modified_vt_att = CAM('simple', original_concat_td_video_text, modified_avg_video_text)
        modified_original_vt_att = CAM('simple', modified_avg_video_text, original_concat_td_video_text)

        merged = concatenate([original_modified_ta_att,modified_original_ta_att,original_modified_av_att,modified_original_av_att,original_modified_vt_att,modified_original_vt_att])

        merged  = Dense(td_units, activation='relu')(merged)
        merged  = MeanOverTime()(merged)
        final_output  = Dense(7     , activation='softmax', name='final_output')(merged)

        # =============================================================================================
        # ======================================= Model ===============================================

        model = Model(inputs=[in_text,in_audio,in_video,in_text_audio,in_text_video,in_audio_text,in_audio_video,in_video_text,in_video_audio],
                      outputs=[output_text_audio,output_text_video,output_audio_text,output_audio_video,output_video_text,output_video_audio,final_output])
        model.compile(loss=['mse','mse','mse','mse','mse','mse','categorical_crossentropy'], sample_weight_mode='None', optimizer='adam', metrics=['acc'])
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        path   = 'weights/'+ dataset + '_' +str(filePath)+ '_' +str(run)+'.hdf5'
        check1 = EarlyStopping(monitor='val_final_output_loss', patience=20)
        check2 = ModelCheckpoint(path, monitor='val_final_output_acc', verbose=1, save_weights_only=True,  save_best_only=True, mode='max')


        history = model.fit([train_text,train_audio,train_video,train_text,train_text,train_audio,train_audio,train_video,train_video],
                            [train_audio,train_video,train_text,train_video,train_text,train_audio,train_label],
                            epochs=50, #auf 50 zurück machen!! 
                            batch_size=16, # auf 16 !!
                            shuffle=True,
                            callbacks=[check1, check2],
                            validation_data=([valid_text,valid_audio,valid_video,valid_text,valid_text,valid_audio,valid_audio,valid_video,valid_video], [valid_audio,valid_video,valid_text,valid_video,valid_text,valid_audio,valid_label]),
                            verbose=1)

        model.load_weights(path)
        result = model.predict([test_text,test_audio,test_video,test_text,test_text,test_audio,test_audio,test_video,test_video])

        np.ndarray.dump(result[len(in_test_label)],open('results/'+ dataset + '_7 classes' +'_' +str(filePath)+'_'+str(run)+'.np', 'wb'))
        best_accuracy = calculate_accuracy(result[-1], test_label)

        open('results/'+dataset + '_7 classes' +'_' + modality  +'.txt', 'a').write(filePath + ', accuracy: ' + str(best_accuracy) + '\n'*2)
        print('best accuracy is: ', best_accuracy)
        plt.plot(history.history['final_output_acc'])
        plt.plot(history.history['val_final_output_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()    
featuresExtraction('MOSI')
for drop in [0.5]:
    for rdrop in [0.3]:                                 # MOUD, MOSI, YOUTUBE, MMMO : 0.3, MOSEI 0.5 
        for r_units in [50]:                            # MOUD, MOSI, YOUTUBE 50, MMMo 200, MOSEI 100
            for td_units in [50]:                      # MOSI 50, MOUD 50, YOUTUBE 200, MOSEI 100, MMMO 100
                attn_type = 'mmmu'
                modalities = ['text','audio','video']
                for i in range(1):
                    for mode in itertools.combinations(modalities, 3):
                        modality = '_'.join(mode)
                        print ('\n',modality)
                        filePath  = modality + '_' + attn_type + '_' + str(drop) + '_' + str(drop) + '_' + str(rdrop) + '_' + str(r_units) + '_' + str(td_units)
                        CIA_model(mode, filePath, attn_type=attn_type, drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units, dataset='MOSI')
