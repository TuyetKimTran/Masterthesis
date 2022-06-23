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

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/MOSEI/mosei_senti_data.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1').astype(np.uint8)

    train_video = data['train']['vision']
    train_audio = data['train']['audio']
    train_text  = data['train']['text']
    train_label = data['train']['labels']

    test_video = data['test']['vision']
    test_audio = data['test']['audio']
    test_text  = data['test']['text']
    test_label = data['test']['labels']


    valid_video = data['valid']['vision']
    valid_audio = data['valid']['audio']
    valid_text  = data['valid']['text']
    valid_label = data['valid']['labels']


    train_label = [[1, 0] if val < 0 else [0, 1] for val in train_label]
    train_label = np.array(train_label)

    test_label = [[1, 0] if val < 0 else [0, 1] for val in test_label]
    test_label = np.array(test_label)

    valid_label = [[1, 0] if val < 0 else [0, 1] for val in valid_label]
    valid_label = np.array(valid_label)

# ================================================== simple 3 =========================================
def CIA_model(mode, filePath, dataset, attn_type='mmmu', drops=[0.7, 0.5, 0.5], r_units=300, td_units=100):

    runs = 1
    best_accuracy = 0

    for run in range(runs):

        drop0  = drops[0]
        drop1  = drops[1]
        r_drop = drops[2]

        in_merged = []
        in_model  = []
        out_model = []
        in_loss   = []
        in_train  = []
        in_valid  = []
        in_test   = []
        in_train_label  = []
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
        final_output  = Dense(2     , activation='softmax', name='final_output')(merged)

        # =============================================================================================
        # ======================================= Model ===============================================

        model = Model(inputs=[in_text,in_audio,in_video,in_text_audio,in_text_video,in_audio_text,in_audio_video,in_video_text,in_video_audio],outputs=[output_text_audio,output_text_video,output_audio_text,output_audio_video,output_video_text,output_video_audio,final_output])
        model.compile(loss=['mse','mse','mse','mse','mse','mse','categorical_crossentropy'], sample_weight_mode='None', optimizer='adam', metrics=['acc'])

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

        np.ndarray.dump(result[len(in_test_label)],open('results/'+ dataset + '_' +str(filePath)+'_'+str(run)+'.np', 'wb'))
        best_accuracy = calculate_accuracy(result[-1], test_label)

        open('results/'+dataset +'_' + modality  +'.txt', 'a').write(filePath + ', accuracy: ' + str(best_accuracy) + '\n'*2)
        print('best accuracy is: ', best_accuracy)
        plt.plot(history.history['final_output_acc'])
        plt.plot(history.history['val_final_output_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()    
featuresExtraction('MOSEI')
for drop in [0.5]:
    for rdrop in [0.5]:       # MOUD, MOSI, YOUTUBE, MMMO : 0.3, MOSEI 0.5
        for r_units in [100]:  # MOUD, MOSI, YOUTUBE 50, MMMo 200, MOSEI 100
            for td_units in [100]: # MOSEI MMMO 100, YOUTUBE 200, MOUD MOSI 50 
                attn_type = 'mmmu'
                modalities = ['text','audio','video']
                for i in range(1):
                    for mode in itertools.combinations(modalities, 3):
                        modality = '_'.join(mode)
                        print ('\n',modality)
                        filePath  = modality + '_' + attn_type + '_' + str(drop) + '_' + str(drop) + '_' + str(rdrop) + '_' + str(r_units) + '_' + str(td_units)
                        CIA_model(mode, filePath, attn_type=attn_type, drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units, dataset='MOSEI')
