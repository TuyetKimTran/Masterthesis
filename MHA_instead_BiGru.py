import numpy as np, json
import pickle, sys, argparse
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from keras.utils.layer_utils import get_source_inputs
from kutilities.layers import MeanOverTime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
import matplotlib.pyplot as plt
import keras
from tensorflow import keras
from keras_multi_head import MultiHeadAttention
from keras.utils.vis_utils import plot_model





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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#=============================================================
import tensorflow as tf
from keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
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

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/'+ dataset+ '/covarep_train.p', 'rb') as f:
        train_audio = pickle.load(f, encoding='latin1')
    train_audio = train_audio/np.max(train_audio)

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/covarep_valid.p', 'rb') as f:
        valid_audio = pickle.load(f, encoding='latin1')
    valid_audio = valid_audio/np.max(valid_audio)


    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/covarep_test.p', 'rb') as f:
        test_audio = pickle.load(f, encoding='latin1')
    test_audio  = test_audio/np.max(test_audio)

    with open ('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/facet_train.p', 'rb') as f:
        train_video = pickle.load(f, encoding='latin1')
    train_video = train_video/np.max(train_video)

    with open ('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/facet_valid.p', 'rb') as f:
        valid_video = pickle.load(f, encoding='latin1')
    valid_video = valid_video/np.max(valid_video)

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/facet_test.p', 'rb') as f:
        test_video  = pickle.load(f, encoding='latin1')
    test_video  = test_video/np.max(test_video)

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/text_train.p', 'rb') as f:
        train_text  = pickle.load(f, encoding='latin1')
    train_text  = train_text/np.max(train_text)

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/text_valid.p', 'rb') as f:
        valid_text  = pickle.load(f, encoding='latin1')
    valid_text  = valid_text/np.max(valid_text)

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/text_test.p', 'rb') as f:
        test_text   = pickle.load(f, encoding='latin1')
    test_text   = test_text/np.max(test_text)

    max_segment_len = train_text.shape[1]

    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/y_train.p', 'rb') as f:
        train_label = pickle.load(f, encoding='latin1')
    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/y_valid.p', 'rb') as f:
        valid_label = pickle.load(f, encoding='latin1')
    with open('/Users/tuyet/Desktop/Masterthesis/datasets/' + dataset + '/y_test.p', 'rb') as f:
        test_label  = pickle.load(f, encoding='latin1')

    
# ================================================== simple 3 =========================================
def CIA_model(mode, filePath, dataset, drops=[0.7, 0.5, 0.5], r_units=300, td_units=100):

    runs = 1
    best_accuracy = 0

    for run in range(runs):

        drop0  = drops[0]
        drop1  = drops[1]
        r_drop = drops[2]

        in_test_label   = []

        # =============================================================================================
        # ============================================== IIM ==========================================
                                        # train_text shape (169, 21, 300)
                                        # Input(shape=(21,300))
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


        # att_layer_text          # td_text_audio
        td_text_audio     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_text_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_text_audio)
        # rnn_text_audio    = Dropout(drop0)(rnn_text_audio)
        # td_text_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text_audio))

        td_text_video     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_text_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_text_video)
        # rnn_text_video    = Dropout(drop0)(rnn_text_video)
        # td_text_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text_video))

        td_audio_text     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_audio_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_audio_text)
        # rnn_audio_text    = Dropout(drop0)(rnn_audio_text)
        # td_audio_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio_text))

        td_audio_video     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_audio_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_audio_video)
        # rnn_audio_video    = Dropout(drop0)(rnn_audio_video)
        # td_audio_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio_video))

        td_video_text     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_video_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_video_text)
        # rnn_video_text    = Dropout(drop0)(rnn_video_text)
        # td_video_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video_text))

        td_video_audio     = MultiHeadAttention(head_num =8)(encoded_text_audio)
        # rnn_video_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(encoded_video_audio)
        # rnn_video_audio    = Dropout(drop0)(rnn_video_audio)
        # td_video_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video_audio))

        # =============================================================================================
        # ================================== single modality ==========================================

        td_text     = MultiHeadAttention(head_num =8)(encoded_text_audio)

        in_text     = Input(shape=(train_text.shape[1], train_text.shape[2]))
        # rnn_text    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_text)
        # rnn_text    = Dropout(drop0)(rnn_text)
        # td_text     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_text))

        td_audio     = MultiHeadAttention(head_num =8)(encoded_text_audio)

        in_audio     = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        # rnn_audio    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_audio)
        # rnn_audio    = Dropout(drop0)(rnn_audio)
        # td_audio     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_audio))

        td_video     = MultiHeadAttention(head_num =8)(encoded_text_audio)

        in_video     = Input(shape=(train_video.shape[1], train_video.shape[2]))
        # rnn_video    = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(in_video)
        # rnn_video    = Dropout(drop0)(rnn_video)
        # td_video     = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_video))

        # =============================================================================================
        # ======================================= Mean and CAM ========================================

        # At first all the three modalities are passed through a separate Bi-GRU. Then, pair-wise concatenation is performed 
        # over the output of Bi-GRU and passed through a fully-connected layer to extract the bi-modal interaction (BI). 
        

        # Input of CAM : CamTA = MeanTA and BiTA 

        # TEXT und AUDIO

        
        original_concat_text_audio = concatenate([td_text, td_audio])
        original_concat_td_text_audio = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_text_audio))
        
        #MeanTa
        modified_avg_text_audio = Average()([td_text_audio,td_audio_text])
        original_modified_ta_att = CAM('simple', original_concat_td_text_audio, modified_avg_text_audio)
        modified_original_ta_att = CAM('simple', modified_avg_text_audio, original_concat_td_text_audio)


        # AUDIO und VIDEO 
        original_concat_audio_video = concatenate([td_audio, td_video])
        original_concat_td_audio_video = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_audio_video))
        
        
        modified_avg_audio_video = Average()([td_audio_video,td_video_audio])
        original_modified_av_att = CAM('simple', original_concat_td_audio_video, modified_avg_audio_video)
        modified_original_av_att = CAM('simple', modified_avg_audio_video, original_concat_td_audio_video)

        # VIDEO und TEXT 
        original_concat_video_text = concatenate([td_video, td_text])
        original_concat_td_video_text = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(original_concat_video_text))
        modified_avg_video_text = Average()([td_video_text,td_text_video])
        original_modified_vt_att = CAM('simple', original_concat_td_video_text, modified_avg_video_text)
        modified_original_vt_att = CAM('simple', modified_avg_video_text, original_concat_td_video_text)



        # Im Diagramm Schritt Concatenate 
        merged = concatenate([original_modified_ta_att,modified_original_ta_att,original_modified_av_att,modified_original_av_att,
                              original_modified_vt_att,modified_original_vt_att])




        merged  = Dense(td_units, activation='relu')(merged)
        merged  = MeanOverTime()(merged)
        final_output  = Dense(3, activation='softmax', name='final_output')(merged)   # hier anpassen wieviele outputs man hat!!

        # =============================================================================================
        # ======================================= Model ===============================================

        model = Model(inputs=[in_text,in_audio,in_video,in_text_audio,in_text_video,in_audio_text,in_audio_video,in_video_text,in_video_audio], # 3x single und die 6 combos
        outputs=[output_text_audio,output_text_video,output_audio_text,output_audio_video,output_video_text,output_video_audio,final_output]) # 6 combos + final output
        model.compile(loss=['mse','mse','mse','mse','mse','mse','categorical_crossentropy'], sample_weight_mode='None', optimizer='adam', metrics=['acc'])

        # vorher metrics=['accuracy'])

        path   = 'weights/'+str(filePath)+'_'+str(run)+'.hdf5'
        print(path)
        check1 = EarlyStopping(monitor='val_final_output_loss', patience=20)
        check2 = ModelCheckpoint(path, monitor='val_final_output_acc', verbose=1,save_best_only=True, save_weights_only=True, mode='max')


        history = model.fit([train_text,train_audio,train_video,train_text,train_text,train_audio,train_audio,train_video,train_video], 
                            [train_audio,train_video,train_text,train_video,train_text,train_audio,train_label],
                            epochs=50, #auf 50 zurück machen!! 
                            batch_size=16, # auf 16 !!
                            shuffle=True,
                            callbacks=[check1, check2],
                            validation_data=([valid_text,valid_audio,valid_video,valid_text,valid_text,valid_audio,valid_audio,valid_video,valid_video], 
                            [valid_audio,valid_video,valid_text,valid_video,valid_text,valid_audio,valid_label]),
                            verbose=1)

        model.load_weights(path)
        result = model.predict([test_text,test_audio,test_video,
                                test_text,test_text,test_audio,test_audio,test_video,test_video])
        # print('results', result)
        # print('results[-1]' , result[-1])
        # print('testlabel' , test_label)
        # print('results[0]', result[0])
        # result 
        np.ndarray.dump(result[len(in_test_label)],open('results/'+ dataset + '_' +str(filePath)+'_'+str(run)+'.np', 'wb'))
        best_accuracy = calculate_accuracy(result[-1], test_label, True)
        print('best accuracy is: ', best_accuracy)

        open('results/'+dataset +'_' + modality  +'.txt', 'a').write(filePath + ', accuracy: ' + str(best_accuracy) + '\n'*2)
        # plt.figure(1)
        # history_dict = history.history
        # history_dict.keys()
        # print(history.history.keys())
        # print(history.history)
        # plot_model(model, to_file='model_plot_MHA_Instead_BiGru.png', show_shapes=True, show_layer_names=True)

        plt.plot(history.history['final_output_acc'])
        plt.plot(history.history['val_final_output_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()    
featuresExtraction('YOUTUBE')          # YOUTUBE; MMMO or MOUD  
for drop in [0.5]:
    for rdrop in [0.3]:                     # MOUD, MOSI, YOUTUBE, MMMO : 0.3, MOSEI 0.5 
        for r_units in [50]:               # MOUD, MOSI, YOUTUBE 50, MMMo 200, MOSEI 100
            for td_units in [200]:   # MOSI 50, MOUD 50, YOUTUBE 200, MOSEI 100, MMMO 100
                modalities = ['text','audio', 'video']
                modification ="MHA_instead_BiGru"
                for i in range(1):
                    for mode in itertools.combinations(modalities, 3):
                        modality = '_'.join(mode)
                        print ('\n',modality)
                        filePath  = modification + '_'  + '_' + str(drop) + '_' + str(drop) + '_' + str(rdrop) + '_' + str(r_units) + '_' + str(td_units)
                        CIA_model(mode, filePath, dataset='YOUTUBE', drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units)


