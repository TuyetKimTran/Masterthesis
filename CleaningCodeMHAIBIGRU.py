
from keras_multi_head import MultiHeadAttention
import time

import matplotlib.pyplot as plt

def CIA_model(dataset, classNo, drop):
    modalitys = 'text_audio_video'
    runs = 1
    best_accuracy = 0

    for run in range(runs):

        in_test_label   = []


        # ============================================== IIM ==========================================
        # Encoded Representation is passed on 
        text_shape         = Input(shape=(train_text.shape[1], train_text.shape[2]))
        encoded_text_audio = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(text_shape))
        encoded_text_audio = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_audio))
        encoded_text_audio = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_text_audio')(encoded_text_audio)) # Encoded Representation
        decoded_text_audio = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_audio))
        decoded_text_audio = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_text_audio))
        output_text_audio  = TimeDistributed(Dense(train_audio.shape[2], activation='tanh'),name='output_text_audio')(decoded_text_audio)


        encoded_text_video = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(text_shape))
        encoded_text_video = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_video))
        encoded_text_video = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_text_video')(encoded_text_video))
        decoded_text_video = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_text_video))
        decoded_text_video = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_text_video))
        output_text_video  = TimeDistributed(Dense(train_video.shape[2], activation='tanh'),name='output_text_video')(decoded_text_video)

        audio_shape      = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        encoded_audio_text = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(audio_shape))
        encoded_audio_text = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_text))
        encoded_audio_text = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_audio_text')(encoded_audio_text))
        decoded_audio_text = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_text))
        decoded_audio_text = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_audio_text))
        output_audio_text  = TimeDistributed(Dense(train_text.shape[2], activation='tanh'),name='output_audio_text')(decoded_audio_text)

        encoded_audio_video = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(audio_shape))
        encoded_audio_video = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_video))
        encoded_audio_video = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_audio_video')(encoded_audio_video))
        decoded_audio_video = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_audio_video))
        decoded_audio_video = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_audio_video))
        output_audio_video  = TimeDistributed(Dense(train_video.shape[2], activation='tanh'),name='output_audio_video')(decoded_audio_video)

        video_shape      = Input(shape=(train_video.shape[1], train_video.shape[2]))
        encoded_video_text = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(video_shape))
        encoded_video_text = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_text))
        encoded_video_text = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_video_text')(encoded_video_text))
        decoded_video_text = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_text))
        decoded_video_text = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_video_text))
        output_video_text  = TimeDistributed(Dense(train_text.shape[2], activation='tanh'),name='output_video_text')(decoded_video_text)

        encoded_video_audio = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(video_shape))
        encoded_video_audio = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_audio))
        encoded_video_audio = Dropout(drop)(TimeDistributed(Dense(1*td_units, activation='tanh'),name='encoded_video_audio')(encoded_video_audio))
        decoded_video_audio = Dropout(drop)(TimeDistributed(Dense(2*td_units, activation='tanh'))(encoded_video_audio))
        decoded_video_audio = Dropout(drop)(TimeDistributed(Dense(3*td_units, activation='tanh'))(decoded_video_audio))
        output_video_audio  = TimeDistributed(Dense(train_audio.shape[2], activation='tanh'),name='output_video_audio')(decoded_video_audio)



        # ================================== Multi-head Attention ==========================================
        """Take the encoded Representation from IIM Module and perform Multi-Head Attention"""

        td_text_audio     = MultiHeadAttention(head_num =5)(encoded_text_audio)
        td_text_video     = MultiHeadAttention(head_num =5)(encoded_text_video)
        td_audio_text     = MultiHeadAttention(head_num =5)(encoded_audio_text)
        td_audio_video    = MultiHeadAttention(head_num =5)(encoded_audio_video)
        td_video_text     = MultiHeadAttention(head_num =5)(encoded_video_text)
        td_video_audio    = MultiHeadAttention(head_num =5)(encoded_video_audio)

        # ================================== single modality ==========================================
        """Parallel process: Take Single modalitys and perform Multi-Head Attention"""

        td_text      = MultiHeadAttention(head_num =5)(text_shape)
        td_audio     = MultiHeadAttention(head_num =2)(audio_shape)
        td_video     = MultiHeadAttention(head_num =5)(video_shape)


        # ======================================= Mean, Bi-modal Interaction, and CAM ========================================
        """Bi-modal Interaction through Pairwise Concatenation and Fully connected layer: BI 
            Mean through taking average() of the output of the MHA 
            Context-aware Attention Module thorugh function CAM (see above), Input is the Mean and the Bi-modal Interaction of the same two Modalitys
        """
        
        # Text and Audio
        concat_TA = concatenate([td_text, td_audio])                                                  # Concatenation          
        bi_TA = Dropout(drop)(TimeDistributed(Dense(td_units, activation='relu'))(concat_TA))         # + FC            = BI  
        mean_TA = Average()([td_text_audio,td_audio_text])                                            # Mean 
        cam_TA = CAM(bi_TA, mean_TA)                                                                  # CAM
        cam_AT = CAM(mean_TA, bi_TA)


        # Audio and Video
        concat_AV = concatenate([td_audio, td_video])
        bi_AV = Dropout(drop)(TimeDistributed(Dense(td_units, activation='relu'))(concat_AV))
        mean_AV = Average()([td_audio_video,td_video_audio])
        cam_AV = CAM(bi_AV, mean_AV)
        cam_VA = CAM(mean_AV, bi_AV)

        # Video and Text
        concat_VT = concatenate([td_video, td_text])
        bi_VT = Dropout(drop)(TimeDistributed(Dense(td_units, activation='relu'))(concat_VT))
        mean_VT = Average()([td_video_text,td_text_video])
        cam_VT = CAM(bi_VT, mean_VT)
        cam_TV = CAM(mean_VT, bi_VT)

        # ======================================= Concatenate ========================================
        merged = concatenate([cam_TA,cam_AT,cam_AV,cam_VA,
                              cam_VT,cam_TV])

        merged  = Dense(td_units, activation='relu')(merged)
        merged  = MeanOverTime()(merged)
        final_output  = Dense(classNo, activation='softmax', name='final_output')(merged)   # hier anpassen wieviele outputs man hat!!
        # ======================================= Model ===============================================


        model = Model(inputs=[text_shape,audio_shape,video_shape,text_shape,text_shape,audio_shape,audio_shape,video_shape,video_shape],outputs=[output_text_audio,output_text_video,output_audio_text,output_audio_video,output_video_text,output_video_audio,final_output])
        model.compile(loss=['mse','mse','mse','mse','mse','mse','categorical_crossentropy'], sample_weight_mode='None', optimizer='adam', metrics=['acc'])
        # plot_model(model, to_file='model_plot_MHA_instead_BiGru.png', show_shapes=True, show_layer_names=True)

        # vorher metrics=['accuracy'])

        path   = '/content/drive/MyDrive/weights/' + dataset + str(modalitys)+'_'+str(run)+'.hdf5'
        check1 = EarlyStopping(monitor='val_final_output_loss', patience=20)
        check2 = ModelCheckpoint(path, monitor='val_final_output_acc', verbose=1, save_weights_only=True,  save_best_only=True, mode='max')


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

        np.ndarray.dump(result[len(in_test_label)],open('/content/drive/MyDrive/results/' + dataset +str(modalitys)+'_'+str(run)+'.np', 'wb'))
        best_accuracy = calculate_accuracy(result[-1], test_label, True)
        print('best accuracy is: ', best_accuracy)

        open('/content/drive/MyDrive/results/'+ dataset + '_' + modelType + modality+'.txt', 'a').write(modalitys + ', accuracy: ' + str(best_accuracy) + '\n'*2)        # plt.figure(1)
        plt.plot(history.history['final_output_acc'])
        plt.plot(history.history['val_final_output_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()   
        %load_ext autotime

dataset = 'MMMO'
modelType = 'MHA_instead_BiGRU'    
classNo = 2  
drop = 0.5 
td_units = 100
featuresExtraction(dataset, classNo)   # hier ersetzen
CIA_model(mode,dataset=dataset, classNo=classNo, td_units=td_units)
