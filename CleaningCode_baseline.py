from keras.utils.vis_utils import plot_model

# ================================================== simple 3 =========================================
def CIA_model(mode, filePath, dataset,classNo, drops=[0.7, 0.5, 0.5], r_units=300, td_units=100):

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
        final_output  = Dense(3     , activation='softmax', name='final_output')(merged)

        # =============================================================================================
        # ======================================= Model ===============================================

        model = Model(inputs=[in_text,in_audio,in_video,in_text_audio,in_text_video,in_audio_text,in_audio_video,in_video_text,in_video_audio],outputs=[output_text_audio,output_text_video,output_audio_text,output_audio_video,output_video_text,output_video_audio,final_output])
        model.compile(loss=['mse','mse','mse','mse','mse','mse','categorical_crossentropy'], sample_weight_mode='None', optimizer='adam', metrics=['acc'])

        path   = '/content/drive/MyDrive/weights/' + dataset + str(filePath)+'_'+str(run)+'.hdf5'
        check1 = EarlyStopping(monitor='val_final_output_loss', patience=20)
        check2 = ModelCheckpoint(path, monitor='val_final_output_acc', verbose=1, save_weights_only=True,  save_best_only=True, mode='max')
        %load_ext autotime


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

        np.ndarray.dump(result[len(in_test_label)],open('/content/drive/MyDrive/results/' + dataset +str(filePath)+'_'+str(run)+'.np', 'wb'))
        best_accuracy = calculate_accuracy(result[-1], test_label)

        open('/content/drive/MyDrive/results/'+ dataset + '_' + modelType + '_' + modality+'.txt', 'a').write(filePath + ', accuracy: ' + str(best_accuracy) + '\n'*2)        # plt.figure(1)

dataset = 'YOUTUBE'
modelType = 'Baseline'
classNo = 3
featuresExtraction(dataset,classNo)   # hier ersetzen
for drop in [0.5]:
    for rdrop in [0.3]:               # MOUD, MOSI, YOUTUBE, MMMO : 0.3, MOSEI 0.5 
        for r_units in [50]:          # MOUD, MOSI, YOUTUBE 50, MMMo 200, MOSEI 100
            for td_units in [200]:    # MOSI 50, MOUD 50, YOUTUBE 200, MOSEI 100, MMMO 100
                modalities = ['text','audio','video']
                for i in range(1):
                    for mode in itertools.combinations(modalities, 3):
                        modality = '_'.join(mode)
                        print ('\n',modality)
                        filePath  = modality 
                        CIA_model(mode, filePath, drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units, dataset=dataset, classNo=classNo)
