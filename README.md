# Masterthesis

## Multimodal sentiment Anaylsis
The proposed model: Multi-head self-attention with Context-aware Attention model (MHCA) effectively capture the intra- and intermodality
dynamics of visual, audio, and text features in multimodal sentiment analysis. The MHCA model consists of three main components: a Context-aware Attention
(CAM) module, a fully connected layer with dropout (Dense Layer + Dropout), and a Multi-Head Attention (MHA) module.
 ![MHCA Model](/assets/images/MHCA_Proposed_Model.png)


## Results
To show, that the MHCA model is effectively using all three modalities for sentiment prediction,the MHCA model is evaluated on all possible input combinations, uni-modal (text only, audio only, video only), bi-modal (text and video, text and audio, audio and video), and tri-modal (text, audio, and video). The MHCA model performs best on all datasets when all three modalities are combined.
 ![Results unimodal vs Bimodal vs Trimodal](/assets\images\UniModalBiModalTrimodal.png)

The MHCA model was compared with state-of-the-art models, including MV-LST, BC-LSTM, TFN, MARN, MFN, MFM, CIA, and the reimplementation of CIA (R-
CIA), in terms of maximum accuracy and F1-score.
 ![Comparative Analysis results](/assets\images\UniModalBiModalTrimodal.png)


## Get started

1. Download or create a linking for  the necessary datasets and save them to your Google Drive.
	Youtube, MMMO, MOUD and MOSI
	You can find the datasets and the link here: https://drive.google.com/drive/folders/1IVgdjfRGSqnai45ksot7UZ5C-1xJBBWZ
2. Open the Python notebook **Multimodal sentiment Analysis.ipynb** on your Computer or in Google Colab. 

3. If you encounter an error, update an outdated modules by replacing the second line in the "layers.py" module: "from keras.engine.topology import Layer" to "from tensorflow.keras.layers import Layer". The error message looks like this:
 ![Error message](/assets/images/error_message.png)

4. Remember to change the directory in the second cell to the directory where you saved the datasets. This cell unzips the datasets.
5. The remaining cells contain functions and the model code. You can easily execute the code in these cells. First is the MHCA, and under the Expreiments section are models of the experiments with the MHCA model.


 ## Load Weights 

You can load the weights and history of the best model

1. The Weights and the history of the training of the the best model are saved in the "Weights_and_history" folder.
2. In the Python Notebook, navigate to the last section. Here you can adjust the *weights_path* variable and *history_path* for the desired model. 