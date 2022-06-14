
## List of papers (ordered based on the year of publication, descending)

[1]	C. Suman, S. Saha, A. Gupta, S. K. Pandey, and P. Bhattacharyya, “A multi-modal personality prediction system,” Knowledge-Based Systems, vol. 236, p. 107715, **2022**, doi: 10.1016/j.knosys.2021.107715.

- Facial, audio, text
- Facial and ambient features using Multi-task Cascaded Convolutional Neural Networks (VGGish CNN), ResNet
- audio features extracted by VGGish Convolutional Neural Networks (VGGish CNN)
- text features extracted by n-gram Convolutional Neural Networks (CNN)
- Concatenation: 
  1. Late fusion : averaging predictions from each modalities (only for comparison) 
  2. Fusion of feature vectors with deep multimodal network
  3. different attention techniques
    1) Context-aware interactive attention
    2) Weighted Attention 
    3) Contextual Inter-modal Attention (MMMU-BA) (GRUs)
    4) QKV (self-attention)
- Dataset: CHA-LEARN
Big five personality traits(OCEAN): Extraversion, Agreeableness, Conscientiousness, Neuroticism and Openness 


[2]	H. Salam, V. Manoranjan, J. Jiang, and O. Celiktutan, Learning Personalised Models for Automatic Self-Reported Personality Recognition, **2022**. [Online]. Available: https://proceedings.mlr.press/v173/salam22a

- existing models: focus on personality recognition without considering user's profile
  - e.g females assess their personality traits differently from males, females report higher Big fice scores 
  - e.g . measures on Conscientiousness and Neuroticism increased in young adult-
  hood, whereas Openness increased in adolescence but then decreased in old age
  - also culture (gestures are different used in some cultures)
  - each user profile, the proposed model learns individual neural network architectures for
  different modalities (i.e., visual, textual, and audio) and fuses them at the decision level.
- work with NAS - neural architecture search to learn different user profiles
- user profiling criteria: gender, age 
UDIVA dataset



[3]	K. El-Demerdash, R. A. El-Khoribi, M. A. Ismail Shoman, and S. Abdou, “Deep learning based fusion strategies for personality prediction,” Egyptian Informatics Journal, vol. 23, no. 1, pp. 47–53, **2022**, doi: 10.1016/j.eij.2021.05.004.

- personality prediction by _text_ 
- personality classification based on transfer learning of pre-trained Language models (LM) features and fusion techniques
  - pre-trained models ElMo, ULMFiT, BERT 
  - datalevel fusion:  "we apply data level fusion for two benchmark personality datasets to generate more reliable, precise, and helpful features than any single dataset."
  - classification level fusion: fixed rules method, each classifier has posterior probability, mean fusion method is best
- Evaluation by accuracy, outperform state-of-art average accuracy 
- Dataset: 
  - myPersonality: Facebook status posts 
		- Essays: volunteers writing 
		- (Label: Author's own questionare big 5 model) by Autognosis: diagnosis by the use of self, self-knowledge)
![image](https://user-images.githubusercontent.com/61424213/169425054-81c67d0d-975c-4a78-8366-312e9d4da13c.png)





[4]	Z. Pan, Z. Luo, J. Yang, and H. Li, “Multi-modal Attention for Speech Emotion Recognition,” Sep. **2020**. [Online]. Available: http://arxiv.org/pdf/2009.04107v1
- cLSTM-MMA : contextual long short term memory block multi modal attention mechanism - facilitates attention across 3 modalities, selectively fuse information 
- Features: Speech, Visual, Text
  - Speech: by OpenSmile toolkit
  - Visual: by 3D-CNNN pre-trained from human action recognition to extract their body language
  - Text: by Word2vec used to embed each word of an utterance's transcript into word2Vec vectors 
- Uses hybrid fusion (combination of early and late fusion) 
  - "early fusion methods do not outperform late fusion" 
  - MMA : hybrid good results as other fusion (early, late and speech-only as baseline) and is more compact (much fewer parameters) network
- IEMOCAP dataset: conversation, 10K videos - 5 minutes, for emotions (6)



[6]	Y. Li et al., “CR-Net: A Deep Classification-Regression Network for Multimodal Apparent Personality Analysis,” Int J Comput Vis, vol. 128, no. 12, pp. 2763–2780, **2020**, doi: 10.1007/s11263-020-01309-y.

- Classification-Regression network(trait classification, regression later) , big 5 traits, video, audio and text 
  - video: separated into global and local cues
  - audio and transcription (text) early fused 
  - all three fused for final prediction
- new log function : regression-to-the-mean problem 
- good related-work section about personality predication 
- "most of previous models rely on finetuning of existing pre-trained models"
- why they use regression: 
  - " score of personality traits is a real number that requires a precision up to 4 decimals according to recent publications"
  - "Unlike ordinal regression techniques, which aggregate ranking classification results for final prediction, we keep a regression loss to obtain accurate personality predictions."
"we sum up the weighted features used for classification as regression input. In this way, features for classification are also considered by the regressor, benefiting the final regression even in cases where the classification outputs may be wrong."
- First impressions v2 dataset (extension of first impressions v1 dataset) 


[7]	W. Wang, D. Tran, and M. Feiszli, “What Makes Training Multi-Modal Classification Networks Hard?,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 12692–12702.
- Paper to get some background to choose/not choose multi-modal classification, overfitting problem occurs 
- information about using Gradient-Blending to address problems that occur using multi-modal networks: overfitting because increased capacity, overfitting because different modalities overfit and generalize at different rates -> training jointly with single optimization strategy does not work

[8]	D. S. Chauhan, M. S. Akhtar, A. Ekbal, and P. Bhattacharyya, “Context-aware Interactive Attention for Multi-modal Sentiment and Emotion Analysis,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), **2019**, pp. 5647–5657. [Online]. Available: https://aclanthology.org/D19-1566
- focus on _emotion_ prediction with (CIA), more information about CIA which is missing in the Paper above
- to get an impression of how a CIA is built.[Link to their code](https://github.com/DushyantChauhan/EMNLP-19-IIM)
- Dataset 5: 
  - Youtube opinion dataset: 47 videos of people expressing their opinions about a variety of topics (reviews)
  - Moud. 79 product review videos, Spanish 
  - ICT-MMMO extension of YouTube to 340 
  - CMU-MOSI 2199 opinion video clip, sentiments
  - CMU-MOSEI 3229 videos, 6 emotions 
- [access dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)
- probably not so relevant - information only for CIA which also used for personality prediction

[9]	J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” Oct. **2018**. [Online]. Available: http://arxiv.org/pdf/1810.04805v2
- Focus: _Speech_, NLP
- until then, only unidirectional models but with bidirectionally training, more sense of language context
- learns contextual relations, context of word based on all surrounding(neighbors)
- state of the art for language
- [useful link for Bert explanation/summary](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270?gi=1e803a048e6b)


[10]	X.-S. Wei, C.-L. Zhang, H. Zhang, and J. Wu, “Deep Bimodal Regression of Apparent Personality Traits from Short Video Sequences,” IEEE Trans. Affective Comput., vol. 9, no. 3, pp. 303–315, **2018**, doi: 10.1109/TAFFC.2017.2762299.
- Winner Team NJU_Lambda of competion ChaLearn V1 First impressions 
- visual + audio modality, personality prediction, big 5 , ChaLearn 
- deep bimodal regression framework
- Visual modality: pre-trained CNN, audio modality: linear regressor 
- [Code](https://github.com/tzzcl/ChaLearn-APA-Code)

[11]	H. Kaya, F. Gurpinar, and A. A. Salah, “Multi-modal Score Fusion and Decision Trees for Explainable Automatic Job Candidate Screening from Video CVs,” in **2017** IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Honolulu, HI, USA, 2017, pp. 1651–1659.
- Winner(1.st place) of First impression v2 dataset coopetition CVPR **2017** ChaLearn Job Candidate Screening Coopetition.
-  _Features_: face, scene, acoustic
  - Face features: Supervised Descent method (SDM), pretrained VGG-face network (already optimized for face recognition on large set of faces) over entire video segment 
  - _Scene features_: ambient (umgebung) features with trained VGG-VD 19 network on  ILSCRC dataset extracted from first image of each video 
  - Acoustic features: openSmile tool  popular to extract acoustic features, configuration same as baseline of another Challenge (interspeech) since most effective for personality recognition
- _Prediction_ : Extreme Learning machine classifiers (ELM) to evaluate each Chanel 
  - ELM: fast learning method for Single hidden layer Feedforward Networks, no backpropagation
- multi-modality through multi-level fusion by ensemble of Decision Trees (random Forests)
- prediction of interview variable: final predictions from the RF model are binarized by thresholding each dimension at corresponding
training set mean
- [github Code](https://github.com/frkngrpnr/jcs)


[12]	Y. Güçlütürk, U. Güçlü, M. A. J. van Gerven, and R. van Lier, “Deep Impression: Audiovisual Deep Residual Networks for Multimodal Apparent Personality Trait Recognition,” vol. 9915, no. 1, pp. 349–358, **2016**, doi: 10.1007/978-3-319-49409-8_28.

- dataset: ChaLearn First Impressions Challenge (3.rd place) V1
- from videos - no feature engineering or visual analysis, big 5 
- simple end-to-end neural network 
- _Features_:
- _network_ : auditory stream 17 layer deep residual network, visual stream 17 layer deep residual network, audiovisual stream: fully connected layer 
- valuation Accuracy ~ 91 %
- auditory stream, visual stream, audiovisual stream of fully connected layer (did not mention if early or late fusion but the way I understood the fusion, it should be late fusion) 
- has [code](https://github.com/yagguc/deep_impression)

[15]	C. Sarkar, S. Bhatia, A. Agarwal, and J. Li, “Feature Analysis for Computational Personality Recognition Using YouTube Personality Data set,” in Proceedings of the **2014** ACM Multi Media on Workshop on Computational Personality Recognition - WCPR '14, Orlando, Florida, USA, 2014, pp. 11–14.

- _Dataset_ : YouTube personality data set
- rather old paper, but gives a short insight about the beginnings of working multimodally when it comes to personality traits + it does consider not only audio, visual and text as modality:
  - paper more focus on also other features (audio-visual, text, word statistics, sentiment features of text, demographic feature (gender) and for each personality type points out which type fits/ does not fit most.  
  - Extraversion: “result proves the logic behind AV and other nonverbal features being the greatest contributors in classifying this personality type"
  - Agreeableness: “sentiment features such as negative word count are significant"
  - Conscientiousness: “their personality is reflected from their gestures and energy as compared to the actual words they speak."
  - Emotional stability: best classified using negative word count, negative sentiment score and other", "gender contributes significantly while text has least significance in the classification of emotional stability class."
  - Openess: "text features play an important role", "audio-visual features are the least significant for classifying this personality type. This might be because, people with this personality types are not restrictive about the words they speak or the way they speak."
-  logistic regression  
- Evaluation: Precision, Recall and F score 


## List of survey

[1]	X. Zhao, Z. Tang, and S. Zhang, “Deep Personality Trait Recognition: A Survey,” Front. Psychol., vol. 13, **2022**, doi: 10.3389/fpsyg.2022.839619.

- best and most recent survey i found
- survey on existing personality trait recognition methods
- "first attempt to present a comprehensive review covering both single and multimodal personality trait analysis related to hand-crafted and deep learning-based feature extraction algorithms"
  - single modality: audio, visual, text, etc.
  - multimodal: bimodal and trimodal modalities 
- challenges and opportunities 
- Divided into
  - Review of datasets 
  - Review of Deep learning techniques 
  - Fusion 

![image](https://user-images.githubusercontent.com/61424213/169427776-e1f69d14-b29e-4f3f-9288-1d3566cb3fd0.png)


[2]	J. C. S. J. Junior et al., “First Impressions: A Survey on Vision-Based Apparent Personality Trait Analysis,” Apr. **2018**. [Online]. Available: http://arxiv.org/pdf/1804.08046v3

-survey focus on single visual modality - analyzing personality from visual instead speech and text which was until 2018



## List of datasets 
![image](https://user-images.githubusercontent.com/61424213/169427751-66dd7268-f212-4554-8a17-0b94b32076d9.png)



### ChaLearn First impressions V1 & V2  2016 / 2017 
- two versions: V1, V2 : V2 same as V1 + variable "job interview" : indicating whether the person should be invited or not to a job interview
- 2016 competition goal: automatically evaluate Big 5
- 2017 Coopetition:  predict interview variable and justify explanation 
- 10.000 human centered short video sequences, 15s duration (6.000 train, 2.000 test, 2.000 validation), 41.6 hours, ~ 4.5 millions frames 
- Big five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism and Openness [0,1]  six annotations
- pairwise comparisons between video avoiding bias
- one transcription as single dictionary: 
keys = names of the videos, values = transcriptions
- annotation: dictionary of dictionaries :  the keys of the outer dictionary are the names of the annotations and their values are dictionaries.;  keys of the inner dictionaries are the names of the videos and their values are the actual annotations corresponding to the keys of the outer dictionaries
- [dataset link](https://chalearnlap.cvc.uab.cat/dataset/20/description/)
[presentation: results, dataset](https://sergioescalera.com/wp-content/uploads/2016/10/corneanu.pdf)

![image](https://user-images.githubusercontent.com/61424213/169427861-2042ef39-9cc4-46a5-a107-e6258ff8d2b7.png)


#### ChaLearn First impressions V2
-  consist of
	- Quantitative competition (first stage). predict "invite for interview" variable
 	- Qualitative coopetition (second stage) justify explanation
- [Code][Winners of 2017](https://github.com/frkngrpnr/jcs), [paper] Multi-modal Score Fusion and Decision Trees for Explainable Automatic Job Candidate Screening from Video CVs 
- [Paper from the organizer of the challenge. Gives insights/information about the dataset and challenge]: Explaining First Impressions: Modeling, Recognizing, and Explaining Apparent Personality from Videos:  introduced the interview variable because 
- V2 of the dataset added interview variable because
  - job recommendation systems/automating recruitment, to have some explanatory mechanism 
  - decision support machines
- [Paper: Design of an Explainable Machine Learning Challenge for Video Interviews] : explain how important exaplanability is when prediction the personality since we have to justify the explanation. 
  - Explainability: What is the rationale behind the decision made? e.g. what visual aspects are important, undesirable negative bias…
  - interpretability: What in the model structure explains its functioning
  - why a determined parameter configuration was chosen, what the parameters mean, how a user could interpret the learned model, what additional knowledge would be required from the user/world to improve the model.

### Youtube Vlogs dataset // unimportant
- 2269 videos, 469 different vloggers 
- between 1 and 6 minutes, total 150h
- can not find the dataset. It was from a Competition WCPR14 but the dataset is no longer available
- [competition](https://sites.google.com/site/wcprst/home/wcpr14)
- this dataset was more used in the beginning of this field
- paper example: " Hi YouTube! Personality Impressions and Verbal Content in Social Video", prediction personality based on what they say in the videos
- paper 2 : "You Are Known by How You Vlog: Personality Impressions
and Nonverbal Behavior in YouTube": audio and visual cues

###  Udiva (Understanding Dyadic Interactions from Video and Audio signals dataset) 
   
- large number of synchronized multi-sensory, multi-view recording collected in a face-to-face dyadic interaction scenario with demographics data (age, gender, ethnicity) and self-reported Big 5 scores
- 188 dyadic human-human interactions between 147 participants
- 90.5 h recording
- age from 4-84, 22 diff. nationalities; dominant language: Spanish, Catalan, English
- 5 different interaction context: Talk, animals game, Lego building, Ghost blitz card game, Gaze events 
- associated event: 2021 Understanding Social Behavior in Dyadic and Small Group Interactions Challenge at ICCV, two tasks
  - automatic self-reported personality recognition 
    § Automatic self-reported personality recognition: focus on automatic personality recognition of single individuals (i.e., a target person) during a dyadic interaction, from two individual views
  - behavior forecasting: focus of this track is to estimate future 2D facial landmarks, hand, and upper body pose of a target individual in a dyadic interaction for 2 seconds (50 frames), given an observed time window of at least 4 seconds of both interlocutors, from two individual views.

- [link to dataset](https://chalearnlap.cvc.uab.cat/dataset/39/description/)
- but no access to it, as long not a researcher affiliated to a university signs the license for it : 
" This document must be signed by the Licensee, a natural person who must be a researcher
affiliated to a university or research institute. Other researchers (subsidiaries and/or students)
affiliated with the same institution for whom the Licensee is responsible may be named at the
end of this document which will allow them to work with the Dataset. The signatory of this
license is responsible for the fulfillment of the conditions of this license by all persons linked
to him/her who work with the Dataset "

### MyPersonality Facebook status posts dataset 

- myPersonality was a Facebook App allowing users to participate in psychological research by filling in personality questionnaires
they don't share data anymore : " In 2018, we decided to stop sharing the data with other scholars. Maintaining the data set, vetting the projects, responding to the inquiries, and complying with various regulations had become too burdensome for the two of us. "  [link](https://sites.google.com/michalkosinski.com/mypersonality)


## Evaluation Metrics
- most paper used Accuracy as evaluation. Only ~2 paper i accrossed used Recall, Precision and f1 Score additionally 

## Other Notes
- personality prediction similar to emotion prediction difference but personality is more diverse therefore more difficult
- not so many datasets available. For prediction personality the most datasets came from competitions. 
- most paper based on apparent personality analysis, not the true personality of the individuals 

[back](index.md)