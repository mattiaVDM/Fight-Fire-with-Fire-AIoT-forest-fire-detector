# Fight Fire with Fire: Detecting Forest Fires with Embedded Machine Learning Models Dealing with Audio and Images on Low Power IoT Devices
Link to the article: https://sciprofiles.com/publication/view/816e32fcbe4eaee09695b20f9effb8a5

Giacomo Peruzzi, Alessandro Pozzebon  and Mattia Van Der Meer 

## Abstract 

Forest fires are the main cause of desertification, and they have a disastrous impact on agricultural and forest ecosystems. Modern fire detection and warning systems rely on several techniques: satellite monitoring, sensor networks, image processing, data fusion, etc. 
Recently, Artificial Intelligence (AI) algorithms have been applied to fire recognition systems, enhancing their efficiency and reliability. However, these devices usually need constant data transmission along with a proper amount of computing power, entailing high costs and energy consumption. 
This paper presents the prototype of a Video Surveillance Unit (VSU) for recognising and signalling the presence of forest fires by exploiting two embedded Machine Learning (ML) algorithms running on a low power device. The ML models take audio samples and images as their respective inputs, allowing for timely fire detection. 
The main result is that while the performances of the two models are comparable when they work independently, their joint usage according to the proposed methodology provides a higher accuracy, precision, recall and F1 score (96.15%, 92.30%, 100.00%, and 96.00%, respectively).
Eventually, each event is remotely signalled by making use of the Long Range Wide Area Network (LoRaWAN) protocol to ensure that the personnel in charge are able to operate promptly.

## 1. Introduction

Forest fires are the main cause of desertification, and they have a disastrous impact on agricultural and forest ecosystems. According to the European Forest Fire Information System [1], around 570 km2 of land were destroyed by fire in 2020. Therefore, setting up an automatic detection system for prompt signalling of fire events is of the utmost importance to limit damages and contain the extent of fires.

To this end, the present paper shows a prototype of an autonomous Video Surveillance Unit (VSU) system, bearing in mind forests as the deployment scenario, capable of early spotting of fires and consequently signalling alarms by making use of the Long Range Wide Area Network (LoRaWAN) protocol to remotely notify in-charge personnel. The VSU is based on low power Internet of Things (IoT) devices capable of running Machine Learning (ML) algorithms.

Indeed, fire recognition is carried out by resorting to two embedded ML models: the former deals with audio signals, while the latter deals with pictures. 

To this end, the VSU is provided with two on-board microphones and an on-board camera to feed the ML models with data captured by sensors in the deployment environment. The models are trained and tested on specially gathered datasets, along with a test video reproducing the deployment scenarios and the various phenomena that may take place there. 

In light of this, the prototype in this work falls within the new paradigm of the Artificial Intelligence of Things (AIoT), which is encountered whenever Artificial Intelligence (AI) meets IoT [2]. 

In its essence, all of the techniques and models included within AI are inherited and properly adapted to be executed on low computational capability devices. aiming at solving the problems [3] such as clustering [4], regression [5], classification [6],
and anomaly detection [7].

The motivation of the paper is to propose a VSU prototype enabled by embedded ML algorithms that is capable of timely detection of forest fires and remote signalling of an alarm. Inferences are executed by making use of the data sampled by the on-board sensors (i.e., a camera and two microphones). On the other hand, the main contributions are the following:

- Collect two ad hoc datasets to train and preliminary test the developed ML models. In particular, the former is a dataset containing audio files labelled into two classes, “Fire” and “No-Fire”, on the basis of the presence of fire-related sounds mixed with sundry environmental noises typical of forests (e.g., wind, insects, animals). Conversely, the latter is a dataset containing pictures labelled into the same classes discriminating the presence of fires within forest environments in a wide range of weather conditions and moments of the day (i.e., at night and in daylight). 

- Present a ML-enabled embedded VSU prototype that simultaneously runs two classifiers dealing with audio and picture data in turn in order to recognise forest fires.

- Test and select the devised embedded ML algorithms and propose alarm conditions in order to enhance the classification capability of the VSU prototype, and finally measure the latency and current absorption of the prototype. The technique presented in this paper proposes a novel approach not found in similar works. This aspect is analysed in detail in Section 2; however, to the best of our knowledge no contribution proposing the usage of embedded ML for the detection of fires exploiting both audio and imaging can be found in literature. The rest of the paper is drawn up as follows. Section 2 summarises the current state-of-the art about the topic. In the next Section, the most significant works in the literature dealing with fire detection are presented, and a comparison with the contribution proposed in this work is provided. Section 3 describes the gathered datasets, while Section 4 presents the system architecture of the VSU prototype. Section 5 shows the tests the VSU prototype underwent, while Section 6 highlights conclusions and final remarks, along with suggestions for future works. 

## 2. Dataset Description 

Datasets Description
With the aim of recognising forest fires, two datasets were created: one containing audio files, and one containing images. Of course, both of the datasets included audio and pictures related to forest contexts (e.g., animal sounds, wind, fire, woods, etc.). 

### 2.1 Audio Dataset

Fire sound recognition in forest environments can be traced back to a classification problem, in which the classifier receives audio signals as inputs and returns the probability that the sample at hand belongs to each of the classes. To this end, two classes were defined: “Fire” and “No-Fire”. For each of them, audio samples were collected in an unbiased and heterogeneous way in order to train the network to distinguish as many inputs as possible. In other words, audio contained the sound of fire mixed with noises related to climatic events, to fauna, or even to unexpected events such as the passage of aeroplanes, as well as similar noises devoid of the sound of fire. Concerning the source of the samples, they were retrieved from the openly available FSD50-K [37] and ESC-50 [38] datasets, while others were specially recorded by making use of an Audio-Technica AT2020 microphone driven by a Behringer UM2 sound card and a PC running Audacity. Such samples were recorded at 44 kHz sampling frequency, while the microphone was placed from 2 m to 10 m distance from the fire. Moreover, the duration of each audio sample included in the dataset was standardised to 5 s to improve the training of the relative NN, and in view of the VSU embedded microphone each file was resampled at 16 kHz. The audio dataset was composed of 2864 samples, divided as shown in Table 1, and can be retrieved at [39]. Audio samples belonging to the class “Fire” can be characterised as follows according to their audio type: 
-  Clean Fire—the sound of fire was the only audible noise 
-  Fire with Animal Sounds—the sound of fire was superimposed on sounds coming from animals living in forests (e.g., birds)
-  Fire with Insect Sounds—the sound of fire was superimposed on sounds coming from insects living in forests (e.g., cicadas) 
-  Fire with Wind Noise—the sound of fire was superimposed on the noise of wind or of shaking leaves • Additional Fire Samples—digitally mounted samples in which the sound of fire was superimposed on the samples belonging to the “No-Fire” class with different intensity levels, with the aim of improving generalisation capability of the relative trained NN
-  Recordings—samples recorded according to the procedure explained above during
the combustion of plant residues at a variable distance from the flames.
Audio samples belonging to the class “No-Fire” can be characterised as follows according to their audio type:
-  Animal Sounds—samples of sounds from animals living in forests (e.g., birds) that
were not exploited as background for the “Fire with Animal Sounds” typology of the
“Fire” class
-  Insect Sounds—samples of sounds from insects living in forests (e.g., cicadas) that
were not exploited as background for the “Fire with Insect Sounds” typology of the
“Fire” class
-  Wind Noise—samples of wind noise or of shaking leaves that were not exploited as
background for the “Fire with Wind Noise” typology of the “Fire” class
Sensors 2023, 23, 783 5 of 23
-  Rain Noise—samples of rain noise or thunder
-  Sundry Samples—a set of samples belonging to events that are marginal with respect
to the context of interest, but which could be recorded by the device (e.g., sounds
produced by agricultural machinery, the sound of aircraft, the buzz of people talking
in the distance), included in order to augment the generalisation capability of the
relative trained NN.
</br>
<p align="center">

 <img src="https://user-images.githubusercontent.com/110095822/211930489-1d09551a-ea22-4dd7-beae-3749cb84905b.png"/>

</p>

### 2.2 Picture Dataset

The picture dataset was collected in the same fashion as the audio dataset. Specifically,
all the possible scenarios in which the prototype VSU might work were considered; thus,
pictures of forests in sundry weather conditions (e.g., clear, rain, fog, etc.) as well as in
different seasons and time of the day were gathered to account for several light conditions.
Pictures belonging to the “Fire” class show burning forests both at night and in daylight.
Concerning the source of the pictures forming the dataset, some were retrieved from the
openly available dataset [40], while others were specially taken by exploiting the camera
embedded in the prototype VSU. Moreover, bearing in mind this camera, the pictures were
resized at 320 × 240 px and converted to greyscale.
The picture dataset was composed of 5060 samples, divided as shown in Table 2, and
can be retrieved at [41].
Pictures of the “Noise” type include greyscale samples as well as samples captured by
covering the camera of the prototype VSU with various objects. These were added to train
the relative NN to distinguish any obstacle that might interfere with the camera field of
view.

<p align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535143-afbb8def-1062-4ca7-bc89-04bc66ca383d.png"/>
 
 </p>

## 3. System Overview

The application scenario for which the prototype of VSU was designed is depicted
in [Figure 1](#fig1), which is a forest context that may be potentially at risk of fire. To this end, the prototype VSU aims at preventing and limiting the extent of possible danger by early detection of any fire occurrence and prompt signalling via LoRaWAN links. For this purpose, the VSU possesses embedded ML algorithms capable of spotting fires by making use of both audios and pictures directly recorded by on-board sensors. All of these were
developed in TensorFlow by means of the Keras Python library. In addition, the VSU
is provided with a LoRaWAN transceiver to send alarms whenever the start of a fire is
recognised. The VSU continuously runs a NN devoted to spotting fires by relying on the
captured audio samples; in case of a positive fire inference, it resorts to another NN whose
objective is to detect fires by exploiting pictures taken with the on-board camera. At last,
a fire alarm is sent via LoRaWAN. Finally, in order to preserve privacy and avoid related
issues, neither audio nor pictures are stored or broadcast, as they are locally recorded
and analysed, then eventually deleted. The hardware composing the VSU prototype is
presented in Section 4.1, the NNs are described in Section 4.2 and Section 4.3, respectively,
and the overall functioning of the scheme is shown in Section 4.4.

<p name="fig1" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535304-2d18312b-7498-4dfc-86d2-9bda7136426c.png" />
 
 </p>

### 3.1 Hardware

[Figure 2](#fig2) shows the block diagram of the VSU prototype. Its core is an STM32H47
microcontroller, produced by STMicroelectronics, which is a dual-core microcontroller with
enough computational power to run embedded ML algorithms. Moreover, the microcontroller was embedded in a development board on which an expansion board containing
peripherals was stacked. It includes two microphones (the MP34DT05 produced by STMicroelectronics), a camera (the HM-01B0 produced by Himax), and a LoRaWAN module
(the CMWX1ZZABZ produced by Murata) driven by the microcontroller. Its specific cores
are a Cortex M4 and Cortex M7, each of which has its own tasks to carry out. The Cortex
M4 is devoted to running the audio NN and driving the microphones to feed inputs to
the relative NN. On the other hand, the Cortex M7 runs the picture NN and controls the
Sensors 2023, 23, 783 7 of 23
camera that takes the pictures used as input to the related NN, and additionally drives the
LoRaWAN module to eventually transmit fire alarms. This choice was motivated by the
fact that the audio NN is far tinier and easier to run than the picture one. Therefore, the
latter requires a more powerful core to be executed (i.e., the Cortex M7). As concerns the
LoRaWAN packets, their payloads contained the mean of the probability of fire, namely
p(f), which is related to the inferences generating the fire alarm, to give personnel more
insight. Moreover, regarding the LoRaWAN network, for an in-depth description readers may refer to a previous work [42], as the same network prototype is exploited. It is
composed of LoRaWAN concentrators receiving LoRaWAN packets sent by the the VSU
prototype. These data are demodulated by the concentrators and then forwarded to a
remote network server and an application server running on the cloud by making use of
the Message Queue Telemetry Transport (MQTT) protocol.

</br> 
<p name="fig2" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535463-7998a98f-1022-4575-887e-914d8b7fa36f.png" />
 
</p>
 
### 4.2. Embedded ML Model: Audio NN
Two audio ML models, called audio NN #1 and audio NN #2, were designed and
developed to compare their performances and accordingly select one of them to be deployed
on the Cortex M4 core of the microcontroller. The comparison leading to the choice is
described in Section 5.3. Both of them accounted for a preprocessing stage with the objective
of extracting features from data and an NN classifier distinguishing whether or not the
audio sample at hand belongs to the class “Fire”. The deployed model needs to predictions
based on the audio recorded by the on-board microphones. Similarly, both were designed,
trained, and subjected to preliminary testing on the audio dataset described in Section 3.1.
Furthermore, because data-driven techniques were exploited, a trial-and-error procedure
was followed throughout the design of the models. The hyperparameters of both audio
NN models and their training parameters are listed in [Table 3](#tab3).

<p name="tab3" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535539-a3aef811-9c1c-43e8-be76-1f787f495a76.png" />
 
</p>

#### 4.2.1. Audio Feature Extraction
Digital microphones output time series data, requiring the adoption of sliding temporal
windows during processing. Specifically, audio NN #1 had a window size of 4 s and a
sliding step of 2 s, while audio NN #2 had a window size of 2 s and a sliding step of 2 s. This
choice entailed multiple overlapping windows, resulting in a data augmentation scheme
leading to the expansion of the audio dataset to 5720 and 8580 samples for audio NN #1
and audio NN #2, respectively.

Feature extraction was carried out by exploiting Mel-scaled spectrograms for each
of the temporal windows. This choice was motivated by the proven effectiveness of
this technique when exploited for non-voice audio data, for instance, in the healthcare
domain [31,43–45], for sound detection [46], and in robotic interfaces [47]. Mel-scaled
spectrograms are spectrograms undergoing Mel filterbanks, in which a series of triangular
filters reduce the correlation between consecutive frequency bins of the spectrogram to
which they are applied. In other words, Mel-scaled spectrograms are derived from linear
spectrograms by passage through Mel filterbanks according to the Mel scale. This translates
into the fact that each triangular filter has a maximum unitary response at its central
frequency that linearly decreases towards a null response in correspondence with the
central frequencies of the two adjacent filters. Specifically, the Mel scale was first exploited
to measure the perception of the pitch of sounds according to listeners’ judging them to be
equal in distance from one another. In particular, a 1000-mel pitch corresponds to a 1 kHz
tone at 40 dB above the listener threshold, meaning that a given frequency f translates to a
mel m as:
<p name="for1" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535891-eebf932f-b45c-46ee-bdc9-3d0f5f34825f.png" />
 
</p>

From a practical perspective, Mel filterbanks are able to extract more features from
low frequencies than from higher ones, with the aim of replicating the nonlinear behaviour
of the human ear.

Spectrograms were derived from temporal frames on which the Fast Fourier Transform
(FFT) was applied. Specifically, both audio NN #1 and audio NN #2 had temporal frames of
0.05 s, overlapping 0.025 s with the next to achieve a 50% superimposition. Then, because
audio was resampled at 16 kHz, in order to match with the sampling frequency of the on
board microphones of the VSU prototype we ensured that each temporal frame contained
800 samples, translating to a 1024-point FFT. In addition, both audio NN #1 and audio NN
#2 adopted 40 filterbanks applied to the spectrograms, with 300 Hz as the lowest frequency
band edge of the first filterbanks. Eventually, audio NN #1 had a threshold of −52 dB as
the noise floor, while the respective value was −72 dB for audio NN #2.

#### 4.2.2. Convolutional Neural Network Classifier

Audio NN #1 has a mono-dimensional Convolutional Neural Network (CNN) as classifier, with two outputs representing the probability that a given input belongs to the “Fire”
or “No-Fire” class. This type of NN was chosen because of its ability to deal with features
arranged in spatial domains as images, including spectrograms. The classifier had a reshape
layer at its input that sorts the extracted features from data and forwards them to the next
layers. Then, a mono-dimensional layer containing 8 neurons was inserted, followed by a
pooling layer, to limit the model dimension by computing the maximum value stemming
from the previous convolutional layer. This means that, bearing in mind the model deployment on the Cortex M4 core of the microcontroller, a nonlinear down-sampling was carried out trading off the required computational power and model complexity for accuracy.
Next, a second block mono-dimensional convolutional layer having 16 neurons was placed,
followed by a pooling layer identical to the previous one. In addition, all the neurons in the
convolutional layers used the Rectified Linear Unit (ReLU) as the activation function. The
last layers of the classifier were a flattening layer to properly rearrange data as input to a
dropout layer having rate of 0.5, which was exploited so to reduce the risk of overfitting
during training owing to the fact that a random fraction of the network connections was
pruned throughout the training, followed by a fully connected layer of 64 ReLU neurons,
followed by another dropout layer with a rate of 0.5 and a softmax layer to provide the
output class probabilities as the result.

Audio NN #2 is more complex; it has a two-dimensional CNN with the same number
of outputs as the first model. It then includes a reshape layer followed by four blocks
of two-dimensional convolutional and pooling layers. The latter compute the maximum
value in the same way as in the other model. On the other hand, the convolutional layers
contain 8, 16, 32, and 64 ReLU neurons in turn. The final stage of the classifier is the same as
audio NN #1, that is, a dropout layer with a rate of 0.5, a fully connected layer of 64 ReLU
neurons, another dropout layer with a rate of 0.5, and a softmax layer.
Both models were trained by adopting the standard training–validation procedure
exploiting the standard back-propagation algorithm, employing a learning rate of 0.005. In
particular, the Categorical Cross-Entropy (CCE) was adopted as the loss function:

<p name="for2" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535930-12a29ee8-c993-471d-9368-843400707cf2.png" />
 
</p>

where y is the target label input and yˆ is the related label for the network input at hand. For
the sake of the training process, the audio dataset was divided into training, validation, and
test subsets according to the respective rates of 0.6, 0.2, and 0.2. Accordingly, the examples
belonging to both classes (i.e., “Fire” and “No-Fire”) were split in the same fashion.

#### 4.2.3. Model Quantisation for Deployment
The general ML model needed to be converted into a more compact version, resulting
in an embedded ML model able to be run on a microcontroller. This procedure was carried
out by resort to the TensorFlow Lite framework, transforming a ML model into a C++
library able to be included in the microcontroller firmware and compiled for deployment.

In particular, both the audio NN models were converted into quantised versions accounting
for only 8 bit integers in order to optimise memory occupancy in the microcontroller.

Hereinafter, only the 8 bit versions are considered in the tests and performance comparison
used to select one of the two models.

### 4.3. Embedded ML Model: Picture NN
The picture NN model was grounded on the MobileNetV2 classifier [48], which is an
open source CNN specially developed by Google to run on embedded devices. Thus, it
is suitable for deployment on the Cortex M7 core of the microcontroller. The picture NN
model has a feature extraction stage and a classification stage aiming at distinguishing the
input pictures on the basis of whether they contain fires in forests, meaning that the sample
at hand belongs to the “Fire” class or does not, in which case the sample belongs to the
“No-Fire” class. The model was designed, trained, and preliminarily tested on the greyscale
version of the picture dataset described in Section 3.2, as the camera on board the VSU
prototype cannot take colour images. In addition, because data driven techniques were
employed, a trial-and-error procedure was adopted during the design of the model. As
is explained later, the picture NN was in charge of validating the output provided by the
audio NN.

#### 4.3.1. Picture Feature Extraction

Due to the memory limits imposed by the hardware on which the model was deployed,
and in view of the fact that the Cortex M7 core of the microcontroller has to deal with
LoRaWAN connectivity, the picture NN took as input resized samples from 320 × 240 px
(i.e., the on-board camera resolution) to 96 × 96 px. Of course, as these proportions could
not be maintained, the pictures underwent a squash process.
Owing to the characteristics of the on-board camera, the input samples were in
greyscale. This facilitated the feature extraction process, as the features could be represented by the grey level of each pixel of the sample.

#### 4.3.2. Convolutional Neural Network Classifier

As previously stated, MobileNetV2 was adopted as the classifier for the picture NN.
MobileNet is a depthwise CNN that can significantly reduce the number of parameters
needed to process images compared to a typical CNN used for the same purpose, and
is specially devised to be run on embedded devices. Because it is a pre-trained model
making use of extracted features from large datasets of images belonging to numerous
recognition problems, a transfer learning step is sufficient, thereby saving time compared
to training the whole architecture. In contrast with the adopted methodology for the
design phase of the audio NN, different architectures were not explored for the picture
NN, as MobileNetV2 largely proved to be effective for the classification problems to be
carried out by low computational power devices, as can be verified from the literature.
Indeed, the application scenarios are countless, including ensuring security in accesses to
public places [49], image processing [50], cactaceae detection [51], analysing the quality of
corn [52], waste management [53,54], body temperature and face mask detection [55], and
food classification [56].
The model was trained by resort to the training–validation procedure by making use
of the back-propagation algorithm, adopting a learning rate of 0.000045. Moreover, the CCE
was used as the loss function (see Equation (2)). In addition, in order to perform training
the greyscale version of the picture dataset was split into training, validation, and test
subsets according to the rates of 0.6, 0.2, and 0.2, respectively. Consequently, the examples
belonging to the two classes of “Fire” and “No-Fire” were divided in the same fashion.

#### 4.3.3. Model Quantisation for Deployment

As stated in Section 4.2.3, the conversion of a ML model meant to be deployed and run
on an embedded device is a crucial step. The procedure explained above was adopted for
the picture NN, that is, it was converted into a C++ library by making use of TensorFlow
Lite. Then, the library was included in the microcontroller firmware to be compiled
and deployed on the device. The picture NN was converted into its version accounting
for only 8 bit integers, as optimising memory occupancy was of the utmost importance because the Cortex M7 core of the microcontroller has to additionally deal with LoRaWAN
transmissions.

### 4.4. VSU Functioning Scheme
The VSU prototype worked by following the functioning scheme shown in [Figure 3](#fig3).
For the time being, without loss of generality, suppose that one of the two audio NN models
has been chosen; the detailed procedure is explained in Section 5.3.

<p name="fig3" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212536111-817c6029-7206-4f5a-b0d9-8d0a76429a35.png" />
 
</p>

First, the Cortex M4 core of the microcontroller is turned on, while the Cortex M7 core
is put into sleep mode to reduce power consumption. Then, the audio NN is run to provide
inferences on the signals recorded by the on-board microphones. Let FcA be the condition
under which the outputs provided by the audio NN can be associated with a fire alarm
(FcA is described in Section 5.3). If FcA is not met, then the audio NN continues to infer;
otherwise, the mean value of the probability of fire related to the outputs of the audio NN
generating FcA, namely, µp(fA)
, is computed and stored. At this stage, the result provided
by the audio NN is validated by making use of the picture NN; therefore, the Cortex M4
core of the microcontroller is put into sleep mode to reduce power consumption, while the
Cortex M7 core is turned on in order to run the picture NN that classifies the pictures taken
by the on-board camera. Let FcP be the condition under which the outputs provided by the
picture NN can be associated with a fire alarm (FcP is described in Section 5.3). If FcP is
not met, the VSU prototype restarts its workflow by turning the Cortex M4 core on and
putting the Cortex M7 into sleep mode; otherwise, the mean value of the probability of fire
related to the outputs of the picture NN generating FcP, namely, µp(fP)
, is computed and
stored. This condition translates into the actual risk of fire, denoting the need for signalling
said phenomenon. To this end, µp(fA)
is recalled and paired with µp(fP)
in order to form the payload of a LoRaWAN packet to be remotely sent. Finally, the VSU prototype restarts
its workflow by turning on the Cortex M4 core and putting the Cortex M7 core into sleep
mode.
