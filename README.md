# Fight Fire with Fire: Detecting Forest Fires with Embedded Machine Learning Models Dealing with Audio and Images on Low Power IoT Devices
Link to the article: https://sciprofiles.com/publication/view/816e32fcbe4eaee09695b20f9effb8a5

<i>Giacomo Peruzzi, Alessandro Pozzebon  and Mattia Van Der Meer </i>

## Table of contents: 

- [Introduction](#intro)
- [Dataset description](#dataset)
- [System overview](#system)
- [Laboratory tests and results](#tests)
- [Conclusions](#conclusions)


## Abstract 

Forest fires are the main cause of desertification, and they have a disastrous impact on agricultural and forest ecosystems. Modern fire detection and warning systems rely on several techniques: satellite monitoring, sensor networks, image processing, data fusion, etc. 
Recently, Artificial Intelligence (AI) algorithms have been applied to fire recognition systems, enhancing their efficiency and reliability. However, these devices usually need constant data transmission along with a proper amount of computing power, entailing high costs and energy consumption. 
This paper presents the prototype of a Video Surveillance Unit (VSU) for recognising and signalling the presence of forest fires by exploiting two embedded Machine Learning (ML) algorithms running on a low power device. The ML models take audio samples and images as their respective inputs, allowing for timely fire detection. 
The main result is that while the performances of the two models are comparable when they work independently, their joint usage according to the proposed methodology provides a higher accuracy, precision, recall and F1 score (96.15%, 92.30%, 100.00%, and 96.00%, respectively).
Eventually, each event is remotely signalled by making use of the Long Range Wide Area Network (LoRaWAN) protocol to ensure that the personnel in charge are able to operate promptly.

## 1. <a name="intro"> Introduction </a>

Forest fires are the main cause of desertification, and they have a disastrous impact on agricultural and forest ecosystems. According to the European Forest Fire Information System , around 570 km2 of land were destroyed by fire in 2020. Therefore, setting up an automatic detection system for prompt signalling of fire events is of the utmost importance to limit damages and contain the extent of fires.

To this end, the present paper shows a prototype of an autonomous Video Surveillance Unit (VSU) system, bearing in mind forests as the deployment scenario, capable of early spotting of fires and consequently signalling alarms by making use of the Long Range Wide Area Network (LoRaWAN) protocol to remotely notify in-charge personnel. The VSU is based on low power Internet of Things (IoT) devices capable of running Machine Learning (ML) algorithms.

Indeed, fire recognition is carried out by resorting to two embedded ML models: the former deals with audio signals, while the latter deals with pictures. 

To this end, the VSU is provided with two on-board microphones and an on-board camera to feed the ML models with data captured by sensors in the deployment environment. The models are trained and tested on specially gathered datasets, along with a test video reproducing the deployment scenarios and the various phenomena that may take place there. 

In light of this, the prototype in this work falls within the new paradigm of the Artificial Intelligence of Things (AIoT), which is encountered whenever Artificial Intelligence (AI) meets IoT . 

In its essence, all of the techniques and models included within AI are inherited and properly adapted to be executed on low computational capability devices. aiming at solving the problems such as clustering, regression, classification,
and anomaly detection.

The motivation of the paper is to propose a VSU prototype enabled by embedded ML algorithms that is capable of timely detection of forest fires and remote signalling of an alarm. Inferences are executed by making use of the data sampled by the on-board sensors (i.e., a camera and two microphones). On the other hand, the main contributions are the following:

- Collect two ad hoc datasets to train and preliminary test the developed ML models. In particular, the former is a dataset containing audio files labelled into two classes, “Fire” and “No-Fire”, on the basis of the presence of fire-related sounds mixed with sundry environmental noises typical of forests (e.g., wind, insects, animals). Conversely, the latter is a dataset containing pictures labelled into the same classes discriminating the presence of fires within forest environments in a wide range of weather conditions and moments of the day (i.e., at night and in daylight). 

- Present a ML-enabled embedded VSU prototype that simultaneously runs two classifiers dealing with audio and picture data in turn in order to recognise forest fires.

- Test and select the devised embedded ML algorithms and propose alarm conditions in order to enhance the classification capability of the VSU prototype, and finally measure the latency and current absorption of the prototype. The technique presented in this paper proposes a novel approach not found in similar works. This aspect is analysed in detail in [Section 1](#intro); however, to the best of our knowledge no contribution proposing the usage of embedded ML for the detection of fires exploiting both audio and imaging can be found in literature. The rest of the paper is drawn up as follows. [Section 2](#dataset) describes the gathered datasets, while [Section 3](#system) presents the system architecture of the VSU prototype. [Section 4](#tests) shows the tests the VSU prototype underwent, while Section 6 highlights conclusions and final remarks, along with suggestions for future works. 

## <a name="dataset">2. Dataset Description </a>

Datasets Description
With the aim of recognising forest fires, two datasets were created: one containing audio files, and one containing images. Of course, both of the datasets included audio and pictures related to forest contexts (e.g., animal sounds, wind, fire, woods, etc.). 

### 2.1 Audio Dataset

Fire sound recognition in forest environments can be traced back to a classification problem, in which the classifier receives audio signals as inputs and returns the probability that the sample at hand belongs to each of the classes. To this end, two classes were defined: “Fire” and “No-Fire”. For each of them, audio samples were collected in an unbiased and heterogeneous way in order to train the network to distinguish as many inputs as possible. In other words, audio contained the sound of fire mixed with noises related to climatic events, to fauna, or even to unexpected events such as the passage of aeroplanes, as well as similar noises devoid of the sound of fire. Concerning the source of the samples, they were retrieved from the openly available FSD50-K and ESC-50 datasets, while others were specially recorded by making use of an Audio-Technica AT2020 microphone driven by a Behringer UM2 sound card and a PC running Audacity. Such samples were recorded at 44 kHz sampling frequency, while the microphone was placed from 2 m to 10 m distance from the fire. Moreover, the duration of each audio sample included in the dataset was standardised to 5 s to improve the training of the relative NN, and in view of the VSU embedded microphone each file was resampled at 16 kHz. The audio dataset was composed of 2864 samples, divided as shown in [Table 1](#tab1), and can be retrieved at . Audio samples belonging to the class “Fire” can be characterised as follows according to their audio type: 
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
<figcaption> <b>Table 1.</b> Audio dataset description. </figcaption>
</br>
<p name="tab1" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/211930489-1d09551a-ea22-4dd7-beae-3749cb84905b.png"/>

</p>

### 2.2 Picture Dataset

The picture dataset was collected in the same fashion as the audio dataset. Specifically,
all the possible scenarios in which the prototype VSU might work were considered; thus,
pictures of forests in sundry weather conditions (e.g., clear, rain, fog, etc.) as well as in
different seasons and time of the day were gathered to account for several light conditions.
Pictures belonging to the “Fire” class show burning forests both at night and in daylight.
Concerning the source of the pictures forming the dataset, some were retrieved from the
openly available dataset, while others were specially taken by exploiting the camera
embedded in the prototype VSU. Moreover, bearing in mind this camera, the pictures were
resized at 320 × 240 px and converted to greyscale.
The picture dataset was composed of 5060 samples, divided as shown in [Table 2](#tab2), and
can be retrieved at .
Pictures of the “Noise” type include greyscale samples as well as samples captured by
covering the camera of the prototype VSU with various objects. These were added to train
the relative NN to distinguish any obstacle that might interfere with the camera field of
view.
<figcaption> <b>Table 2.</b> Picture dataset description. </figcaption>
</br>
<p name="tab2" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535143-afbb8def-1062-4ca7-bc89-04bc66ca383d.png"/>
 
 </p>

## <a name="system"> 3. System Overview </a>

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
presented in Section 3.1, the NNs are described in Section 3.2 and Section 3.3, respectively,
and the overall functioning of the scheme is shown in Section 3.4.

<p name="fig1" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535304-2d18312b-7498-4dfc-86d2-9bda7136426c.png" />
  <figcaption> <b>Figure 1.</b> Application scenario of the prototype VSU. </figcaption>
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
insight. Moreover, regarding the LoRaWAN network, for an in-depth description readers may refer to a previous work, as the same network prototype is exploited. It is
composed of LoRaWAN concentrators receiving LoRaWAN packets sent by the the VSU
prototype. These data are demodulated by the concentrators and then forwarded to a
remote network server and an application server running on the cloud by making use of
the Message Queue Telemetry Transport (MQTT) protocol.

</br> 
<p name="fig2" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535463-7998a98f-1022-4575-887e-914d8b7fa36f.png" />
 <figcaption> <b>Figure 2.</b> VSU prototype block diagram.  </figcaption>
 
</p>
 
### 3.2. Embedded ML Model: Audio NN
Two audio ML models, called audio NN #1 and audio NN #2, were designed and
developed to compare their performances and accordingly select one of them to be deployed
on the Cortex M4 core of the microcontroller. The comparison leading to the choice is
described in Section 3.3. Both of them accounted for a preprocessing stage with the objective
of extracting features from data and an NN classifier distinguishing whether or not the
audio sample at hand belongs to the class “Fire”. The deployed model needs to predictions
based on the audio recorded by the on-board microphones. Similarly, both were designed,
trained, and subjected to preliminary testing on the audio dataset described in Section 2.1.
Furthermore, because data-driven techniques were exploited, a trial-and-error procedure
was followed throughout the design of the models. The hyperparameters of both audio
NN models and their training parameters are listed in [Table 3](#tab3).
<figcaption> <b>Table 3.</b> Hyperparameters and training parameters for audio NN #1 and audio NN #2.
 </figcaption>
 </br>
<p name="tab3" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212535539-a3aef811-9c1c-43e8-be76-1f787f495a76.png" />
 
</p>

#### 3.2.1. Audio Feature Extraction
Digital microphones output time series data, requiring the adoption of sliding temporal
windows during processing. Specifically, audio NN #1 had a window size of 4 s and a
sliding step of 2 s, while audio NN #2 had a window size of 2 s and a sliding step of 2 s. This
choice entailed multiple overlapping windows, resulting in a data augmentation scheme
leading to the expansion of the audio dataset to 5720 and 8580 samples for audio NN #1
and audio NN #2, respectively.

Feature extraction was carried out by exploiting Mel-scaled spectrograms for each
of the temporal windows. This choice was motivated by the proven effectiveness of
this technique when exploited for non-voice audio data, for instance, in the healthcare
domain , for sound detection , and in robotic interfaces. Mel-scaled
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

#### 3.2.2. Convolutional Neural Network Classifier

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

#### 3.2.3. Model Quantisation for Deployment
The general ML model needed to be converted into a more compact version, resulting
in an embedded ML model able to be run on a microcontroller. This procedure was carried
out by resort to the TensorFlow Lite framework, transforming a ML model into a C++
library able to be included in the microcontroller firmware and compiled for deployment.

In particular, both the audio NN models were converted into quantised versions accounting
for only 8 bit integers in order to optimise memory occupancy in the microcontroller.

Hereinafter, only the 8 bit versions are considered in the tests and performance comparison
used to select one of the two models.

### 3.3. Embedded ML Model: Picture NN
The picture NN model was grounded on the MobileNetV2 classifier, which is an
open source CNN specially developed by Google to run on embedded devices. Thus, it
is suitable for deployment on the Cortex M7 core of the microcontroller. The picture NN
model has a feature extraction stage and a classification stage aiming at distinguishing the
input pictures on the basis of whether they contain fires in forests, meaning that the sample
at hand belongs to the “Fire” class or does not, in which case the sample belongs to the
“No-Fire” class. The model was designed, trained, and preliminarily tested on the greyscale
version of the picture dataset described in Section 2.2, as the camera on board the VSU
prototype cannot take colour images. In addition, because data driven techniques were
employed, a trial-and-error procedure was adopted during the design of the model. As
is explained later, the picture NN was in charge of validating the output provided by the
audio NN.

#### 3.3.1. Picture Feature Extraction

Due to the memory limits imposed by the hardware on which the model was deployed,
and in view of the fact that the Cortex M7 core of the microcontroller has to deal with
LoRaWAN connectivity, the picture NN took as input resized samples from 320 × 240 px
(i.e., the on-board camera resolution) to 96 × 96 px. Of course, as these proportions could
not be maintained, the pictures underwent a squash process.
Owing to the characteristics of the on-board camera, the input samples were in
greyscale. This facilitated the feature extraction process, as the features could be represented by the grey level of each pixel of the sample.

#### 3.3.2. Convolutional Neural Network Classifier

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
public places , image processing, cactaceae detection, analysing the quality of
corn , waste management, body temperature and face mask detection, and
food classification.
The model was trained by resort to the training–validation procedure by making use
of the back-propagation algorithm, adopting a learning rate of 0.000045. Moreover, the CCE
was used as the loss function (see Equation (2)). In addition, in order to perform training
the greyscale version of the picture dataset was split into training, validation, and test
subsets according to the rates of 0.6, 0.2, and 0.2, respectively. Consequently, the examples
belonging to the two classes of “Fire” and “No-Fire” were divided in the same fashion.

#### 3.3.3. Model Quantisation for Deployment

As stated in Section 3.2.3, the conversion of a ML model meant to be deployed and run
on an embedded device is a crucial step. The procedure explained above was adopted for
the picture NN, that is, it was converted into a C++ library by making use of TensorFlow
Lite. Then, the library was included in the microcontroller firmware to be compiled
and deployed on the device. The picture NN was converted into its version accounting
for only 8 bit integers, as optimising memory occupancy was of the utmost importance because the Cortex M7 core of the microcontroller has to additionally deal with LoRaWAN
transmissions.

### 3.4. VSU Functioning Scheme
The VSU prototype worked by following the functioning scheme shown in [Figure 3](#fig3).
For the time being, without loss of generality, suppose that one of the two audio NN models
has been chosen; the detailed procedure is explained in Section 4.3.

<p name="fig3" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212536111-817c6029-7206-4f5a-b0d9-8d0a76429a35.png" />
  <figcaption> <b>Figure 3.</b> VSU prototype functioning flow chart.  </figcaption>
</p>

First, the Cortex M4 core of the microcontroller is turned on, while the Cortex M7 core
is put into sleep mode to reduce power consumption. Then, the audio NN is run to provide
inferences on the signals recorded by the on-board microphones. Let FcA be the condition
under which the outputs provided by the audio NN can be associated with a fire alarm
(FcA is described in Section 4.3). If FcA is not met, then the audio NN continues to infer;
otherwise, the mean value of the probability of fire related to the outputs of the audio NN
generating FcA, namely, µp(fA)
, is computed and stored. At this stage, the result provided
by the audio NN is validated by making use of the picture NN; therefore, the Cortex M4
core of the microcontroller is put into sleep mode to reduce power consumption, while the
Cortex M7 core is turned on in order to run the picture NN that classifies the pictures taken
by the on-board camera. Let FcP be the condition under which the outputs provided by the
picture NN can be associated with a fire alarm (FcP is described in Section 4.3). If FcP is
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

## <a name="tests"> 4. Laboratory Tests and Results </a>
The laboratory tests conducted on the VSU prototype aimed at the following objectives:
• Assessing the classification performance on the relative test set for audio NN #1, audio
NN #2, and the picture NN;
• Assessing the hardware performance of audio NN #1, audio NN #2, and the picture
NN;
• Selecting one model between audio NN #1 and audio NN #2 on the basis of their
performance on an ad hoc edited video;
• Choosing FcA and FcP on the basis of the performance of the selected audio NN and
the picture NN on the same video mentioned above;
• Assessing the performance of the VSU prototype from the point of view of classification and latency, making use of the same video as above;
• Measuring the current drawn by the VSU prototype.

### 4.1. Performance on Test Set and Hardware
A preliminary test consisted of assessing the classification accuracy on the test set. To
this end, audio NN #1, audio NN #2, and the picture NN were provided with samples
belonging to the relative test sets as inputs, derived from the 0.2 ratio of the audio dataset
(see Section 2.1) for audio NN #1 and audio NN #2 and from the picture dataset (see
Section 2.2) for the picture NN. The results are reported in [Figure 4](#fig4) in the form of confusion
matrices.
Audio NN #1 had an overall accuracy of 90.890%, audio NN #2 had 95.375%, and the
picture NN had 87.495%, meaning that all the model showed a satisfactory generalisation
capability stemming from an adequate training stage. As concerns True Positives (TPs) and
True Negatives (TNs) classification, audio NN #1 achieved 88.53% and 93.25%, audio NN
#2 95.86% and 94.89%, and the picture NN 88.86% and 86.13% respectively. Conversely,
regarding misclassifications, the False Positive (FP) and False Negatives (FN) rates were
2.894% and 8.961% for audio NN #1, 3.604% and 2.707% for audio NN #2, and 10.22% and
7.109% for the picture NN, respectively. On the whole, misclassifications were limited; that
said, the most dangerous outcomes are related to FNs, as this means that an actual fire
occurred and the model did not recognise it. These instances can potentially be reduced by
a model accounting for audio and pictures simultaneously, as the proposed VSU prototype
is intended to operate. Indeed, the next tests aimed to assess the behaviour of the VSU
prototype according to the procedure described in Section 3.4. In addition, when the output
class probability is between 0.4 and 0.6 an uncertain outcome is provided, meaning that the
model result is not reliable. This phenomenon does not take place often, occurring no more
than in 4.028% of cases for the picture NN, 3.859% for audio NN #1, and 1.502% for audio
NN #2.
From the hardware perspective, [Table 4](#tab4) summarises the three models. Owing to the
deployment of the models on the microcontroller, it is of the utmost importance to minimise
RAM and flash occupancy in order to allow their execution. Specifically, the microcontroller
features 2 MB of flash memory and 1 MB of RAM, and both these memories are shared
among the two cores. However, figures in [Table 4](#tab4) suggest that the ML models can be
correctly executed along with the sundry routines (e.g., initialization, data transmission,
etc.) that the microcontroller has to carry out. Moreover, in order to limit the current draw
of the VSU prototype, the execution time is important, as it is directly proportional to
current drawn. Therefore, focusing solely on hardware specifics, audio NN #1 is preferable
to audio NN #2 despite the latter being slightly more accurate.

<p name="fig4" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212542958-ea8ca8dd-5d00-4ff8-8894-d3db3912c0b1.png" />
 <figcaption> <b>Figure 4.</b> Confusion matrices showing the performance of the models on the test set: (a) audio NN
#1, (b) audio NN #2, and (c) picture NN.  </figcaption>
</p>
<figcaption> <b>Table 4.</b> Hardware performance of the ML models. </figcaption>
 </br>
<p name="tab4" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212542991-a192314d-59e0-4598-a8cd-9542771409e3.png" />
 
</p>

### 4.2. Ad Hoc Edited Test Video
Due to the obvious inability of setting fires in either the laboratory or within real
forests, an ad hoc video was specially edited for testing purposes;
It is an hour-long slideshow consisting of alternating pictures taken at different times of
the day every 30 s, to which sounds were added. Pictures and sounds were matched to
the displayed scenario, meaning that if no fire was present, audio samples not related to
fire were reproduced, while if a fire was present the audio samples related to fire were
inserted. The audio and pictures used to test the generalisation capabilities of the VSU
prototype did not belong to the datasets described in Section 2. The video was sorted
into 26 sections, including equally distributed scenarios belonging to both classes (i.e.,
“Fire” and “No-Fire”). Moreover, the 13 parts belonging to the “Fire” class were further
categorised into 3 typologies according to the extent of fire within the picture and in the
corresponding reproduced sound. The first test was a Low Noise Test (LNT), in which
audio samples belonging to the “Fire” class were mixed with audio samples belonging
to the “No-Fire” class by setting the volume of the “Fire” samples to double that of the
“No-Fire” samples. Regarding the images, they had fire in the foreground. The second test was the Mild Noise Test (MNT), in which audio samples belonging to the “Fire” class were
mixed with audio samples belonging to the “No-Fire” class by setting both volumes at the
same level, while the pictures had clearly visible fires of modest size. The third test was
the Extreme Noise Test (ENT), in which audio samples belonging to the “Fire” class were
mixed with audio samples belonging to the “No-Fire” class by setting the volume of the
“Fire” samples to the half that of the “No-Fire” samples. Concerning the images, they had
fires either of a smaller size or far in the background.

### 4.3. Performance on Test Video, Audio NN Model Selection, and Condition of Fire Definition
The objectives of these tests were multiple. For the audio NN models, the tests were
used to select one model between audio NN #1 and audio NN #2 and to define FcA.
Conversely, for the picture NN only the definition of FcP needed to be accomplished.
The models were deployed on the microcontroller, and the VSU prototype was tested
on the test video described in Section 3.2 by assessing the performance of each of the
models separately. The tests were carried out in a dark environment where the only light
source was a 24-inch full HD monitor. Audio was reproduced by exploiting a system of
four 40 W loudspeakers arranged at the corners of a square having sides 2 m in length. In
particular, one of the sides hosted the monitor and the VSU was placed in the centre of the
square. Then, the video was reproduced three times (i.e., once for each of the embedded ML
models) and the predictions of each model (i.e., the probability that the acquired sample
belonged to the “Fire” class) were monitored and stored using a PC acting as a data-logger.
The methods used to define FcA and FcP are analogous; therefore, without loss of
generality, we only focus on FcA here. As it is advisable to take into account more than a
single inference in order to enhance the reliability of predictions, let nA be the number of
consecutive inferences such that p(fA) (i.e., the probability that the input audio sample
at hand belongs to the “Fire” class) is greater or equal to thA (i.e., the minimum p(fA) for
which the output of the audio NN model can be interpreted as belonging to the input audio
sample in the “Fire” class). Then, let minTP(thA) be the minimum number of consecutive
inferences related as TPs for a given thA, and let MAXFP(thA) be the maximum number
of consecutive inferences related as FPs for a given thA. The objective is to identify the
maximum thA∗ resulting
<br/>
<p name="for3" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543234-d74a366d-e3c7-4da4-91ac-88dc1a4eee1d.png" />
 
</p>

<br/>

where the right-hand side is increased to follow a conservative approach. This means that

<br/>

<p name="for4" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543290-25d0f6c1-13d7-4a77-9395-cd4bd463d44d.png" />
 
</p>

<br/>

Then, nA∗ can be retrieved by considering that if

<p name="for5" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543304-730f7201-8fdd-4e0e-94f2-cd6a207330d7.png" />

</p>

<br/>

no FPs are generated, while FNs (i.e., the most dangerous outcomes) are avoided if

<br/>

<p name="for6" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543330-c1c62e4f-7cc0-4589-b420-255e3ff86ebd.png" />
 
</p>

<br/>

meaning that conditions (5) and (6) merge into

<br/>

<p name="for6" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543412-04ed685d-412f-482c-ba42-ec3ac33c5c73.png" />
 
</p>

<br/>

In light of this, FcA is the pair thA∗ and nA∗, while FcP is the pair thP∗ and nP∗.
Concerning the audio NN models, FcA for the audio NN #1 (i.e., FcA1
), and FcA for
the audio NN #2 (i.e., FcA2
) were found by checking conditions (4) and (7) by exploiting the
test video described in Section 3.2 and having resort to the the aforementioned test setup.
In particular, the relative minTP(thA) and MAXFP(thA) for each model were evaluated for
thA by varying 0.6 to 1 with steps of 0.001; it would make no sense to analyse the model behaviours for thA < 0.6, as the classifier would not output a “Fire” result in any case.
Moreover, the relative thA* and nA∗ forming FcA1 and FcA2 where graphically identified, as shown in [Figure 5](#fig5), where the terms of Equation (3) are plotted. 

This test proved that for FcA1, thA∗ = 0.808 and nA∗ = 2, while thA∗ = 0.940 and nA∗ = 2 for FcA2.


<p name="fig5" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543584-c7450a31-b8a5-45ca-b8c6-f3112070fba9.png" />
 <figcaption> <b>Figure 5.</b> FcA condition identification: (a) audio NN #1 and (b) audio NN #2.
  </figcaption>
</p>

Then, model selection was carried out by looking at the accuracy, recall, precision
and F1 score metrics on the test video for the two models considering FcA1
and FcA2, respectively. The results are shown in [Table 5](#tab5), showing that the two models performed
the same. Therefore, because the microcontroller has a limited amount of memory and
its Cortex M4 core has to deal with other routines, audio NN #1 was preferred thanks to
its being better from the hardware perspective (see [Table 4](#tab4)). Hereinafter, this model is
referred as the audio NN model; its output in terms of p(fA) for the test video is displayed
in [Figure 6](#fig6).

<figcaption> <b>Table 5.</b> Test video results for audio NN model selection. </figcaption>
 </br>
<p name="tab5" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543694-ba715619-c03b-445c-9c48-2e7ad38c5bbc.png" />
 <figcaption> <b>Figure 5.</b> FcA condition identification: (a) audio NN #1 and (b) audio NN #2.
</p>

Regarding the picture NN, FcP was found by adopting the same methodology as that
used for the audio models. [Figure 7](#fig7) reports the results, showing that FcP had thP∗ = 0.823 and nP∗ = 4. [Table 6](#tab6) and [Figure 8](#fig8) show the picture NN model results on the test video.

<p name="fig6" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543759-1eb7a3f4-108f-409b-8538-de9b5a1bcb12.png" />
 <figcaption> <b>Figure 6.</b> Audio NN model results on the test video. The green areas represent input samples
belonging to the “No-Fire” class. The yellow, pink, and grey areas represent input samples belonging
to the “Fire” class in LNT, MNT, and ENT, respectively. The solid blue line stands for the trend of
p(fA), while the dash-dotted line represents thA∗. Finally, the vertical red lines mark the instants at
which the audio NN model signalled a fire alarm.
.
</p>

<p name="fig7" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543810-c2b7ec8b-5e68-429f-aedd-691194723335.png" />
 <figcaption> <b>Figure 7.</b> FcP condition identification.
</p>

<figcaption> <b>Table 6.</b> Test video results for the picture NN model.
 </br>
<p name="tab6" align="center">
 <img src="https://user-images.githubusercontent.com/110095822/212543829-3d08f60c-686f-475e-9f9a-9050c802479b.png" />
</p>

<p name="fig8" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543862-dff7a55f-9c2a-4cd1-bba1-a3b594a2eb89.png" />
 <figcaption> <b>Figure 8.</b> Picture NN model results on the test video. The green areas represent input samples
belonging to the class“No-Fire”. The yellow, pink, and grey areas represent input samples belonging
to the “Fire” class in LNT, MNT, and ENT, respectively. The solid blue line stands for the trend of
p(fP), while the dash-dotted line represents thP∗. Finally, the vertical red lines mark the instants at
which the picture NN model signalled a fire alarm.
</p>

### 4.4. Performance of the VSU Prototype
After identifying the best audio NN model, FcA, and FcP, the VSU prototype was
tested according to the functioning scheme shown in Section 3.4 on the test video described
in Section 3.2 and following the methodology therein. [Figure 9](#fig9) reports the relative results
in terms of p(f) over time. Moreover, [Table 7](#tab7) summarises the tests results and compares
them with the results of the same tests for the audio NN and picture NN models alone,
while [Figure 10](#fig10) shows a comparison displaying the classification metrics.
Although the picture NN is the best in terms of recall, its usage without the audio
NN causes too many FPs. Its adoption and exploitation according to the VSU prototype
functioning scheme results in improvement of the system’s precision and accuracy, reducing
FPs and avoiding FNs (the most dangerous outcome). Indeed, the test results prove that the
picture NN is able to recognise all the alarms coming from the audio NN while blocking
the propagation of FPs and FNs, thereby enchaining the precision of the whole system up
to 100%. Moreover, the only misclassification took place in an ENT frame of the test video.
Concerning the avoidance of FNs, this was ensured even by the picture NN on its own.
However, in addition to the aforementioned motivations, using such a network would be
too energy consuming, as demonstrated in Section 4.5.

<p name="fig9" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212543968-c8a09c03-c4a7-4595-8ad4-d63093669035.png" />
 <figcaption> <b>Figure 9.</b> VSU prototype results on the test video. The green areas represent input samples belonging
to the “No-Fire” class. The yellow, pink, and grey areas represent input samples belonging to the
“Fire” class in LNT, MNT, and ENT, respectively. The solid blue line stands for the trend of p(f), the
dash-dotted line represents thA∗, and the dashed line represents thP∗. Finally, the vertical red lines
mark the instants at which the VSU prototype signalled a fire alarm.
</p>
<figcaption> <b>Table 7.</b>Test video results for the VSU prototype compared with the results of the audio NN and
picture NN models alone.
 </br>
<p name="tab7" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212544004-e09fa5a9-7572-4a79-8b65-75c4fc3bd747.png" />
</p>

[Figure 11](#fig11) shows the Cumulative Distribution Function (CDF) related to the VSU
prototype alarm latency owing to data acquisition, processing, and elaboration during the
test video inferencing. The mean latency was 37.1667 s; from the latency CDF it can be
noted that the latency is above the mean, with a probability of 0.16. However, the CDF
additionally shows that the latency is less than 58 s with a probability of 0.92, while its
maximum value was 120 s, meaning that the VSU prototype is able to signal fires in a
timely fashion.

<p name="fig10" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212544037-995f63fe-d2d1-43ab-9d9e-e2d90be24e47.png" />
 <figcaption> <b>Figure 10.</b> . Performance comparison of the audio NN model, picture NN model, and VSU prototype.
</p>
<p name="fig11" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212544076-ffb1803f-5f9e-45ea-bc9c-a3e56fdfd03c.png" />
 <figcaption> <b>Figure 11.</b> VSU prototype alarm latency CDF.
</p>

### 4.5. Current Drawn by the VSU Prototype
The last test for the VSU prototype was the measurement of its current drawn during
operation. Because the VSU prototype is powered by exploiting a constant voltage input
source, its power consumption is directly proportional to the current drawn. To this end, the
prototype was tested on an ENT segment of the test video to assess a worst case scenario,
and its current consumption was measured. The VSU prototype was powered via a dual
channel bench power supply and the current draw was measured with a digital multimeter
Agilent 34410A. The instrument was controlled via LabVIEW, and the readings were
acquired at a sampling frequency of 5 Hz. Then, the sampled measurements were analysed
by means of MATLAB, with the results provided in [Figure 12](#fig12). Four working periods can be
identified: setup, audio NN inferences, picture NN inferences, and LoRaWAN transmission
(i.e., “LoRaWAN Tx” in  [Figure 12](#fig12)). The setup phase lasted 3.12 s, during which the VSU
executed the initial routines, requiring a mean current absorption of 144.3 mA. The audio
NN inferences took a time interval of 12.38 s, during which the two inferences can be easily
spotted, as current peaks (i.e., 215 mA and 218 mA) were experienced. During this period,
a mean current absorption of 135 mA was required, and only the Cortex M4 core of the
microcontroller was running. The picture NN inferences required 17.43 s. Similarly, the
four inferences can be seen as current peaks (respectively, at 267 mA, 211 mA, 221 mA,
and 216 mA) were present. Throughout this working phase, a mean current absorption of
150.8 mA was experienced, and only the Cortex M7 core of the microcontroller was running.
Finally, the LoRaWAN transmission phase lasted 7.49 s, and the current peak of 221 mA
related to the transmission can be seen. In this working period, a mean power absorption
of 126.1 mA was needed, and the only active core was the Cortex M7. This test hints at the
fact that, at least in this stage, the VSU prototype cannot only be battery-powered, and an
energy harvesting system (e.g., photovoltaic panels) can be exploited in order to recharge
a backup battery powering the system. Another alternative could be to provide the VSU
prototype with duty cycling policies, though this would increase the alarm latency.

<p name="fig12" align="center">

 <img src="https://user-images.githubusercontent.com/110095822/212544140-d4e3e9fa-e8d3-40b9-a702-298e1c0a5e00.png" />
 <figcaption> <b>Figure 12.</b> VSU prototype current drawn.
</p>


## <a name="conclusions"> 5. Conclusions </a>
This paper proposed a prototype of an embedded IoT device implementing a VSU
for early detection of forest fires. It uses two embedded ML algorithms, the former to deal
with audio signals and the latter for pictures. To this end, two datasets were specifically
gathered to train and test the models, and an ad hoc test video was created to test the
system generalisation capability. Acoustic and visual data are acquired and sampled by
the on-board microphones and camera, making the prototype fully stand-alone. Audio
is continuously captured and analysed by the dedicated ML model; as soon as a fire
is detected, pictures are taken that the ML model can assess the actual presence of fire.
The working principle was proven to improve classification metrics, minimising FPs and
especially FNs.
More precisely, the VSU prototype achieved accuracy, recall, precision and
F1 scores on the test video of 96.15%, 92.3%, 100%, and 96%, respectively, providing a mean
alarm latency of 37.1667 s. In case of a fire, a LoRaWAN alarm is remotely sent to notify
the personnel in charge. In future works, this prototype can be further developed and
optimised from the point of view of both hardware components and software routines. Its
classification features could be enhanced, and a reduction of its power consumption could
be realised by adopting duty cycling policies. Moreover, the prototype could be equipped
with data saving capabilities to collect an on-site dataset, which could help to further refine
the ML models at the deployment site by adopting reinforcement learning paradigms.

<b>Data Availability Statement:</b> [Audio dataset](https://drive.google.com/file/d/15PQ-my8cA1blUIbAGRY8Jhq_d8Z7qim7/view), [picture dataset](https://drive.google.com/file/d/1QEAt4JiNxu5zZpXkWVnJm5sgtZm15Cf4/view?usp=share_link), and [video test](https://drive.google.com/file/d/1QEAt4JiNxu5zZpXkWVnJm5sgtZm15Cf4/view?usp=share_link).

