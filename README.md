# Fight Fire with Fire: Detecting Forest Fires with Embedded Machine Learning Models Dealing with Audio and Images on Low Power IoT Devices
Link to the article: https://sciprofiles.com/publication/view/816e32fcbe4eaee09695b20f9effb8a5

Giacomo Peruzzi, Alessandro Pozzebon  and Mattia Van Der Meer 

## Abstract 

Forest fires are the main cause of desertification, and they have a disastrous impact on agricultural and forest ecosystems. Modern fire detection and warning systems rely on several techniques: satellite monitoring, sensor networks, image processing, data fusion, etc. 
Recently, Artificial Intelligence (AI) algorithms have been applied to fire recognition systems, enhancing their efficiency and reliability. However, these devices usually need constant data transmission along with a proper amount of computing power, entailing high costs and energy consumption. 
This paper presents the prototype of a Video Surveillance Unit (VSU) for recognising and signalling the presence of forest fires by exploiting two embedded Machine Learning (ML) algorithms running on a low power device. The ML models take audio samples and images as their respective inputs, allowing for timely fire detection. 
The main result is that while the performances of the two models are comparable when they work independently, their joint usage according to the proposed methodology provides a higher accuracy, precision, recall and F1 score (96.15%, 92.30%, 100.00%, and 96.00%, respectively).
Eventually, each event is remotely signalled by making use of the Long Range Wide Area Network (LoRaWAN) protocol to ensure that the personnel in charge are able to operate promptly.

## Introduction

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

## Dataset Description 

Datasets Description
With the aim of recognising forest fires, two datasets were created: one containing audio files, and one containing images. Of course, both of the datasets included audio and pictures related to forest contexts (e.g., animal sounds, wind, fire, woods, etc.). 

### Audio Dataset

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

![image](https://user-images.githubusercontent.com/110095822/211930489-1d09551a-ea22-4dd7-beae-3749cb84905b.png)
