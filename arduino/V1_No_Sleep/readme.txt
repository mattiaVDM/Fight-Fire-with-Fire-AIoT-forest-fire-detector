To make the compilation work it is needed to add the Arduino zipped libraries for both the audio and video models available at:
\Modelli\2CLASSES\MFE-CONV1D-2C\MFE-CONV1D-2C.zip
\Tesi\Modelli\VIDEO\ei-fireimagev2-arduino-1.0.2.zip
From Arduino ide, split memory like this:  1MB dedicated to M4 Core and 1MB dedicated to M7 Core

LoRa MKRWAN library needs this fix to work on the Portenta H7 Vision Shield:
https://github.com/arduino-libraries/MKRWAN/pull/93
