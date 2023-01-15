/*
  Portenta - TestSDCARD

  The sketch shows how to mount an SDCARD and list its content.

  The circuit:
   - Portenta H7 + Vision Shield
   - Portenta H7 + Portenta Breakout

  This example code is in the public domain.
*/
#include "SDMMCBlockDevice.h"
#include "FATFileSystem.h"
#define alarmfile "/fs/last_alarm.txt"

SDMMCBlockDevice block_device;
mbed::FATFileSystem fs("fs");

char cToStr[2];
unsigned char c;
char payload[85];
char temp_scores[35];
char temp_fscore[10];

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Mounting SDCARD...");
  int err =  fs.mount(&block_device);
  if (err) {
    // Reformat if we can't mount the filesystem
    // this should only happen on the first boot
    Serial.println("No filesystem found, formatting... ");
    err = fs.reformat(&block_device);
  }
  if (err) {
     Serial.println("Error formatting SDCARD ");
     while(1);
  }
  
  unsigned char c;
  FILE *fd = fopen(alarmfile, "r");
  while (!feof(fd)){                           // while not end of file
        c=fgetc(fd);
        if((char)c == 33) break; // get a character/byte from the file
        Serial.print((char)c);                    // show it as a text character
  }
  fclose(fd);
  bootM4();
   
}

void loop() {

  delay(5000);
  temp_fscore = "";
  temp_scores="";
  payload="";
  cToStr[1] = '\0';
  
  for(int i=0;i<5;i++){
    float score = 0.1
    if(i <4){
      sprintf(temp_fscore,"%.03f-", score);
      strcat(temp_scores,temp_fscore);
    }
    if(i==4){
      sprintf(temp_fscore,"%.03f!", score);
      strcat(temp_scores,temp_fscore);
    }
    score+=0.1;
  }
  strcpy(payload, "FIRE DETECTED!Video scores:");
  strcat(payload, temp_scores);
  strcat(payload, "Audio scores:");
  FILE *fd = fopen(alarmfile, "r");
  while (!feof(fd)){                           // while not end of file
      c=fgetc(fd);
      if((char)c == 33) break; // get a character/byte from the file
      cToStr[0] = (char)c;
      strcat(payload,cToStr); 
  }
  Serial.println(payload);
}
