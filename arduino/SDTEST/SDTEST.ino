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
#define txt ".txt"
#define root "/fs/"

SDMMCBlockDevice block_device;
mbed::FATFileSystem fs("fs");
uint16_t photo = 0;

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
  }
 
}

void loop() {

  delay(3000);
  
  char filename[10];
  char pic_num[5];
  itoa(photo, pic_num, 10);
  strcpy(filename,root);
  strcat(filename,pic_num);
  strcat(filename,txt);
  
  FILE *fd = fopen(filename, "a");
  
  String file = String(filename);
  Serial.print("Trying to create ");
  Serial.println(file);
  if(fd==NULL) {
    // Something was wrong
    Serial.println("Error creating file\n");
  }
  else {
          Serial.println("File created\n");
          fprintf(fd, "Welcome SD!\n");
          fclose(fd);
  }
  photo+=1;
  
}
