#include "Arduino.h"
#include "RPC.h"

void setup() {
  Serial.begin(115200);
  RPC.begin();
}

void loop() {
  String data = "";
  while (RPC.available()) {
    data += (char)RPC.read();
  }
  if (data != "") {
    Serial.write(data.c_str(), data.length());
    if(data.charAt(0) == '*'){
      Serial.println("ALARM RECEIVED ON M7");
    }
  }
}
