#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* ssid = "ESP32_Servo_AP";
const char* password = "12341234"; 

const unsigned int localPort = 8888;
WiFiUDP udp;
char packetBuffer[256]; // Fixed syntax typo from previous string literal declaration

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
const int SERVOMIN = 110, SERVOMAX = 510;

struct ServoConfig {
  uint8_t channel;
  float defaultAngle;
  bool isAllowed;
};

ServoConfig servoRegistry[16] = {
  {0,  0.0,   false},  // Gripper explicitly forced to 0.0
  {1,  90.0,  true},  
  {4,  90.0,  true},  // CH 4 added: Change 90.0 to your preferred startup angle
  {13, 90.0,  true},  // CH 13 added: Change 90.0 to your preferred startup angle
  {14, 135.0, true},  
  {15, 150.0, true}   
  // Any channels not defined here default to false and remain dead
};

// INDIVIDUAL TRACKING ARRAYS FOR EACH CHANNEL
bool enabled = false;
bool channelMoving[16] = {false};
float currentAngles[16] = {150.0};
float targetAngles[16] = {150.0};
unsigned long lastStep = 0;

uint16_t pulseFor(float angle) {
  return SERVOMIN + (uint16_t)(angle * (SERVOMAX - SERVOMIN) / 180.0f);
}

int getRegistryIndex(uint8_t ch) {
  for (int i = 0; i < 16; i++) {
    if (servoRegistry[i].isAllowed && servoRegistry[i].channel == ch) {
      return i;
    }
  }
  return -1; 
}

void processCommand(char* text) {
  Serial.print("VERIFY - Port Payload: ");
  Serial.println(text);

  if (strcmp(text, "ARM") == 0) {
    if (enabled) return;
    
    Serial.println("ARMING ALLOWED DICTIONARY CHANNELS...");
    for (int i = 0; i < 16; i++) {
      if (servoRegistry[i].isAllowed) {
        uint8_t ch = servoRegistry[i].channel;
        float defAng = servoRegistry[i].defaultAngle;
        
        pwm.setPWM(ch, 0, pulseFor(defAng));
        currentAngles[ch] = defAng;
        targetAngles[ch] = defAng;
        channelMoving[ch] = false;
        
        Serial.print("Initialized Config CH ");
        Serial.print(ch);
        Serial.print(" to ");
        Serial.print(defAng);
        Serial.println(" deg.");
      }
    }
    enabled = true;
    Serial.println("System ARMED safely.");
    return;
  }
  
  if (strcmp(text, "OFF") == 0) {
    enabled = false;
    for (uint8_t i = 0; i < 16; i++) {
      pwm.setPWM(i, 0, 4096); 
      channelMoving[i] = false;
    }
    Serial.println("All channels safely powered down.");
    return;
  }

  if (!enabled) {
    Serial.println("Ignored: System is not ARMED yet.");
    return;
  }

  char* colonToken = strchr(text, ':');
  if (colonToken != NULL) {
    *colonToken = '\0'; 
    int chInput = atoi(text);
    char* end;
    float requested = strtof(colonToken + 1, &end);

    if (chInput >= 0 && chInput <= 15 && end != (colonToken + 1) && !*end && isfinite(requested)) {
      if (getRegistryIndex(chInput) == -1) {
        Serial.print("REJECTED: Channel ");
        Serial.print(chInput);
        Serial.println(" is NOT in allowed internal dictionary!");
        return;
      }

      // Track target instructions specifically for this channel slot
      targetAngles[chInput] = requested;
      channelMoving[chInput] = true;
      
      Serial.print("Driving CH ");
      Serial.print(chInput);
      Serial.print(" toward ");
      Serial.print(requested);
      Serial.println(" deg.");
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  
  if (pwm.begin()) {
    pwm.setPWMFreq(50);
    for (uint8_t i = 0; i < 16; i++) {
      pwm.setPWM(i, 0, 4096);
    }
  }

  WiFi.softAP(ssid, password);
  udp.begin(localPort);
  Serial.print("SoftAP Server Setup Complete. Local IP: ");
  Serial.println(WiFi.softAPIP());
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read((uint8_t*)packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = '\0';
      processCommand(packetBuffer);
    }
  }

  // MULTI-CHANNEL BACKGROUND RAMP ENGINE
  unsigned long now = millis();
  if (now - lastStep >= 20) {
    lastStep = now;
    
    for (uint8_t ch = 0; ch < 16; ch++) {
      if (channelMoving[ch]) {
        float delta = targetAngles[ch] - currentAngles[ch];
        float next = fabsf(delta) <= .2f ? targetAngles[ch] : currentAngles[ch] + (delta > 0 ? .2f : -.2f);
        
        pwm.setPWM(ch, 0, pulseFor(next));
        currentAngles[ch] = next;
        
        if (currentAngles[ch] == targetAngles[ch]) {
          channelMoving[ch] = false;
          Serial.print("CH ");
          Serial.print(ch);
          Serial.println(" - Target step index reached.");
        }
      }
    }
  }
}
