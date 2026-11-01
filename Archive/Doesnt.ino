

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char* ssid = "ESP32_Servo_AP";
const char* password = "12341234";

const unsigned int localPort = 8888;

WiFiUDP udp;
char packetBuffer[256];

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int SERVOMIN = 110;
const int SERVOMAX = 510;

const float STEP_SIZE = 0.2f;
const unsigned long STEP_INTERVAL = 20;

struct ServoConfig {
    uint8_t channel;
    float defaultAngle;
    bool isAllowed;
};

ServoConfig servoRegistry[16] = {
    {0,  70.0,  true},
    {1,  90.0,  true},
    {4,  90.0,  true},
    {13, 70.0,  true},
    {14, 90.0,  true},
    {15, 120.0, true}
};

bool enabled = false;

bool channelMoving[16] = {false};

float currentAngles[16] = {0};
float targetAngles[16] = {0};

unsigned long lastStep = 0;


// --------------------------------------------------
// SERVO FUNCTIONS
// --------------------------------------------------

uint16_t pulseFor(float angle) {
    angle = constrain(angle, 0.0f, 180.0f);

    return SERVOMIN +
        (uint16_t)(angle * (SERVOMAX - SERVOMIN) / 180.0f);
}

int getRegistryIndex(uint8_t ch) {
    for (int i = 0; i < 16; i++) {
        if (servoRegistry[i].isAllowed &&
            servoRegistry[i].channel == ch) {
            return i;
        }
    }

    return -1;
}


// --------------------------------------------------
// ARM
// --------------------------------------------------

void armServos() {

    if (enabled) {
        Serial.println("Already ARMED.");
        return;
    }

    Serial.println("ARMING...");

    for (int i = 0; i < 16; i++) {

        if (!servoRegistry[i].isAllowed)
            continue;

        uint8_t ch = servoRegistry[i].channel;

        float defAng = servoRegistry[i].defaultAngle;

        currentAngles[ch] = defAng;
        targetAngles[ch] = defAng;

        channelMoving[ch] = false;

        pwm.setPWM(ch, 0, pulseFor(defAng));

        Serial.print("CH ");
        Serial.print(ch);
        Serial.print(" initialized: ");
        Serial.println(defAng);
    }

    enabled = true;

    Serial.println("SYSTEM ARMED.");
}


// --------------------------------------------------
// OFF
// --------------------------------------------------

void disableServos() {

    enabled = false;

    for (uint8_t ch = 0; ch < 16; ch++) {

        pwm.setPWM(ch, 0, 4096);

        channelMoving[ch] = false;
    }

    Serial.println("SYSTEM OFF. ALL PWM OUTPUTS DISABLED.");
}


// --------------------------------------------------
// COMMAND PROCESSING
// --------------------------------------------------

void processCommand(char* text) {

    Serial.print("RECEIVED: ");
    Serial.println(text);

    if (strcmp(text, "ARM") == 0) {
        armServos();
        return;
    }

    if (strcmp(text, "OFF") == 0) {
        disableServos();
        return;
    }

    if (!enabled) {
        Serial.println("IGNORED: SYSTEM NOT ARMED.");
        return;
    }

    char* colon = strchr(text, ':');

    if (colon == NULL) {
        Serial.println("INVALID COMMAND FORMAT.");
        return;
    }

    *colon = '\0';

    char* endChannel;
    char* endAngle;

    long chInput = strtol(text, &endChannel, 10);
    float requested = strtof(colon + 1, &endAngle);

    if (*endChannel != '\0' ||
        endChannel == text ||
        *endAngle != '\0' ||
        endAngle == colon + 1 ||
        !isfinite(requested)) {

        Serial.println("INVALID COMMAND VALUES.");
        return;
    }

    if (chInput < 0 || chInput > 15) {
        Serial.println("INVALID CHANNEL.");
        return;
    }

    if (getRegistryIndex((uint8_t)chInput) == -1) {
        Serial.println("CHANNEL NOT AUTHORIZED.");
        return;
    }

    if (requested < 0 || requested > 180) {
        Serial.println("ANGLE OUT OF RANGE.");
        return;
    }

    targetAngles[chInput] = requested;

    channelMoving[chInput] =
        fabsf(targetAngles[chInput] - currentAngles[chInput]) > 0.001f;

    Serial.print("CH ");
    Serial.print(chInput);
    Serial.print(" TARGET: ");
    Serial.println(requested);
}


// --------------------------------------------------
// UDP RECEIVER
// --------------------------------------------------

void receiveCommands() {

    while (true) {

        int packetSize = udp.parsePacket();

        if (!packetSize)
            break;

        int len = udp.read(
            (uint8_t*)packetBuffer,
            sizeof(packetBuffer) - 1
        );

        if (len <= 0)
            continue;

        packetBuffer[len] = '\0';

        // Remove trailing whitespace
        while (len > 0) {

            char c = packetBuffer[len - 1];

            if (c == '\n' || c == '\r' ||
                c == ' ' || c == '\t') {

                packetBuffer[--len] = '\0';

            } else {
                break;
            }
        }

        // Remove leading whitespace
        char* command = packetBuffer;

        while (*command == ' ' || *command == '\t')
            command++;

        if (*command != '\0')
            processCommand(command);
    }
}


// --------------------------------------------------
// MOVEMENT ENGINE
// --------------------------------------------------

void updateServos() {

    if (!enabled)
        return;

    unsigned long now = millis();

    if (now - lastStep < STEP_INTERVAL)
        return;

    lastStep = now;

    for (uint8_t ch = 0; ch < 16; ch++) {

        if (!channelMoving[ch])
            continue;

        float delta = targetAngles[ch] - currentAngles[ch];

        float next;

        if (fabsf(delta) <= STEP_SIZE) {
            next = targetAngles[ch];
        }
        else {
            next = currentAngles[ch] +
                (delta > 0 ? STEP_SIZE : -STEP_SIZE);
        }

        currentAngles[ch] = next;

        pwm.setPWM(ch, 0, pulseFor(next));

        if (fabsf(currentAngles[ch] - targetAngles[ch]) < 0.001f) {

            channelMoving[ch] = false;

            Serial.print("CH ");
            Serial.print(ch);
            Serial.println(" MOVEMENT COMPLETE.");
        }
    }
}


// --------------------------------------------------
// SETUP
// --------------------------------------------------

void setup() {

    Serial.begin(115200);

    Wire.begin(21, 22);

    if (!pwm.begin()) {
        Serial.println("PCA9685 NOT FOUND!");
        while (true) {
            delay(1000);
        }
    }

    pwm.setPWMFreq(50);

    for (uint8_t ch = 0; ch < 16; ch++) {
        pwm.setPWM(ch, 0, 4096);
    }

    WiFi.mode(WIFI_AP);

    WiFi.softAP(ssid, password);

    udp.begin(localPort);

    Serial.println();
    Serial.println("----------------------------");
    Serial.println("ESP32 SERVO SERVER READY");
    Serial.print("IP: ");
    Serial.println(WiFi.softAPIP());
    Serial.print("PORT: ");
    Serial.println(localPort);
    Serial.println("WAITING FOR ARM COMMAND...");
    Serial.println("----------------------------");
}


// --------------------------------------------------
// MAIN LOOP
// --------------------------------------------------

void loop() {

    receiveCommands();

    updateServos();

}
