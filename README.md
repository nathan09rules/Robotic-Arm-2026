# Robotic-Arm-2026

This Github Repo comprises of 3 seperate part that are yet to be completely integrated together this requiring forther application of use:

## 1) Hand Detection Algorithm
uses 2 cammeras positiona set distance away with rotation and the cameras properities calibrated for every insitialtion location

## 2) Robotic arm Inverse Kinematics
Uses Inverese kinematic library form ikpy to make the required roations required for the arm to reach to designated positon 

## 3) 3D arm library 
Uses blender intented to visualise the current state the porgram thinks the arm is in



# REMEMBER
- the current esp32 does not accept -1 as an input change that
- sudo chmod 666 /dev/ttyUSB0


# Pending
- The fucking IK does not work at all now with the hand point coords fucking hell
- camera UI visuals are still fucking missing
- have not finsihed fucking head of the thing still


# Channels base config
Photo 1
  {0,  70.0,  true},
  {1,  90.0,  true},  
  {4,  200.0,  true},   
  {13, 70.0,  true},    
  {14, 90.0, true},  
  {15, 120.0, true}   


Photo 2
  {0,  70.0,  true},
  {1,  90.0,  true},  
  {4,  200.0,  true},   
  {13, 70.0,  true},    
  {14, 90.0, true},  
  {15, 120.0, true}  


Photo 3
  {0,  70.0,  true},
  {1,  90.0,  true},  
  {4,  200.0,  true},   
  {13, 90.0,  true},     l 
  {14, 90.0, true},  
  {15, 80.0, true}  

Photo 4
  {0,  70.0,  true},
  {1,  90.0,  true},  
  {4,  90.0,  true},   
  {13, 90.0,  true},    
  {14, 90.0, true},  
  {15, 90.0, true}  

Photo 5
  {0,  70.0,  true},
  {1,  90.0,  true},  
  {4,  200.0,  true},   
  {13, 70.0,  true},    
  {14, 90.0, true},  
  {15, 120.0, true}  
