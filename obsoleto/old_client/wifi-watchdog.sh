#!/bin/bash
ping -c 1 8.8.8.8 > /dev/null
if [ $? != 0 ]; then
  logger "WiFi caiu, reiniciando wlan0"
  sudo ifconfig wlan0 down
  sleep 5
  sudo ifconfig wlan0 up
fi
