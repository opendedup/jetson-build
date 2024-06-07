#!/bin/bash

# Jetson Orin fan speed script for Ubuntu 22.04

# Function to read fan speed from sysfs
get_fan_speed() {
  local fan_speed_raw=$(cat /sys/devices/pwm-fan/target_pwm)
  local fan_speed=$(( fan_speed_raw * 255 / 100 )) # Convert to 0-255 range
  echo "$fan_speed"
}

# Get current fan speed
current_fan_speed=$(get_fan_speed)

# Output fan speed
echo "Current fan speed: $current_fan_speed (out of 255)"
