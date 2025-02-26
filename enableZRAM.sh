#!/usr/bin/env bash

# -----------------------------------------------------------
#  enableZRAM.sh
#
#  This script sets up a zram-based swap occupying
#  half of the available RAM using lz4 compression.
#  It also checks for any existing /dev/zram0 usage
#  and cleans it up before proceeding.
#
#  Usage:
#    sudo ./enableZRAM.sh
#
#  Cleanup manually:
#    sudo swapoff /dev/zram0
#    echo 1 | sudo tee /sys/block/zram0/reset
# -----------------------------------------------------------

echo "Setting up temporary zram swap..."
if [[ "$EUID" -ne 0 ]]; then
  echo "Error: Please run as root (e.g., sudo $0)"
  exit 1
fi

# --- Step 1: Load zram module
modprobe zram

# --- Step 2: If zram0 is in use, clean up
if swapon --summary | grep -q "/dev/zram0"; then
  echo "Detected existing zram0 swap; turning it off..."
  swapoff /dev/zram0 2>/dev/null
fi

if [[ -e /sys/block/zram0/reset ]]; then
  echo "Resetting existing zram0 device..."
  echo 1 > /sys/block/zram0/reset
fi

# --- Step 3: Determine half of the available RAM (in bytes)
ZRAM_SIZE=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') * 1024 / 2 ))

# --- Step 4: Configure zram0
echo "Configuring zram0 with lz4 compression and size $((ZRAM_SIZE / 1024 / 1024)) MB..."
echo "lz4" > /sys/block/zram0/comp_algorithm
echo "$ZRAM_SIZE" > /sys/block/zram0/disksize

# --- Step 5: Create and enable the swap area on /dev/zram0
# ----with prioirity -p 100 being higher than other available swap spaces
mkswap -L zram-swap /dev/zram0 > /dev/null 2>&1
swapon /dev/zram0 -p 100

# --- verification/info
echo "zram swap created and activated:"
swapon --summary
