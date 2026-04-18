#!/bin/bash
set -e

DEVICE="/dev/sda"
MAPPER_NAME="data8tb"
MOUNT_POINT="/media/igogo/8TB_A"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "Already mounted at $MOUNT_POINT"
    df -h "$MOUNT_POINT"
    exit 0
fi

# Check device exists
if [ ! -b "$DEVICE" ]; then
    echo "Error: $DEVICE not found. Is the disk connected?"
    exit 1
fi

# Unlock LUKS if not already open
if [ ! -b "/dev/mapper/$MAPPER_NAME" ]; then
    echo "Unlocking LUKS volume on $DEVICE..."
    sudo cryptsetup luksOpen "$DEVICE" "$MAPPER_NAME"
fi

# Create mount point
sudo mkdir -p "$MOUNT_POINT"

# Mount
echo "Mounting to $MOUNT_POINT..."
sudo mount "/dev/mapper/$MAPPER_NAME" "$MOUNT_POINT"

# Show result
echo "Mounted successfully."
df -h "$MOUNT_POINT"

# Check Samba status
echo ""
echo "=== Samba status ==="
if systemctl is-active --quiet smbd; then
    echo "smbd: running"
else
    echo "smbd: stopped — starting..."
    sudo systemctl start smbd
fi

if systemctl is-active --quiet nmbd; then
    echo "nmbd: running"
else
    echo "nmbd: stopped — starting..."
    sudo systemctl start nmbd
fi

echo ""
sudo smbstatus --brief 2>/dev/null || echo "No active connections."
