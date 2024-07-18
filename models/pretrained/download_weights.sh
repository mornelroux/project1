#!/bin/bash

# Base URL for weight download
BASE_URL="https://github.com/THU-MIG/yolov10/releases/download/v1.1/"

# Function to download weights
download_weight() {
    local weight_type=$1
    local url="${BASE_URL}/yolov10${weight_type}.pt"
    echo "Downloading YOLOv10 ${weight_type} weight from ${url}"
    wget -q --show-progress -O "yolov10${weight_type}.pt" "$url"
    echo "Download Complete"
}

# Display help functions
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Download YOLOv10 pretrained weights"
    echo "Options:"
    echo "  n       Download YOLOv10n weight"
    echo "  s       Download YOLOv10s weight"
    echo "  m       Download YOLOv10m weight"
    echo "  b       Download YOLOv10b weight"
    echo "  l       Download YOLOv10l weight"
    echo "  x       Download YOLOv10x weight"
    echo "  all     Download all YOLOv10 weights"
    echo "  help    Show this help message"
}

# Check if no arguments are passed
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

#Handle arguments
case $1 in
    n)
        download_weight "n"
        ;;
    s)
        download_weight "s"
        ;;
    m)
        download_weight "m"
        ;;
    b)
        download_weight "b"
        ;;
    l)
        download_weight "l"
        ;;
    x)
        download_weight "x"
        ;;
    all)
        download_weight "n"
        download_weight "s"
        download_weight "m"
        download_weight "b"
        download_weight "l"
        download_weight "x"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Invalid option: $1"
        show_help
        exit 1
        ;;
esac