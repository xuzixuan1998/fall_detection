#!/bin/bash

target_dir="data/URFall/images"
# Loop through the range of file numbers
for i in $(seq -w 01 30); do
  # Define the zip file name
  zip_file="fall-${i}-cam1-rgb.zip"
  unzip -q "$zip_file" -d "$target_dir"
done
