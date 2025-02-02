#!/bin/bash

# Specify the directory to search
directory="./"

# Find and delete empty .txt files
find "$directory" -type f -name "*.txt" -empty -delete

echo "Empty .txt files deleted from $directory"
