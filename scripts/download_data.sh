#!/usr/bin/env bash
set -euo pipefail

# Google Drive ID
FILE_ID="1y-Z-PbwG1BxJFQJoQEHj7A0e7LyyFyBL"

# local target dir and file name, please modify if you want to customize your download path
ZIP_NAME="proteingym.zip"
DATA_DIR="data"


# check if gdown is installed
if command -v gdown >/dev/null 2>&1; then
    echo "Using gdown to download file..."
else
    echo "Error: gdown is not installed. Please install it using the following command:"
    echo "  pip install gdown"
    exit 1
fi

gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${DATA_DIR}/${ZIP_NAME}"


unzip -o "${DATA_DIR}/$ZIP_NAME" -d "${DATA_DIR}/"
rm "${DATA_DIR}/$ZIP_NAME"

echo "donwloaded to ${DATA_DIR}/"
