#!/bin/bash

# Use from the repository root directory
# Writes OCR preditions into the <dataset_path>/pretrained_ocr_predictions/ directory

if [[ $# < 2 ]]; then
	echo "Usage: <ocr_config> <dataset_path>"
	exit -1
fi

OCR_CONFIG="$1"
DATASET_PATH="$2"
OUT_DATASET_PATH="$2"

python user_scripts/parse_folder.py \
    -c                   "$OCR_CONFIG" \
    -x                   "$DATASET_PATH/page_xml/" \
    -i                   "$DATASET_PATH/images/" \
    --output-xml-path    "$OUT_DATASET_PATH/pretrained_ocr_predictions/page_xml" \
    --output-logit-path  "$OUT_DATASET_PATH/pretrained_ocr_predictions/logits" \
    --output-line-path   "$OUT_DATASET_PATH/pretrained_ocr_predictions/lines" \
    --output-transcriptions-file-path "$OUT_DATASET_PATH/pretrained_ocr_predictions/transcriptions.txt"
