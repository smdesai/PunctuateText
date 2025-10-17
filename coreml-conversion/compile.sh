#!/bin/sh
set -eu

BASE="."
MODELS="Punctuate FullStopPunctuation"
COMPILED_DIR="MLModels"

rm -fr "${COMPILED_DIR}"
mkdir -p "${COMPILED_DIR}"

for MODEL_NAME in ${MODELS}; do
    MODEL="${BASE}/${MODEL_NAME}.mlpackage"
    MODEL_FILE="${MODEL}/Data/com.apple.CoreML/${MODEL_NAME}.mlmodel"
    MODEL_COMPILED_DIR="${COMPILED_DIR}"

    if [ ! -d "${MODEL}" ]; then
        echo "Skipping ${MODEL_NAME}: ${MODEL} not found."
        continue
    fi

    if [ ! -f "${MODEL_FILE}" ]; then
        echo "Skipping ${MODEL_NAME}: ${MODEL_FILE} not found."
        continue
    fi

    mkdir -p "${MODEL_COMPILED_DIR}"

    echo "Compiling ${MODEL_NAME}..."
    xcrun coremlcompiler compile "${MODEL_FILE}" "${MODEL_COMPILED_DIR}"
done
