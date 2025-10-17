import os
import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from conversion_utils import (
    Conversion,
    apply_conversion,
    update_manifest_model_name,
)

def convert_to_coreml(model_path):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        return_dict=False,
        torchscript=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-large")
    tokenized = tokenizer(
        ["Sample input text to trace the model"],
        return_tensors="pt",
        max_length=512,  # token sequence length
        padding="max_length",
    )

    traced_model = torch.jit.trace(
        model,
        (tokenized["input_ids"], tokenized["attention_mask"])
    )

    outputs = [ct.TensorType(name="output")]

    mlpackage = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                f"{name}",
                shape=tensor.shape,
                dtype=np.float32,
            )
            for name, tensor in tokenized.items()
        ],
        outputs=outputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    return mlpackage


def convert(conversion_type, model_path, saved_name):
    model = convert_to_coreml(model_path)
    try:
        new_model = apply_conversion(model, conversion_type)
    except ValueError as error:
        print(error)
        return

    saved_path = f"{saved_name}.mlpackage"
    new_model.save(saved_path)

    manifest_file = os.path.join(saved_path, "Manifest.json")
    update_manifest_model_name(manifest_file, saved_name)


model_name = "oliverguhr/fullstop-punctuation-multilang-large"
#convert(Conversion.QUANTIZE, model_name, "FullStopPunctuation")
convert(Conversion.NONE, model_name, "FullStopPunctuation")
