"""
coreml backend (https://coremltools.readme.io/)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation
import coremltools as ct
import backend
import os
from PIL import Image
import numpy as np

class BackendCoreML(backend.Backend):
    def __init__(self, args):
        super(BackendCoreML, self).__init__()
        self.device = args.device
        self.class_label_index = {}
        self.set_class_labels()

    def version(self):
        return ct.__version__

    def name(self):
        """Name of the runtime."""
        return "coreml"

    def image_format(self):
        """image_format."""
        return "NCHW"
    
    def set_class_labels(self):
        path = os.path.dirname(os.path.abspath(__file__))
        label_file = os.path.join(path, 'synset_words.txt')
        with open(label_file, 'r') as f:
            labels = f.readlines()

        for i, label in enumerate(labels):
            label = label.split()[1:]
            label = " ".join(label)
            self.class_label_index[label] = i

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        if self.device == "CPU":
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        elif self.device == "CPU_AND_GPU":
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        elif self.device == "CPU_AND_NPU":
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)

        # get input and output names
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = self.model.input_names
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = self.model.output_names

        return self

    def predict(self, feed):
        """Run the prediction."""
        key = [key for key in feed.keys()][0]
        img = feed[key]
        img = np.squeeze(img)
        img = Image.fromarray(np.uint8(img))
        out = self.model.predict({key: img})[self.outputs]
        output_index = self.class_label_index[out]
        return output_index
