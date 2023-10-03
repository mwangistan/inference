"""
coreml backend (https://coremltools.readme.io/)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation
import coremltools as ct
import backend
import torch
from torchvision import transforms

class BackendCoreML(backend.Backend):
    def __init__(self, args):
        super(BackendCoreML, self).__init__()
        self.device = args.device

    def version(self):
        return ct.__version__

    def name(self):
        """Name of the runtime."""
        return "coreml"

    def image_format(self):
        """image_format."""
        return "NCHW"

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
        img = torch.tensor(feed[key]).squeeze(0)
        return self.model.predict({key: transforms.ToPILImage()(img)})
        
