"""
coreml backend (https://coremltools.readme.io/)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation
import coremltools as ct
import backend

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
        if self.device == "cpu":
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        elif self.device == "cpu_and_gpu":
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        elif self.device == "cpu_and_npu":
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
        return self.model.predict(feed)
        
