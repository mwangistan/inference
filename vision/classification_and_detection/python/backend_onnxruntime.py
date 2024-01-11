"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import os
import backend


class BackendOnnxruntime(backend.Backend):
    def __init__(self, args):
        super(BackendOnnxruntime, self).__init__()
        self.profiling = args.enable_profiling
        self.threads = args.intra_op_threads
        self.graph_optimization_level = args.graph_optimization_level
        self.thread_spinning = args.thread_spinning
        self.execution_mode = args.execution_mode

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        opt = rt.SessionOptions()

        # Tracing/profiling
        if self.profiling:
            opt.enable_profiling = True

        # Graph optimization
        if self.graph_optimization_level == "ORT_ENABLE_ALL":
            opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.graph_optimization_level == "ORT_DISABLE_ALL":
            opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        if self.graph_optimization_level == "ORT_ENABLE_BASIC":
            opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if self.graph_optimization_level == "ORT_ENABLE_EXTENDED":
            opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Thread execution mode
        if self.execution_mode == "ORT_SEQUENTIAL":
            opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        if self.execution_mode == "ORT_PARALLEL":
            opt.execution_mode = rt.ExecutionMode.ORT_PARALLEL

        # Thread spinning
        if self.thread_spinning:
            opt.add_session_config_entry("session.intra_op.allow_spinning", "1")
        else:
            opt.add_session_config_entry("session.intra_op.allow_spinning", "0")

        # Number of threads
        opt.intra_op_num_threads = self.threads

        # By default all optimizations are enabled
        # https://onnxruntime.ai/docs/performance/graph-optimizations.html
        # Enable only upto extended optimizations on aarch64 due to an accuracy issue
        if os.environ.get("HOST_PLATFORM_FLAVOR", "") == "aarch64":
            opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # self.sess = rt.InferenceSession(model_path, opt)
        if len(rt.get_all_providers()) > 1 and os.environ.get("USE_GPU", "yes").lower() not in [ "0", "false", "off", "no" ]:
            self.sess = rt.InferenceSession(model_path, opt, providers=["CUDAExecutionProvider"])
        else:
            self.sess = rt.InferenceSession(model_path, opt, providers=["CPUExecutionProvider"])
            
        # get input and output names
        if not inputs:
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(self.outputs, feed)