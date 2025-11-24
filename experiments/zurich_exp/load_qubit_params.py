import importlib.util
import os


def load_qubit_params(file_path):
    file_name = "qubit_params"
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
