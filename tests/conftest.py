import os

import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    run_model = os.environ.get("QIL_MODEL_TESTS") == "1"
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    skip_model = pytest.mark.skip(reason="set QIL_MODEL_TESTS=1 to run tests that load HF models")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "model" in item.keywords and not run_model:
            item.add_marker(skip_model)
