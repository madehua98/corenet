#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
from pathlib import Path
from typing import Any

# LIBRARY_ROOT is the folder that contains `corenet/` module.
LIBRARY_ROOT = Path(__file__).parent.parent

MIN_TORCH_VERSION = "1.11.0"

SUPPORTED_IMAGE_EXTNS = [".png", ".jpg", ".jpeg"]  # Add image formats here
SUPPORTED_VIDEO_CLIP_VOTING_FN = ["sum", "max"]
SUPPORTED_VIDEO_READER = ["pyav", "decord"]

DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_VIDEO_FRAMES = 8
DEFAULT_LOG_FREQ = 500

DEFAULT_ITERATIONS = 300000
DEFAULT_EPOCHS = 300
DEFAULT_MAX_ITERATIONS = DEFAULT_MAX_EPOCHS = 10000000

TMP_RES_FOLDER = "results_tmp"

TMP_CACHE_LOC = "/ML-A100/team/mm/models/tmp/corenet"
#TMP_CACHE_LOC = "/home/data_llm/madehua/tmp/corenet"

Path(TMP_CACHE_LOC).mkdir(parents=True, exist_ok=True)

DATACOMP_COUNT = 2906
LAION_COUNT = 2116
RECIPE_COUNT = 1373
CC12M_COUNT = 34
# DATA_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/datacomp_1b"
# LAION_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/laion2b"
# RECIPE_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/recipe1M+"
# CC12M_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/cc12m"

DATA_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/datacomp_1b_label"
LAION_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/laion2b_label"
RECIPE_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/recipe1M+_label"
CC12M_CACHE_DIR = "/ML-A100/team/mm/models/catlip_data/cc12m_label"
#DATA_CACHE_DIR = "/media/fast_data/catlip_data/cache"


Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def is_test_env() -> bool:
    """
    Returns:
        True iff the module is loaded by pytest.

    Note:
        In `conftest.py` file, we set CORENET_ENTRYPOINT=pytest.
    Previously, we used to rely on the existence of "PYTEST_CURRENT_TEST" variable,
    which is set automatically by pytest. But the issue was that the `conftest.py`
    itself and some fixtures are run before "PYTEST_CURRENT_TEST" gets set.
    """
    return os.environ.get("CORENET_ENTRYPOINT") == "pytest"


def if_test_env(then: Any, otherwise: Any) -> Any:
    return then if os.environ.get("CORENET_ENTRYPOINT") == "pytest" else otherwise
