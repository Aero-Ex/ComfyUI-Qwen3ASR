import os

import folder_paths

for _key, _path in (
    ("qwen3asr", os.path.join(folder_paths.models_dir, "qwen3asr")),
):
    folder_paths.add_model_folder_path(_key, _path)

from .nodes_qwen3asr import NODE_CLASS_MAPPINGS as ASR_NODES, NODE_DISPLAY_NAME_MAPPINGS as ASR_NAMES
from .nodes_qwen3align import NODE_CLASS_MAPPINGS as ALIGN_NODES, NODE_DISPLAY_NAME_MAPPINGS as ALIGN_NAMES

NODE_CLASS_MAPPINGS = {**ASR_NODES, **ALIGN_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**ASR_NAMES, **ALIGN_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
