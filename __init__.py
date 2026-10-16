import os

import folder_paths

folder_paths.add_model_folder_path("qwen3asr", os.path.join(folder_paths.models_dir, "qwen3asr"))

from .nodes_qwen3asr import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
