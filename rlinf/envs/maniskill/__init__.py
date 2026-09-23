# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.util
import os
import pathlib
import pkgutil


def import_all_tasks():
    package_name = __name__ + ".tasks"
    package_path = pathlib.Path(__file__).parent / "tasks"

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"{package_name}.{module_name}")

    try:
        importlib.import_module("mani_skill_gs")
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        importlib.import_module("tasks.task_UprightStack")
    except (ModuleNotFoundError, ImportError):
        # RoboFPE's tasks package imports legacy tasks eagerly. Some of those
        # tasks target an older ManiSkill API, so load UprightStack directly.
        task_file = pathlib.Path(
            os.environ.get("ROBOFPE_ROOT", "/data/yingxi/RoboFPE")
        ) / "mani_envs" / "tasks" / "task_UprightStack.py"
        if task_file.is_file():
            # UprightStack-v1 uses TableSceneBuilder; this import is only a
            # legacy symbol needed while the file defines unused gen variants.
            table_module = importlib.import_module("mani_skill.utils.scene_builder.table")
            if not hasattr(table_module, "noTableSceneBuilder"):
                table_module.noTableSceneBuilder = table_module.TableSceneBuilder
            spec = importlib.util.spec_from_file_location(
                "robofpe_task_UprightStack", task_file
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load RoboFPE task file: {task_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

import_all_tasks()
