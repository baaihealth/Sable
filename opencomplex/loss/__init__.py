# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import importlib as importlib

_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [
    os.path.basename(f)[:-3]
    for f in _files
    if os.path.isfile(f) and not f.endswith("__init__.py")
]
_modules = [(m, importlib.import_module("." + m, __name__)) for m in __all__]
for _m in _modules:
    globals()[_m[0]] = _m[1]

# Avoid needlessly cluttering the global namespace
del _files, _m, _modules
