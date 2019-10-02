# Copyright (C) 2019 Project AGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Installation Instructions."""

from setuptools import setup, find_packages
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

install_requires = [
    'pagi >= 0.1.0',
    'opencv-python',
    'h5py',
    'tqdm'
]

class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

setup(
    name='pagi-aha',
    version='0.1.0',
    author='ProjectAGI',
    author_email='info@agi.io',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/ProjectAGI/aha',
    license='Apache 2.0',
    description='Artificial Hippocampus Algorithm.',
    install_requires=install_requires,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    keywords='aha artificial hippocampus algorithm pagi'
)
