import os 
from os.path import basename, splitext
from glob import glob
from setuptools import setup
from setuptools import find_packages


setup(
    name="pl_deep_sort",
    version = "0.0", 
    description = "all code needed to run pl deep sort",
    keywords = "deep_sort",
    packages = find_packages("src"),
    package_dir = {'': "src"},
    py_modules = [splitext(basename(path))[0] for path in glob('src/*.py')]
)