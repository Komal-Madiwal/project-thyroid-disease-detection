from setuptools import find_packages,setup ## setup , find_packages = function , setuptools = module / lib
from typing import List


setup(
    name='thyroiddiseasedetection',
    version='0.0.1',
    author='komal madiwal',
    author_email='komalmadiwal10@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)