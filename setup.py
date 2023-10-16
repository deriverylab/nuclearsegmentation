from setuptools import setup, find_packages

setup(name='nuclearsegmentation',
    version='1.0.0',
    description='Code from Watson et al., 2023, for segmenting cell nuclei',
    author='Joe Watson, Ben Porebski',
    url='https://github.com/deriverylab/nuclearsegmentation',
    scripts=["predict.py"],
    packages=find_packages())
