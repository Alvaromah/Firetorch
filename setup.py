from setuptools import setup, find_packages

setup(
    name='firetorch',
    version='0.0.1',
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib',
        'opencv-python',
        'opencv-python-headless',
    ],
    packages=find_packages(exclude=("examples", "labs")),
)
