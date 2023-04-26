import os
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()


setup(
    name='NeuralNetPy',
    version='0.0.4',
    author='Joel Maldonado-Ruiz',
    author_email='jmaldonadoruiz0@gmail.com',
    description='NeuralNetPy is a neural network library created using numpy that allows you to create, train, and save deep learning models.',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    license='GNU General Public License v3.0',
    keywords=['Neural Network', 'Python', 'AI', 'Numpy', 'Deep Learning'],
    packages=find_packages(),
    install_requires=['numpy >= 1.23.5', 'matplotlib >= 3.6.3'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
