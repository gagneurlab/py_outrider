from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="py_outrider",
    version="0.1.0",
    author="Stefan Loipfinger, Ines Scheller",
    author_email="scheller@in.tum.de",
    description="Python backend package for OUTRIDER2 R package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gagneurlab/py_outrider/",
    packages=find_packages(),
    install_requires=['tensorflow>=2.3.0',
                      'tensorflow-probability>=0.10.0',
                      'scikit-learn>=0.23.1',
                      'statsmodels>=0.11.1',
                      'numpy>=1.19.2',
                      'pandas>=1.1.5',
                      'anndata>=0.7.0',
                      'nipals>=0.5.2'
                      ],
    entry_points={
        "console_scripts": ['py_outrider = py_outrider.__main__:main']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
