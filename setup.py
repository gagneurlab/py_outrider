import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_outrider", 
    version="0.1.0",
    author="Stefan Loipfinger, Ines Scheller",
    author_email="scheller@in.tum.de",
    description="Python backend package for OUTRIDER R package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cmm.in.tum.de/gagneurlab/py_outrider/",
    packages=['py_outrider'],
    install_requires=['tensorflow>=2.1.0',
                      'tensorflow-probability>=0.8.0',
                      'scikit-learn',
                      'statsmodels',
                      'xarray',
                      'zarr',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'nipals'
                      ],
    entry_points = {
        "console_scripts": ['py_outrider = py_outrider.__main__:main']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
