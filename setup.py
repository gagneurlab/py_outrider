import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_outrider", 
    version="0.0.1",
    author="Stefan Loipfinger, Ines Scheller",
    author_email="scheller@in.tum.de",
    description="Python backend package for OUTRIDER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cmm.in.tum.de/gagneurlab/py_outrider/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
