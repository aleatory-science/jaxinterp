import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaxinterp",
    version="0.0.1",
    author="University of Copenhagen (Aleatory Science Team)",
    author_email="ahmad@di.ku.dk",
    description="High-level interpreter for JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aleatory-science/jaxinterp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'jax', 'jaxlib'
    ]
)