from setuptools import setup, find_packages

setup(
    name="dagnese-fno",
    version="0.1.0",
    author="Giovanni D'Agnese",
    author_email="jovannidagnese2@gmail.com",
    description="Diffeomorphic Implicit Fourier Neural Operators with D'Agnese Topological Barrier Loss",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GiovanniDAgnese-paper/DIF-FNO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
)
