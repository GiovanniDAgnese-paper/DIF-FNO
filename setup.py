from setuptools import setup, find_packages

setup(
    name="dagnese-fno",
    version="1.0.0",
    author="Giovanni D'Agnese",
    author_email="jovannidagnese2@gmail.com",
    description="D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators with Topological Guarantees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GiovanniDagnese-paper/DIF-FNO",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
)
