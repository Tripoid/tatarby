"""
Setup script for the Tatarby Russian-Tatar translation system.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tatarby",
    version="1.0.0",
    author="Tatarby Team",
    description="High-quality Russian to Tatar neural machine translation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tripoid/tatarby",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tatarby-train=scripts.train:main",
            "tatarby-eval=scripts.eval_model:main",
            "tatarby-translate=scripts.inference:main",
            "tatarby-demo=scripts.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine translation",
        "neural machine translation",
        "russian",
        "tatar",
        "transformers",
        "nlp",
        "artificial intelligence"
    ],
)