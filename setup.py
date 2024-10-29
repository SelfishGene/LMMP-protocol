from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lmmp-protocol",
    version="0.1.0",
    author="David Beniaguev",
    author_email="",
    description="LMMP - a Large Multimodal Model Protocol for encoding images as text for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SelfishGene/LMMP-protocol",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "Pillow",
        "numpy",
        "scikit-learn",
        "umap-learn",
        "google-generativeai",
        "python-dotenv",
    ],
)