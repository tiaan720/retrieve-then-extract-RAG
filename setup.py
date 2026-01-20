"""
Setup script for the Weaviate embedding pipeline.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="weaviate-embedding-pipeline",
    version="0.1.0",
    author="Your Name",
    description="A RAG pipeline with Weaviate vector database and Ollama embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tiaan720/retrieve-then-extract-RAG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
