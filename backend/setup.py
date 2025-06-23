from setuptools import setup, find_packages

setup(
    name="rag_backend",
    version="0.1",
    packages=find_packages(where="app"),
    package_dir={"": "app"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "chromadb",
        "sentence-transformers",
        "pypdf",
        "whisper",
        "moviepy",
        "python-multipart",
    ],
)