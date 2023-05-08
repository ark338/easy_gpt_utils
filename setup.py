from setuptools import setup, find_packages

setup(
    name="easy_gpt_utils",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "numpy"
    ],
    author="Hou Wei",
    author_email="messenger929@163.com",
    description="Easy GPT utils include 1. chat completion 2. embedding and 3. vector database",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ark338/easy_gpt_utils",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
