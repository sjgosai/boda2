from setuptools import find_namespace_packages, setup

setup(
    name="cifar10",
    author="Sager J. Gosai",
    version='0.0.0',
    author_email="sgosai@broadinstitute.org",
    description="CIFAR10 dummy model",
    packages=find_namespace_packages(include=["cifar10.*"]),
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sjgosai/boda2/analysis/SG00__getting_started"
)