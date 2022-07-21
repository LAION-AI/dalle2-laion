from setuptools import setup, find_packages

setup(
    name = "dalle2-laion",
    version = "0.0.1",
    packages = find_packages(exclude=[]),
    include_package_data = True,
    install_requires = [
        "packaging>=21.0",
        "pydantic>=1.9.0",
        "torch>=1.10",
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
        "click>=8.0.0"
        "dalle2-pytorch"
    ]
)
