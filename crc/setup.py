from setuptools import setup
from setuptools import find_packages

setup(
    name="crc",
    version="0.1.0",
    # install_requires=['Your-Library'],
    packages=find_packages("crc"),
    package_dir={"": "crc"},
    # url='https://github.com/your-name/your_app',
    # license='MIT',
    # author='Your NAME',
    # author_email='your@email.com',
    description="crc",
)
