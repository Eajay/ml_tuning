from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
      long_description = f.read()

setup(name='ml_tuning',
      version='0.0.1',
      description='Tuning some machine learning function parameters',
      license='MIT',
      long_description=long_description,
      author='Yijiang Zheng',
      author_email='yijiang_zheng@outlook.com',
      url="https://github.com/Eajay/ml_tuning",
      download_url="https://github.com/Eajay/ml_tuning.git",
      # packages=['ml_tuning'],
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      install_requires=['numpy'],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Intended Audience :: Developers",      # Define that your audience are developers
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
      python_requires='>=3.6',
      )

