from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='quanfima',
      version='0.1a3',
      description='The package for morphological analysis and visualization of fibrous materials.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='biomaterials fiber analysis material science visualization',
      url='http://github.com/rshkarin/quanfima',
      author='Roman Shkarin, Andrei Shkarin',
      author_email='roman.v.shkarin@gmail.com, andrei.shkarin@gmail.com',
      license='MIT',
      packages=['quanfima', 'docs'],
      python_requires='>=2.6,<3',
      install_requires=[
          'matplotlib==2.0.2',
          'numpy>=1.13.3',
          'pandas>=0.19.2',
          'scikit-image>=0.12.3',
          'scikit-learn>=0.18.1',
          'scipy>=0.19.0',
          'imageio>=2.3.0'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
