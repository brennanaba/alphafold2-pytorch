from setuptools import setup, find_packages

setup(
    name='alphafold2',
    version='0.0.1',
    description='Set of functions from AF2 paper',
    license='BSD 3-clause license',
    maintainer='Brennan Abanades',
    maintainer_email='brennan.abanadeskenyon@stx.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('alphafold2', 'alphafold2.*')),
    install_requires=[
        'numpy',
        'einops>=0.3',
        'torch>=1.6',
        'plotly',
    ],
)