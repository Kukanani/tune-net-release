from distutils.core import setup

setup(
    name='tune',
    version='0.0dev',
    packages=['tune'],
    license='Unreleased/Proprietary',
    long_description=open('README.md').read(),
    requires=['matplotlib', 'torch', 'torchvision', ]
)
