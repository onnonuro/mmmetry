from setuptools import setup
from pip._internal.req import parse_requirements

requirements = parse_requirements('requirements.txt', session=False)
install_requires = [str(req.requirement) for req in requirements]

setup(
    name='mmmetry',
    version='0.0.1',
    author='onnonuro',
    packages=['mmmetry'],
    install_requires=install_requires,
    license='AGPL-3.0',
)