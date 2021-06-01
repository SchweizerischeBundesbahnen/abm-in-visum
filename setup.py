from setuptools import setup

requirements = []
with open("requirements.txt", "r") as f:
    requirements += [s for s in [line.strip(" \n") for line in f]
                     if not s.startswith("#") and s != "" and not s.startswith("--")]

setup(
    name='ABM-in-Visum',
    version='1.0',
    description='Collaborative project for activity-based transport demand modelling within the PTV Visum software',
    license='LICENSE.txt',
    long_description=open('README.md').read(),
    author='Swiss Federal Railyways, PTV Group',
    author_email='patrick.manser@sbb.com',
    packages=['abminvisum'],
    install_requires=requirements,
)
