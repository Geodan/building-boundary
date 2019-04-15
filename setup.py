from setuptools import setup, find_packages


def read(filename):
    with open(filename) as f:
        return f.read()


setup(
    name="buildingboundary",
    version="0.2.0",
    author="Chris Lucas",
    author_email="chris.lucas@geodan.nl",
    description=(
        "A script to trace the boundary of a building (part) in a point cloud."
    ),
    license="MIT",
    keywords="building boundary trace point cloud",
    packages=find_packages(),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'shapely',
        'cgal-bindings'
    ],
    zip_safe=False
)
