#! /usr/bin/env python
from numpy.distutils.core import setup

descr = """Workflow for the Swann Lab electrophysiology analyses"""

DISTNAME = 'SwannLabResources'
DESCRIPTION = descr
MAINTAINER = 'Alex Rockhill'
MAINTAINER_EMAIL = 'arockhil@uoregon.edu'
DOWNLOAD_URL = 'https://github.com/alexrockhill/SwannLabResources.git'
VERSION = '0.0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'swann'
          ],
          )
