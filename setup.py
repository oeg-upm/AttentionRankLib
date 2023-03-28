

from pathlib import Path

from setuptools import find_packages, setup

# requirements = Path(__file__).parent/ 'requirements.txt'

# with requirements.open(mode='rt', encoding='utf-8') as fp:
#    install_requires = [line.strip() for line in fp]

setup(
    name='attentionrank',
    version='0.0.1',
    description='AttentionRank library for Term Extraction',
    author='Pablo Calleja, David Viñas, Elena Montiel',
    email='p.calleja@upm.es',
    license='Apache 2',
    python_requrires='>=3.7',
    # packages=find_packages(
    #    where='src',
    #    include=['widaug','widaug.*'],  # alternatively: `exclude=['additional*']`
    # ),
    # package_dir={"": "src"},
    packages=['attentionrank'],
    package_dir={'': 'src'},
    # install_requires=[],
    setup_requires=[],
    # install_requires=install_requires,
    include_package_data=True,

    # test_requires=[]
    # test_suite='tests'

)