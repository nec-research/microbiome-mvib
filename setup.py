from setuptools import find_packages, setup


def _safe_read_lines(f, remove_git_lines=True):
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]

    if remove_git_lines:
        r = [l for l in r if not l.startswith("git+ssh")]

    return r


def readme():
    with open('README.md') as f:
        return f.read()


def description():
    description = (
        """This package includes the Multimodal Variational Information Bottleneck algorithm,
        used for microbiome-based disease prediction. 
        
        See: https://www.biorxiv.org/content/10.1101/2021.06.08.447505v1"""
    )
    return description


install_requires = _safe_read_lines("./requirements.txt")

setup(
    name='microbiome_mvib',
    version='0.1.0',
    description=description(),
    long_description=readme(),
    keywords="microbiome",
    url="https://github.com/nec-research/microbiome-mvib",
    author="NEC Laboratories Europe GmbH",
    author_email="filippo.grazioli@neclab.eu",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers/Researcher',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
