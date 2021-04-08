import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='stonesoup_kaggle_drone_tracking',
    author='Defence Science Technology Laboratory',
    author_email='oss@dstl.gov.uk',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha'
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=[
          'stonesoup~=0.1b5',
      ],
    entry_points={'stonesoup.plugins':
        'kaggle_drone = stonesoup_kaggle_drone_tracking',
    }
)
