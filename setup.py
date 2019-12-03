from setuptools import setup, find_packages

with open('README.md') as f:
	readme = f.read()

with open('LICENSE') as f:
	license = f.read()

with open('requirements.txt') as f:
	requirements = f.read()

setup(
	name='dicom_data_preprocess',
	version='0.0.2',
	description='The dicom data preprocessing block of the Medical Image Segmentation Analysis Pipeline',
	longdescription=readme,
	author='Christine Hsu',
	author_email='christine.wy.hsu@gmail.com',
	url='https://github.com/hsuchristine',
	license=license,
	packages=find_packages(exclude=('tests')),
	install_required=requirements
	)