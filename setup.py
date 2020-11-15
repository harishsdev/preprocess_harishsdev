import setuptools

with open('Readme.md','r') as file:
	long_description=file.read()

setuptools.setup(
    name='preprocess_harishsdev',
    version='0.0.1',
    author='Harish Shankam',
    author_email='harishsdev@gmail.com',
    description='This is preprocess package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages,
    classifiers=[
    'Programming Language::python::3',
    'License::OSI Aproved::MIT License',
    "operating system :: OS Independent",
    python_requires='>=3.5'
    ]
)