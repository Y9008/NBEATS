
from setuptools import setup, Extension

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    
setup(
  name = 'NBEATS',        
  packages = ['NBEATS'],   
  version = '1.3.8',     
  license='MIT',
  description="This library uses nbeats-pytorch as base and accomplishes univariate time series forecasting using N-BEATS.",
  long_description=long_description,
  long_description_content_type='text/markdown' , 
  author = 'Yazdan Khan',                   
  author_email = 'yazdan0891@gmail.com',     
  url = 'https://github.com/Y9008/NBEATS',     keywords = ['nbeats', 'timeseries', 'forecast', 'neural beats' , 'univariate timeseries forecast ', 'timeseries forecast', 'univariate timeseries forecast'],   
  install_requires=[
          'nbeats-pytorch',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',         
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
  ],
  python_requires='>=3.6'
)
