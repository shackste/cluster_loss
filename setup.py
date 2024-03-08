from setuptools import setup

setup(
    name='cluster_loss',
    version='0.1',
    description='Package for Pytorch Loss functions based on clusters in the target set with the purpose of fitting the target distribution',
    author='Stefan Hackstein',
    author_email='shackst3@gmail.com',
    packages=['cluster_loss'],
    install_requires=['kmeans_pytorch', 'torch', 'numpy', 'geomloss', 'pytorch_fid', 'corner', 'umap-learn']
)
