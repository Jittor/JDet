from setuptools import setup,find_packages
import os
path = os.path.join(os.path.dirname(__file__),"python")
setup(
    name="jdet",
    version="0.1",
    author="Jittor Group",
    author_email="jittor@qq.com",
    description="Jittor Aerial Image Detection",
    url="http://jittor.com",
    python_requires='>=3.7',
    packages=find_packages(path),
    package_dir={'': path},
    install_requires=[
        "shapely",
        "pyyaml",
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
        "jittor",
        "tensorboardX",
        "opencv-python",
        "tqdm",
        "pycocotools",
        "terminaltables",
    ],
)