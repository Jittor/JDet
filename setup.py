from setuptools import setup
import os
path = os.path.dirname(__file__)

setup(
    name="jdet",
    version="0.1",
    author="Jittor Group",
    author_email="jittor@qq.com",
    description="Jittor Aerial Image Detection",
    url="http://jittor.com",
    python_requires='>=3.7',
    packages=["jdet"],
    package_dir={'': os.path.join(path, 'python')},
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