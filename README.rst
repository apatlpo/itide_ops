=============================
Internal-Tide-Obs
=============================

Gathers various notebooks and tools to assess internal tide observability

Features
--------


Note on install [fastparquet](https://pypi.org/project/fastparquet/):

```
conda install -c conda-forge fastparquet
```

Note about installing [h3-py](https://github.com/uber/h3-py) on datarmor:

```
# unload all intel modules
module unload mpt intel-fc-16 intel-cmkl-16 cmake
# reset CC environment variable
setenv CC cc
# install cmake
conda install -c conda-forge cmake
# install h3
pip install h3
```


Binder
--------

Try these notebooks on pangeo.binder.io_ : |Binder|

See http://pangeo.io for more information.

Features
--------

* TODO

.. _pangeo.binder.io: http://binder.pangeo.io/

.. |Binder| image:: http://binder.pangeo.io/badge.svg
    :target: http://binder.pangeo.io/v2/gh/apatlpo/itide_ops/master

