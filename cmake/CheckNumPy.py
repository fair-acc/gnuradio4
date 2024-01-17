import sys
try:
    import numpy
    print(numpy.get_include())
except ImportError:
    sys.exit(1)
