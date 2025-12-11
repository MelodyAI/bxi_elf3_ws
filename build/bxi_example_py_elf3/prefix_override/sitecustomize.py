import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/szz/bxi/bxi_elf3_ws/install/bxi_example_py_elf3'
