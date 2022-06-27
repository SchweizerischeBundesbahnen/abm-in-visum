import sys

sys.path.extend(['D:\\Program Files\\JetBrains\\PyCharm 2019.2.1\\helpers\\pydev',
                 'D:\\Program Files\\JetBrains\\PyCharm 2019.2.1\\helpers\\pycharm_display',
                 'D:\\Program Files\\JetBrains\\PyCharm 2019.2.1\\helpers\\third_party\\thriftpy'])
import pydevd_pycharm

pydevd_pycharm.settrace('localhost', port=21000, suspend=False)
