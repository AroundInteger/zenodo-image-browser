import skimage
import sys

print(f'scikit-image version: {skimage.__version__}')

try:
    from skimage.feature import peak_local_maxima
    print('peak_local_maxima found in skimage.feature')
except ImportError as e:
    print('peak_local_maxima NOT found in skimage.feature')
    print(e)

try:
    from skimage.morphology import peak_local_maxima
    print('peak_local_maxima found in skimage.morphology')
except ImportError as e:
    print('peak_local_maxima NOT found in skimage.morphology')
    print(e)

try:
    from skimage.feature import peak
    print('peak module found in skimage.feature')
    print('peak module contents:', dir(peak))
except ImportError as e:
    print('peak module NOT found in skimage.feature')
    print(e) 