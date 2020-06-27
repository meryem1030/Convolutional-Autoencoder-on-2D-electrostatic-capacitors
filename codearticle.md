```python
import keras
from keras import backend as K
import matplotlib as mpl
from matplotlib import pyplot as plt
%matplotlib inline
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from os import listdir
from os.path import isfile, join
import numpy as np
import numpy
import cv2
from skimage.io import imread
import skimage
from keras.preprocessing import image
from PIL import Image
import os, sys
```

    Using TensorFlow backend.
    


```python
mpl.rc('image',cmap='gray')
# Load an color image in grayscale
simplified = cv2.imread('simplify.png',cv2.IMREAD_GRAYSCALE)
print(simplified.shape)
plt.imshow(simplified)
plt.show()
#Convert an imagefrom 8-bit to 16-bit signed integer format.
simplified=skimage.img_as_float(simplified, force_copy=False)
simplified = np.array(simplified, dtype=np.float64)
simplified = simplified.reshape(-1, 256,256, 1)
#the scale will be in the range(0,1)
simplified=simplified/np.max(simplified)
plt.imshow(simplified[0,...,0])
plt.show()
print(np.max(simplified))
print(np.min(simplified))
print(simplified.shape)
print(simplified.dtype)
plt.imshow(simplified[0,...,0])
```

    (256, 256)
    


![png](output_1_1.png)



![png](output_1_2.png)


    1.0
    0.0
    (1, 256, 256, 1)
    float64
    




    <matplotlib.image.AxesImage at 0x1a6536170c8>




![png](output_1_5.png)



```python
mypath=r'C:\Users\merye\images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
print(len(onlyfiles))
images = numpy.empty(len(onlyfiles),dtype=object)
from skimage import io
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath,onlyfiles[n]),cv2.IMREAD_GRAYSCALE)
    
    #plt.show()
    images[n]=cv2.resize(images[n], (256,256))
    plt.imshow(images[n])
    plt.show()
    
    
    images[n] = images[n].reshape((-1, 256,256, 1))
    images[n] = skimage.img_as_float(images[n], force_copy=False)  
    images[n] = np.array(images[n], dtype=np.float64)
    images[n] = images[n]/np.max(images[n])
    
    
    print(images[0].shape)
```

    550
    


![png](output_2_1.png)


    (1, 256, 256, 1)
    


![png](output_2_3.png)


    (1, 256, 256, 1)
    


![png](output_2_5.png)


    (1, 256, 256, 1)
    


![png](output_2_7.png)


    (1, 256, 256, 1)
    


![png](output_2_9.png)


    (1, 256, 256, 1)
    


![png](output_2_11.png)


    (1, 256, 256, 1)
    


![png](output_2_13.png)


    (1, 256, 256, 1)
    


![png](output_2_15.png)


    (1, 256, 256, 1)
    


![png](output_2_17.png)


    (1, 256, 256, 1)
    


![png](output_2_19.png)


    (1, 256, 256, 1)
    


![png](output_2_21.png)


    (1, 256, 256, 1)
    


![png](output_2_23.png)


    (1, 256, 256, 1)
    


![png](output_2_25.png)


    (1, 256, 256, 1)
    


![png](output_2_27.png)


    (1, 256, 256, 1)
    


![png](output_2_29.png)


    (1, 256, 256, 1)
    


![png](output_2_31.png)


    (1, 256, 256, 1)
    


![png](output_2_33.png)


    (1, 256, 256, 1)
    


![png](output_2_35.png)


    (1, 256, 256, 1)
    


![png](output_2_37.png)


    (1, 256, 256, 1)
    


![png](output_2_39.png)


    (1, 256, 256, 1)
    


![png](output_2_41.png)


    (1, 256, 256, 1)
    


![png](output_2_43.png)


    (1, 256, 256, 1)
    


![png](output_2_45.png)


    (1, 256, 256, 1)
    


![png](output_2_47.png)


    (1, 256, 256, 1)
    


![png](output_2_49.png)


    (1, 256, 256, 1)
    


![png](output_2_51.png)


    (1, 256, 256, 1)
    


![png](output_2_53.png)


    (1, 256, 256, 1)
    


![png](output_2_55.png)


    (1, 256, 256, 1)
    


![png](output_2_57.png)


    (1, 256, 256, 1)
    


![png](output_2_59.png)


    (1, 256, 256, 1)
    


![png](output_2_61.png)


    (1, 256, 256, 1)
    


![png](output_2_63.png)


    (1, 256, 256, 1)
    


![png](output_2_65.png)


    (1, 256, 256, 1)
    


![png](output_2_67.png)


    (1, 256, 256, 1)
    


![png](output_2_69.png)


    (1, 256, 256, 1)
    


![png](output_2_71.png)


    (1, 256, 256, 1)
    


![png](output_2_73.png)


    (1, 256, 256, 1)
    


![png](output_2_75.png)


    (1, 256, 256, 1)
    


![png](output_2_77.png)


    (1, 256, 256, 1)
    


![png](output_2_79.png)


    (1, 256, 256, 1)
    


![png](output_2_81.png)


    (1, 256, 256, 1)
    


![png](output_2_83.png)


    (1, 256, 256, 1)
    


![png](output_2_85.png)


    (1, 256, 256, 1)
    


![png](output_2_87.png)


    (1, 256, 256, 1)
    


![png](output_2_89.png)


    (1, 256, 256, 1)
    


![png](output_2_91.png)


    (1, 256, 256, 1)
    


![png](output_2_93.png)


    (1, 256, 256, 1)
    


![png](output_2_95.png)


    (1, 256, 256, 1)
    


![png](output_2_97.png)


    (1, 256, 256, 1)
    


![png](output_2_99.png)


    (1, 256, 256, 1)
    


![png](output_2_101.png)


    (1, 256, 256, 1)
    


![png](output_2_103.png)


    (1, 256, 256, 1)
    


![png](output_2_105.png)


    (1, 256, 256, 1)
    


![png](output_2_107.png)


    (1, 256, 256, 1)
    


![png](output_2_109.png)


    (1, 256, 256, 1)
    


![png](output_2_111.png)


    (1, 256, 256, 1)
    


![png](output_2_113.png)


    (1, 256, 256, 1)
    


![png](output_2_115.png)


    (1, 256, 256, 1)
    


![png](output_2_117.png)


    (1, 256, 256, 1)
    


![png](output_2_119.png)


    (1, 256, 256, 1)
    


![png](output_2_121.png)


    (1, 256, 256, 1)
    


![png](output_2_123.png)


    (1, 256, 256, 1)
    


![png](output_2_125.png)


    (1, 256, 256, 1)
    


![png](output_2_127.png)


    (1, 256, 256, 1)
    


![png](output_2_129.png)


    (1, 256, 256, 1)
    


![png](output_2_131.png)


    (1, 256, 256, 1)
    


![png](output_2_133.png)


    (1, 256, 256, 1)
    


![png](output_2_135.png)


    (1, 256, 256, 1)
    


![png](output_2_137.png)


    (1, 256, 256, 1)
    


![png](output_2_139.png)


    (1, 256, 256, 1)
    


![png](output_2_141.png)


    (1, 256, 256, 1)
    


![png](output_2_143.png)


    (1, 256, 256, 1)
    


![png](output_2_145.png)


    (1, 256, 256, 1)
    


![png](output_2_147.png)


    (1, 256, 256, 1)
    


![png](output_2_149.png)


    (1, 256, 256, 1)
    


![png](output_2_151.png)


    (1, 256, 256, 1)
    


![png](output_2_153.png)


    (1, 256, 256, 1)
    


![png](output_2_155.png)


    (1, 256, 256, 1)
    


![png](output_2_157.png)


    (1, 256, 256, 1)
    


![png](output_2_159.png)


    (1, 256, 256, 1)
    


![png](output_2_161.png)


    (1, 256, 256, 1)
    


![png](output_2_163.png)


    (1, 256, 256, 1)
    


![png](output_2_165.png)


    (1, 256, 256, 1)
    


![png](output_2_167.png)


    (1, 256, 256, 1)
    


![png](output_2_169.png)


    (1, 256, 256, 1)
    


![png](output_2_171.png)


    (1, 256, 256, 1)
    


![png](output_2_173.png)


    (1, 256, 256, 1)
    


![png](output_2_175.png)


    (1, 256, 256, 1)
    


![png](output_2_177.png)


    (1, 256, 256, 1)
    


![png](output_2_179.png)


    (1, 256, 256, 1)
    


![png](output_2_181.png)


    (1, 256, 256, 1)
    


![png](output_2_183.png)


    (1, 256, 256, 1)
    


![png](output_2_185.png)


    (1, 256, 256, 1)
    


![png](output_2_187.png)


    (1, 256, 256, 1)
    


![png](output_2_189.png)


    (1, 256, 256, 1)
    


![png](output_2_191.png)


    (1, 256, 256, 1)
    


![png](output_2_193.png)


    (1, 256, 256, 1)
    


![png](output_2_195.png)


    (1, 256, 256, 1)
    


![png](output_2_197.png)


    (1, 256, 256, 1)
    


![png](output_2_199.png)


    (1, 256, 256, 1)
    


![png](output_2_201.png)


    (1, 256, 256, 1)
    


![png](output_2_203.png)


    (1, 256, 256, 1)
    


![png](output_2_205.png)


    (1, 256, 256, 1)
    


![png](output_2_207.png)


    (1, 256, 256, 1)
    


![png](output_2_209.png)


    (1, 256, 256, 1)
    


![png](output_2_211.png)


    (1, 256, 256, 1)
    


![png](output_2_213.png)


    (1, 256, 256, 1)
    


![png](output_2_215.png)


    (1, 256, 256, 1)
    


![png](output_2_217.png)


    (1, 256, 256, 1)
    


![png](output_2_219.png)


    (1, 256, 256, 1)
    


![png](output_2_221.png)


    (1, 256, 256, 1)
    


![png](output_2_223.png)


    (1, 256, 256, 1)
    


![png](output_2_225.png)


    (1, 256, 256, 1)
    


![png](output_2_227.png)


    (1, 256, 256, 1)
    


![png](output_2_229.png)


    (1, 256, 256, 1)
    


![png](output_2_231.png)


    (1, 256, 256, 1)
    


![png](output_2_233.png)


    (1, 256, 256, 1)
    


![png](output_2_235.png)


    (1, 256, 256, 1)
    


![png](output_2_237.png)


    (1, 256, 256, 1)
    


![png](output_2_239.png)


    (1, 256, 256, 1)
    


![png](output_2_241.png)


    (1, 256, 256, 1)
    


![png](output_2_243.png)


    (1, 256, 256, 1)
    


![png](output_2_245.png)


    (1, 256, 256, 1)
    


![png](output_2_247.png)


    (1, 256, 256, 1)
    


![png](output_2_249.png)


    (1, 256, 256, 1)
    


![png](output_2_251.png)


    (1, 256, 256, 1)
    


![png](output_2_253.png)


    (1, 256, 256, 1)
    


![png](output_2_255.png)


    (1, 256, 256, 1)
    


![png](output_2_257.png)


    (1, 256, 256, 1)
    


![png](output_2_259.png)


    (1, 256, 256, 1)
    


![png](output_2_261.png)


    (1, 256, 256, 1)
    


![png](output_2_263.png)


    (1, 256, 256, 1)
    


![png](output_2_265.png)


    (1, 256, 256, 1)
    


![png](output_2_267.png)


    (1, 256, 256, 1)
    


![png](output_2_269.png)


    (1, 256, 256, 1)
    


![png](output_2_271.png)


    (1, 256, 256, 1)
    


![png](output_2_273.png)


    (1, 256, 256, 1)
    


![png](output_2_275.png)


    (1, 256, 256, 1)
    


![png](output_2_277.png)


    (1, 256, 256, 1)
    


![png](output_2_279.png)


    (1, 256, 256, 1)
    


![png](output_2_281.png)


    (1, 256, 256, 1)
    


![png](output_2_283.png)


    (1, 256, 256, 1)
    


![png](output_2_285.png)


    (1, 256, 256, 1)
    


![png](output_2_287.png)


    (1, 256, 256, 1)
    


![png](output_2_289.png)


    (1, 256, 256, 1)
    


![png](output_2_291.png)


    (1, 256, 256, 1)
    


![png](output_2_293.png)


    (1, 256, 256, 1)
    


![png](output_2_295.png)


    (1, 256, 256, 1)
    


![png](output_2_297.png)


    (1, 256, 256, 1)
    


![png](output_2_299.png)


    (1, 256, 256, 1)
    


![png](output_2_301.png)


    (1, 256, 256, 1)
    


![png](output_2_303.png)


    (1, 256, 256, 1)
    


![png](output_2_305.png)


    (1, 256, 256, 1)
    


![png](output_2_307.png)


    (1, 256, 256, 1)
    


![png](output_2_309.png)


    (1, 256, 256, 1)
    


![png](output_2_311.png)


    (1, 256, 256, 1)
    


![png](output_2_313.png)


    (1, 256, 256, 1)
    


![png](output_2_315.png)


    (1, 256, 256, 1)
    


![png](output_2_317.png)


    (1, 256, 256, 1)
    


![png](output_2_319.png)


    (1, 256, 256, 1)
    


![png](output_2_321.png)


    (1, 256, 256, 1)
    


![png](output_2_323.png)


    (1, 256, 256, 1)
    


![png](output_2_325.png)


    (1, 256, 256, 1)
    


![png](output_2_327.png)


    (1, 256, 256, 1)
    


![png](output_2_329.png)


    (1, 256, 256, 1)
    


![png](output_2_331.png)


    (1, 256, 256, 1)
    


![png](output_2_333.png)


    (1, 256, 256, 1)
    


![png](output_2_335.png)


    (1, 256, 256, 1)
    


![png](output_2_337.png)


    (1, 256, 256, 1)
    


![png](output_2_339.png)


    (1, 256, 256, 1)
    


![png](output_2_341.png)


    (1, 256, 256, 1)
    


![png](output_2_343.png)


    (1, 256, 256, 1)
    


![png](output_2_345.png)


    (1, 256, 256, 1)
    


![png](output_2_347.png)


    (1, 256, 256, 1)
    


![png](output_2_349.png)


    (1, 256, 256, 1)
    


![png](output_2_351.png)


    (1, 256, 256, 1)
    


![png](output_2_353.png)


    (1, 256, 256, 1)
    


![png](output_2_355.png)


    (1, 256, 256, 1)
    


![png](output_2_357.png)


    (1, 256, 256, 1)
    


![png](output_2_359.png)


    (1, 256, 256, 1)
    


![png](output_2_361.png)


    (1, 256, 256, 1)
    


![png](output_2_363.png)


    (1, 256, 256, 1)
    


![png](output_2_365.png)


    (1, 256, 256, 1)
    


![png](output_2_367.png)


    (1, 256, 256, 1)
    


![png](output_2_369.png)


    (1, 256, 256, 1)
    


![png](output_2_371.png)


    (1, 256, 256, 1)
    


![png](output_2_373.png)


    (1, 256, 256, 1)
    


![png](output_2_375.png)


    (1, 256, 256, 1)
    


![png](output_2_377.png)


    (1, 256, 256, 1)
    


![png](output_2_379.png)


    (1, 256, 256, 1)
    


![png](output_2_381.png)


    (1, 256, 256, 1)
    


![png](output_2_383.png)


    (1, 256, 256, 1)
    


![png](output_2_385.png)


    (1, 256, 256, 1)
    


![png](output_2_387.png)


    (1, 256, 256, 1)
    


![png](output_2_389.png)


    (1, 256, 256, 1)
    


![png](output_2_391.png)


    (1, 256, 256, 1)
    


![png](output_2_393.png)


    (1, 256, 256, 1)
    


![png](output_2_395.png)


    (1, 256, 256, 1)
    


![png](output_2_397.png)


    (1, 256, 256, 1)
    


![png](output_2_399.png)


    (1, 256, 256, 1)
    


![png](output_2_401.png)


    (1, 256, 256, 1)
    


![png](output_2_403.png)


    (1, 256, 256, 1)
    


![png](output_2_405.png)


    (1, 256, 256, 1)
    


![png](output_2_407.png)


    (1, 256, 256, 1)
    


![png](output_2_409.png)


    (1, 256, 256, 1)
    


![png](output_2_411.png)


    (1, 256, 256, 1)
    


![png](output_2_413.png)


    (1, 256, 256, 1)
    


![png](output_2_415.png)


    (1, 256, 256, 1)
    


![png](output_2_417.png)


    (1, 256, 256, 1)
    


![png](output_2_419.png)


    (1, 256, 256, 1)
    


![png](output_2_421.png)


    (1, 256, 256, 1)
    


![png](output_2_423.png)


    (1, 256, 256, 1)
    


![png](output_2_425.png)


    (1, 256, 256, 1)
    


![png](output_2_427.png)


    (1, 256, 256, 1)
    


![png](output_2_429.png)


    (1, 256, 256, 1)
    


![png](output_2_431.png)


    (1, 256, 256, 1)
    


![png](output_2_433.png)


    (1, 256, 256, 1)
    


![png](output_2_435.png)


    (1, 256, 256, 1)
    


![png](output_2_437.png)


    (1, 256, 256, 1)
    


![png](output_2_439.png)


    (1, 256, 256, 1)
    


![png](output_2_441.png)


    (1, 256, 256, 1)
    


![png](output_2_443.png)


    (1, 256, 256, 1)
    


![png](output_2_445.png)


    (1, 256, 256, 1)
    


![png](output_2_447.png)


    (1, 256, 256, 1)
    


![png](output_2_449.png)


    (1, 256, 256, 1)
    


![png](output_2_451.png)


    (1, 256, 256, 1)
    


![png](output_2_453.png)


    (1, 256, 256, 1)
    


![png](output_2_455.png)


    (1, 256, 256, 1)
    


![png](output_2_457.png)


    (1, 256, 256, 1)
    


![png](output_2_459.png)


    (1, 256, 256, 1)
    


![png](output_2_461.png)


    (1, 256, 256, 1)
    


![png](output_2_463.png)


    (1, 256, 256, 1)
    


![png](output_2_465.png)


    (1, 256, 256, 1)
    


![png](output_2_467.png)


    (1, 256, 256, 1)
    


![png](output_2_469.png)


    (1, 256, 256, 1)
    


![png](output_2_471.png)


    (1, 256, 256, 1)
    


![png](output_2_473.png)


    (1, 256, 256, 1)
    


![png](output_2_475.png)


    (1, 256, 256, 1)
    


![png](output_2_477.png)


    (1, 256, 256, 1)
    


![png](output_2_479.png)


    (1, 256, 256, 1)
    


![png](output_2_481.png)


    (1, 256, 256, 1)
    


![png](output_2_483.png)


    (1, 256, 256, 1)
    


![png](output_2_485.png)


    (1, 256, 256, 1)
    


![png](output_2_487.png)


    (1, 256, 256, 1)
    


![png](output_2_489.png)


    (1, 256, 256, 1)
    


![png](output_2_491.png)


    (1, 256, 256, 1)
    


![png](output_2_493.png)


    (1, 256, 256, 1)
    


![png](output_2_495.png)


    (1, 256, 256, 1)
    


![png](output_2_497.png)


    (1, 256, 256, 1)
    


![png](output_2_499.png)


    (1, 256, 256, 1)
    


![png](output_2_501.png)


    (1, 256, 256, 1)
    


![png](output_2_503.png)


    (1, 256, 256, 1)
    


![png](output_2_505.png)


    (1, 256, 256, 1)
    


![png](output_2_507.png)


    (1, 256, 256, 1)
    


![png](output_2_509.png)


    (1, 256, 256, 1)
    


![png](output_2_511.png)


    (1, 256, 256, 1)
    


![png](output_2_513.png)


    (1, 256, 256, 1)
    


![png](output_2_515.png)


    (1, 256, 256, 1)
    


![png](output_2_517.png)


    (1, 256, 256, 1)
    


![png](output_2_519.png)


    (1, 256, 256, 1)
    


![png](output_2_521.png)


    (1, 256, 256, 1)
    


![png](output_2_523.png)


    (1, 256, 256, 1)
    


![png](output_2_525.png)


    (1, 256, 256, 1)
    


![png](output_2_527.png)


    (1, 256, 256, 1)
    


![png](output_2_529.png)


    (1, 256, 256, 1)
    


![png](output_2_531.png)


    (1, 256, 256, 1)
    


![png](output_2_533.png)


    (1, 256, 256, 1)
    


![png](output_2_535.png)


    (1, 256, 256, 1)
    


![png](output_2_537.png)


    (1, 256, 256, 1)
    


![png](output_2_539.png)


    (1, 256, 256, 1)
    


![png](output_2_541.png)


    (1, 256, 256, 1)
    


![png](output_2_543.png)


    (1, 256, 256, 1)
    


![png](output_2_545.png)


    (1, 256, 256, 1)
    


![png](output_2_547.png)


    (1, 256, 256, 1)
    


![png](output_2_549.png)


    (1, 256, 256, 1)
    


![png](output_2_551.png)


    (1, 256, 256, 1)
    


![png](output_2_553.png)


    (1, 256, 256, 1)
    


![png](output_2_555.png)


    (1, 256, 256, 1)
    


![png](output_2_557.png)


    (1, 256, 256, 1)
    


![png](output_2_559.png)


    (1, 256, 256, 1)
    


![png](output_2_561.png)


    (1, 256, 256, 1)
    


![png](output_2_563.png)


    (1, 256, 256, 1)
    


![png](output_2_565.png)


    (1, 256, 256, 1)
    


![png](output_2_567.png)


    (1, 256, 256, 1)
    


![png](output_2_569.png)


    (1, 256, 256, 1)
    


![png](output_2_571.png)


    (1, 256, 256, 1)
    


![png](output_2_573.png)


    (1, 256, 256, 1)
    


![png](output_2_575.png)


    (1, 256, 256, 1)
    


![png](output_2_577.png)


    (1, 256, 256, 1)
    


![png](output_2_579.png)


    (1, 256, 256, 1)
    


![png](output_2_581.png)


    (1, 256, 256, 1)
    


![png](output_2_583.png)


    (1, 256, 256, 1)
    


![png](output_2_585.png)


    (1, 256, 256, 1)
    


![png](output_2_587.png)


    (1, 256, 256, 1)
    


![png](output_2_589.png)


    (1, 256, 256, 1)
    


![png](output_2_591.png)


    (1, 256, 256, 1)
    


![png](output_2_593.png)


    (1, 256, 256, 1)
    


![png](output_2_595.png)


    (1, 256, 256, 1)
    


![png](output_2_597.png)


    (1, 256, 256, 1)
    


![png](output_2_599.png)


    (1, 256, 256, 1)
    


![png](output_2_601.png)


    (1, 256, 256, 1)
    


![png](output_2_603.png)


    (1, 256, 256, 1)
    


![png](output_2_605.png)


    (1, 256, 256, 1)
    


![png](output_2_607.png)


    (1, 256, 256, 1)
    


![png](output_2_609.png)


    (1, 256, 256, 1)
    


![png](output_2_611.png)


    (1, 256, 256, 1)
    


![png](output_2_613.png)


    (1, 256, 256, 1)
    


![png](output_2_615.png)


    (1, 256, 256, 1)
    


![png](output_2_617.png)


    (1, 256, 256, 1)
    


![png](output_2_619.png)


    (1, 256, 256, 1)
    


![png](output_2_621.png)


    (1, 256, 256, 1)
    


![png](output_2_623.png)


    (1, 256, 256, 1)
    


![png](output_2_625.png)


    (1, 256, 256, 1)
    


![png](output_2_627.png)


    (1, 256, 256, 1)
    


![png](output_2_629.png)


    (1, 256, 256, 1)
    


![png](output_2_631.png)


    (1, 256, 256, 1)
    


![png](output_2_633.png)


    (1, 256, 256, 1)
    


![png](output_2_635.png)


    (1, 256, 256, 1)
    


![png](output_2_637.png)


    (1, 256, 256, 1)
    


![png](output_2_639.png)


    (1, 256, 256, 1)
    


![png](output_2_641.png)


    (1, 256, 256, 1)
    


![png](output_2_643.png)


    (1, 256, 256, 1)
    


![png](output_2_645.png)


    (1, 256, 256, 1)
    


![png](output_2_647.png)


    (1, 256, 256, 1)
    


![png](output_2_649.png)


    (1, 256, 256, 1)
    


![png](output_2_651.png)


    (1, 256, 256, 1)
    


![png](output_2_653.png)


    (1, 256, 256, 1)
    


![png](output_2_655.png)


    (1, 256, 256, 1)
    


![png](output_2_657.png)


    (1, 256, 256, 1)
    


![png](output_2_659.png)


    (1, 256, 256, 1)
    


![png](output_2_661.png)


    (1, 256, 256, 1)
    


![png](output_2_663.png)


    (1, 256, 256, 1)
    


![png](output_2_665.png)


    (1, 256, 256, 1)
    


![png](output_2_667.png)


    (1, 256, 256, 1)
    


![png](output_2_669.png)


    (1, 256, 256, 1)
    


![png](output_2_671.png)


    (1, 256, 256, 1)
    


![png](output_2_673.png)


    (1, 256, 256, 1)
    


![png](output_2_675.png)


    (1, 256, 256, 1)
    


![png](output_2_677.png)


    (1, 256, 256, 1)
    


![png](output_2_679.png)


    (1, 256, 256, 1)
    


![png](output_2_681.png)


    (1, 256, 256, 1)
    


![png](output_2_683.png)


    (1, 256, 256, 1)
    


![png](output_2_685.png)


    (1, 256, 256, 1)
    


![png](output_2_687.png)


    (1, 256, 256, 1)
    


![png](output_2_689.png)


    (1, 256, 256, 1)
    


![png](output_2_691.png)


    (1, 256, 256, 1)
    


![png](output_2_693.png)


    (1, 256, 256, 1)
    


![png](output_2_695.png)


    (1, 256, 256, 1)
    


![png](output_2_697.png)


    (1, 256, 256, 1)
    


![png](output_2_699.png)


    (1, 256, 256, 1)
    


![png](output_2_701.png)


    (1, 256, 256, 1)
    


![png](output_2_703.png)


    (1, 256, 256, 1)
    


![png](output_2_705.png)


    (1, 256, 256, 1)
    


![png](output_2_707.png)


    (1, 256, 256, 1)
    


![png](output_2_709.png)


    (1, 256, 256, 1)
    


![png](output_2_711.png)


    (1, 256, 256, 1)
    


![png](output_2_713.png)


    (1, 256, 256, 1)
    


![png](output_2_715.png)


    (1, 256, 256, 1)
    


![png](output_2_717.png)


    (1, 256, 256, 1)
    


![png](output_2_719.png)


    (1, 256, 256, 1)
    


![png](output_2_721.png)


    (1, 256, 256, 1)
    


![png](output_2_723.png)


    (1, 256, 256, 1)
    


![png](output_2_725.png)


    (1, 256, 256, 1)
    


![png](output_2_727.png)


    (1, 256, 256, 1)
    


![png](output_2_729.png)


    (1, 256, 256, 1)
    


![png](output_2_731.png)


    (1, 256, 256, 1)
    


![png](output_2_733.png)


    (1, 256, 256, 1)
    


![png](output_2_735.png)


    (1, 256, 256, 1)
    


![png](output_2_737.png)


    (1, 256, 256, 1)
    


![png](output_2_739.png)


    (1, 256, 256, 1)
    


![png](output_2_741.png)


    (1, 256, 256, 1)
    


![png](output_2_743.png)


    (1, 256, 256, 1)
    


![png](output_2_745.png)


    (1, 256, 256, 1)
    


![png](output_2_747.png)


    (1, 256, 256, 1)
    


![png](output_2_749.png)


    (1, 256, 256, 1)
    


![png](output_2_751.png)


    (1, 256, 256, 1)
    


![png](output_2_753.png)


    (1, 256, 256, 1)
    


![png](output_2_755.png)


    (1, 256, 256, 1)
    


![png](output_2_757.png)


    (1, 256, 256, 1)
    


![png](output_2_759.png)


    (1, 256, 256, 1)
    


![png](output_2_761.png)


    (1, 256, 256, 1)
    


![png](output_2_763.png)


    (1, 256, 256, 1)
    


![png](output_2_765.png)


    (1, 256, 256, 1)
    


![png](output_2_767.png)


    (1, 256, 256, 1)
    


![png](output_2_769.png)


    (1, 256, 256, 1)
    


![png](output_2_771.png)


    (1, 256, 256, 1)
    


![png](output_2_773.png)


    (1, 256, 256, 1)
    


![png](output_2_775.png)


    (1, 256, 256, 1)
    


![png](output_2_777.png)


    (1, 256, 256, 1)
    


![png](output_2_779.png)


    (1, 256, 256, 1)
    


![png](output_2_781.png)


    (1, 256, 256, 1)
    


![png](output_2_783.png)


    (1, 256, 256, 1)
    


![png](output_2_785.png)


    (1, 256, 256, 1)
    


![png](output_2_787.png)


    (1, 256, 256, 1)
    


![png](output_2_789.png)


    (1, 256, 256, 1)
    


![png](output_2_791.png)


    (1, 256, 256, 1)
    


![png](output_2_793.png)


    (1, 256, 256, 1)
    


![png](output_2_795.png)


    (1, 256, 256, 1)
    


![png](output_2_797.png)


    (1, 256, 256, 1)
    


![png](output_2_799.png)


    (1, 256, 256, 1)
    


![png](output_2_801.png)


    (1, 256, 256, 1)
    


![png](output_2_803.png)


    (1, 256, 256, 1)
    


![png](output_2_805.png)


    (1, 256, 256, 1)
    


![png](output_2_807.png)


    (1, 256, 256, 1)
    


![png](output_2_809.png)


    (1, 256, 256, 1)
    


![png](output_2_811.png)


    (1, 256, 256, 1)
    


![png](output_2_813.png)


    (1, 256, 256, 1)
    


![png](output_2_815.png)


    (1, 256, 256, 1)
    


![png](output_2_817.png)


    (1, 256, 256, 1)
    


![png](output_2_819.png)


    (1, 256, 256, 1)
    


![png](output_2_821.png)


    (1, 256, 256, 1)
    


![png](output_2_823.png)


    (1, 256, 256, 1)
    


![png](output_2_825.png)


    (1, 256, 256, 1)
    


![png](output_2_827.png)


    (1, 256, 256, 1)
    


![png](output_2_829.png)


    (1, 256, 256, 1)
    


![png](output_2_831.png)


    (1, 256, 256, 1)
    


![png](output_2_833.png)


    (1, 256, 256, 1)
    


![png](output_2_835.png)


    (1, 256, 256, 1)
    


![png](output_2_837.png)


    (1, 256, 256, 1)
    


![png](output_2_839.png)


    (1, 256, 256, 1)
    


![png](output_2_841.png)


    (1, 256, 256, 1)
    


![png](output_2_843.png)


    (1, 256, 256, 1)
    


![png](output_2_845.png)


    (1, 256, 256, 1)
    


![png](output_2_847.png)


    (1, 256, 256, 1)
    


![png](output_2_849.png)


    (1, 256, 256, 1)
    


![png](output_2_851.png)


    (1, 256, 256, 1)
    


![png](output_2_853.png)


    (1, 256, 256, 1)
    


![png](output_2_855.png)


    (1, 256, 256, 1)
    


![png](output_2_857.png)


    (1, 256, 256, 1)
    


![png](output_2_859.png)


    (1, 256, 256, 1)
    


![png](output_2_861.png)


    (1, 256, 256, 1)
    


![png](output_2_863.png)


    (1, 256, 256, 1)
    


![png](output_2_865.png)


    (1, 256, 256, 1)
    


![png](output_2_867.png)


    (1, 256, 256, 1)
    


![png](output_2_869.png)


    (1, 256, 256, 1)
    


![png](output_2_871.png)


    (1, 256, 256, 1)
    


![png](output_2_873.png)


    (1, 256, 256, 1)
    


![png](output_2_875.png)


    (1, 256, 256, 1)
    


![png](output_2_877.png)


    (1, 256, 256, 1)
    


![png](output_2_879.png)


    (1, 256, 256, 1)
    


![png](output_2_881.png)


    (1, 256, 256, 1)
    


![png](output_2_883.png)


    (1, 256, 256, 1)
    


![png](output_2_885.png)


    (1, 256, 256, 1)
    


![png](output_2_887.png)


    (1, 256, 256, 1)
    


![png](output_2_889.png)


    (1, 256, 256, 1)
    


![png](output_2_891.png)


    (1, 256, 256, 1)
    


![png](output_2_893.png)


    (1, 256, 256, 1)
    


![png](output_2_895.png)


    (1, 256, 256, 1)
    


![png](output_2_897.png)


    (1, 256, 256, 1)
    


![png](output_2_899.png)


    (1, 256, 256, 1)
    


![png](output_2_901.png)


    (1, 256, 256, 1)
    


![png](output_2_903.png)


    (1, 256, 256, 1)
    


![png](output_2_905.png)


    (1, 256, 256, 1)
    


![png](output_2_907.png)


    (1, 256, 256, 1)
    


![png](output_2_909.png)


    (1, 256, 256, 1)
    


![png](output_2_911.png)


    (1, 256, 256, 1)
    


![png](output_2_913.png)


    (1, 256, 256, 1)
    


![png](output_2_915.png)


    (1, 256, 256, 1)
    


![png](output_2_917.png)


    (1, 256, 256, 1)
    


![png](output_2_919.png)


    (1, 256, 256, 1)
    


![png](output_2_921.png)


    (1, 256, 256, 1)
    


![png](output_2_923.png)


    (1, 256, 256, 1)
    


![png](output_2_925.png)


    (1, 256, 256, 1)
    


![png](output_2_927.png)


    (1, 256, 256, 1)
    


![png](output_2_929.png)


    (1, 256, 256, 1)
    


![png](output_2_931.png)


    (1, 256, 256, 1)
    


![png](output_2_933.png)


    (1, 256, 256, 1)
    


![png](output_2_935.png)


    (1, 256, 256, 1)
    


![png](output_2_937.png)


    (1, 256, 256, 1)
    


![png](output_2_939.png)


    (1, 256, 256, 1)
    


![png](output_2_941.png)


    (1, 256, 256, 1)
    


![png](output_2_943.png)


    (1, 256, 256, 1)
    


![png](output_2_945.png)


    (1, 256, 256, 1)
    


![png](output_2_947.png)


    (1, 256, 256, 1)
    


![png](output_2_949.png)


    (1, 256, 256, 1)
    


![png](output_2_951.png)


    (1, 256, 256, 1)
    


![png](output_2_953.png)


    (1, 256, 256, 1)
    


![png](output_2_955.png)


    (1, 256, 256, 1)
    


![png](output_2_957.png)


    (1, 256, 256, 1)
    


![png](output_2_959.png)


    (1, 256, 256, 1)
    


![png](output_2_961.png)


    (1, 256, 256, 1)
    


![png](output_2_963.png)


    (1, 256, 256, 1)
    


![png](output_2_965.png)


    (1, 256, 256, 1)
    


![png](output_2_967.png)


    (1, 256, 256, 1)
    


![png](output_2_969.png)


    (1, 256, 256, 1)
    


![png](output_2_971.png)


    (1, 256, 256, 1)
    


![png](output_2_973.png)


    (1, 256, 256, 1)
    


![png](output_2_975.png)


    (1, 256, 256, 1)
    


![png](output_2_977.png)


    (1, 256, 256, 1)
    


![png](output_2_979.png)


    (1, 256, 256, 1)
    


![png](output_2_981.png)


    (1, 256, 256, 1)
    


![png](output_2_983.png)


    (1, 256, 256, 1)
    


![png](output_2_985.png)


    (1, 256, 256, 1)
    


![png](output_2_987.png)


    (1, 256, 256, 1)
    


![png](output_2_989.png)


    (1, 256, 256, 1)
    


![png](output_2_991.png)


    (1, 256, 256, 1)
    


![png](output_2_993.png)


    (1, 256, 256, 1)
    


![png](output_2_995.png)


    (1, 256, 256, 1)
    


![png](output_2_997.png)


    (1, 256, 256, 1)
    


![png](output_2_999.png)


    (1, 256, 256, 1)
    


![png](output_2_1001.png)


    (1, 256, 256, 1)
    


![png](output_2_1003.png)


    (1, 256, 256, 1)
    


![png](output_2_1005.png)


    (1, 256, 256, 1)
    


![png](output_2_1007.png)


    (1, 256, 256, 1)
    


![png](output_2_1009.png)


    (1, 256, 256, 1)
    


![png](output_2_1011.png)


    (1, 256, 256, 1)
    


![png](output_2_1013.png)


    (1, 256, 256, 1)
    


![png](output_2_1015.png)


    (1, 256, 256, 1)
    


![png](output_2_1017.png)


    (1, 256, 256, 1)
    


![png](output_2_1019.png)


    (1, 256, 256, 1)
    


![png](output_2_1021.png)


    (1, 256, 256, 1)
    


![png](output_2_1023.png)


    (1, 256, 256, 1)
    


![png](output_2_1025.png)


    (1, 256, 256, 1)
    


![png](output_2_1027.png)


    (1, 256, 256, 1)
    


![png](output_2_1029.png)


    (1, 256, 256, 1)
    


![png](output_2_1031.png)


    (1, 256, 256, 1)
    


![png](output_2_1033.png)


    (1, 256, 256, 1)
    


![png](output_2_1035.png)


    (1, 256, 256, 1)
    


![png](output_2_1037.png)


    (1, 256, 256, 1)
    


![png](output_2_1039.png)


    (1, 256, 256, 1)
    


![png](output_2_1041.png)


    (1, 256, 256, 1)
    


![png](output_2_1043.png)


    (1, 256, 256, 1)
    


![png](output_2_1045.png)


    (1, 256, 256, 1)
    


![png](output_2_1047.png)


    (1, 256, 256, 1)
    


![png](output_2_1049.png)


    (1, 256, 256, 1)
    


![png](output_2_1051.png)


    (1, 256, 256, 1)
    


![png](output_2_1053.png)


    (1, 256, 256, 1)
    


![png](output_2_1055.png)


    (1, 256, 256, 1)
    


![png](output_2_1057.png)


    (1, 256, 256, 1)
    


![png](output_2_1059.png)


    (1, 256, 256, 1)
    


![png](output_2_1061.png)


    (1, 256, 256, 1)
    


![png](output_2_1063.png)


    (1, 256, 256, 1)
    


![png](output_2_1065.png)


    (1, 256, 256, 1)
    


![png](output_2_1067.png)


    (1, 256, 256, 1)
    


![png](output_2_1069.png)


    (1, 256, 256, 1)
    


![png](output_2_1071.png)


    (1, 256, 256, 1)
    


![png](output_2_1073.png)


    (1, 256, 256, 1)
    


![png](output_2_1075.png)


    (1, 256, 256, 1)
    


![png](output_2_1077.png)


    (1, 256, 256, 1)
    


![png](output_2_1079.png)


    (1, 256, 256, 1)
    


![png](output_2_1081.png)


    (1, 256, 256, 1)
    


![png](output_2_1083.png)


    (1, 256, 256, 1)
    


![png](output_2_1085.png)


    (1, 256, 256, 1)
    


![png](output_2_1087.png)


    (1, 256, 256, 1)
    


![png](output_2_1089.png)


    (1, 256, 256, 1)
    


![png](output_2_1091.png)


    (1, 256, 256, 1)
    


![png](output_2_1093.png)


    (1, 256, 256, 1)
    


![png](output_2_1095.png)


    (1, 256, 256, 1)
    


![png](output_2_1097.png)


    (1, 256, 256, 1)
    


![png](output_2_1099.png)


    (1, 256, 256, 1)
    


```python
#reshape the matrix (len,256,256,1)
images = np.vstack(images).astype(np.float64)
print(images.shape)
print(images.dtype)
print(len(images))
print(np.max(images[1]))
```

    (550, 256, 256, 1)
    float64
    550
    1.0
    


```python
dif=numpy.empty(len(onlyfiles), dtype=object)
for i in range(0,len(onlyfiles)):
    dif[i]=abs(images[i]-simplified)
    threshold=np.max(dif[i])/4
    dif[i][dif[i] < threshold] = 0
    dif[i][dif[i] > 0] = 1
plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i, ..., 0], cmap='gray')
    plt.gray()
plt.figure()
print(dif.shape)
plt.imshow(dif[1][0,...,0], cmap='gray')
```

    (550,)
    




    <matplotlib.image.AxesImage at 0x1a65374b688>




![png](output_4_2.png)



![png](output_4_3.png)



```python
dif = np.vstack(dif).astype(np.float64)
print(dif.dtype)
print(dif.shape)
len(dif)
```

    float64
    (550, 256, 256, 1)
    




    550




```python
plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(dif[i,...,0], cmap='gray')
    plt.gray()
```


![png](output_6_0.png)



```python
import numpy
import numpy as np
A=6
B=6
W=0.2 # Width of contamination to be read in
H=0.2 # Height of contamination, to be read in
Pc=8.4
Wind = max(round(W/A*255), 1)
Hind = max(round(H/B*255), 1)
dPind = (Pc-1)/10
X_file = open('axisn.txt','r')
X = []
for line in X_file:
    X.extend([float(i) for i in line.split()])
#X=[2.9,2.9,2.801]
#Y=[3.3,5.2,2.7]
Y_file = open('ordinaten.txt','r')
Y = []
for line in Y_file:
    Y.extend([float(i) for i in line.split()])
for i in range(0,len(X)):
    X[i]=max(round(X[i]/A*255),1)
    Y[i]=max(round(Y[i]/B*255),1)

M = numpy.zeros((len(onlyfiles),256,256,1),dtype=float)
print(M.shape)
for s in range(0,len(M)):
    Wnew=X[s]+Wind-1
    Hnew=Y[s]+Hind-1
    if Wnew > 255:
        Wnew = 255
    for c in range(X[s],Wnew):
        if Hnew > 255:
            Hnew = 255
        for r in range(Y[s],Hnew):
            M[s][r][c][0]=dPind
                
    if len(M[s][M[s] != 0]) == 0:
        print(s,":",X[s],Wnew,Y[s],Hnew)
print(M.shape)
print(M[8][M[8] != 0])
```

    (550, 256, 256, 1)
    (550, 256, 256, 1)
    [0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74
     0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74
     0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74 0.74
     0.74 0.74 0.74 0.74 0.74 0.74 0.74]
    


```python
print(M.shape)
plt.imshow(M[8,...,0])
print(M.shape)
```

    (550, 256, 256, 1)
    (550, 256, 256, 1)
    


![png](output_8_1.png)



```python
pip install sklearn
```

    Requirement already satisfied: sklearn in c:\users\merye\anaconda3\envs\tf\lib\site-packages (0.0)
    Requirement already satisfied: scikit-learn in c:\users\merye\anaconda3\envs\tf\lib\site-packages (from sklearn) (0.23.1)
    Requirement already satisfied: joblib>=0.11 in c:\users\merye\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (0.15.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\merye\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in c:\users\merye\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (1.4.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\users\merye\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (1.18.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
#split data as %80 training and %20 testing
from sklearn.model_selection import train_test_split
train_E,valid_E,train_T,valid_T= train_test_split(M,dif,
                                                  test_size=0.2,random_state=13)
```


```python
plt.figure(figsize=(20,4))
print("Input")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_E[i, ..., 0], cmap='gray')
    plt.gray()
plt.figure(figsize=(20,4))
print("Output")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_T[i, ..., 0], cmap='gray')
    plt.gray()
```

    Input
    Output
    


![png](output_11_1.png)



![png](output_11_2.png)



```python
print(len(train_E))
print(train_E.shape)
print(train_T.shape)
print(len(valid_E))
print(valid_E.shape)
print(valid_T.shape)
```

    440
    (440, 256, 256, 1)
    (440, 256, 256, 1)
    110
    (110, 256, 256, 1)
    (110, 256, 256, 1)
    


```python
number_epochs = 60
batch_size = 16
learning_rate = 0.0001
num_workers=0
in_channel=1
#for Id in range(0,len(testfiles)):
m,n=256,256
input_images= Input(shape = (m,n,in_channel))
```


```python
input_images
```




    <tf.Tensor 'input_1:0' shape=(None, 256, 256, 1) dtype=float32>




```python
def autoencoder(input_images):

#encoder
    CA1 = Conv2D(256, (3, 3), activation='relu', padding='same')(input_images)
    MP1 = MaxPooling2D((2, 2), padding='same')(CA1)
    CA2 = Conv2D(128, (3, 3), activation='relu', padding='same')(MP1)
    MP2 = MaxPooling2D((2, 2), padding='same')(CA2)
    CA3 = Conv2D(64, (3, 3), activation='relu', padding='same')(MP2)
    MP3 = MaxPooling2D((2, 2), padding='same')(CA3)
    CA4 = Conv2D(32, (3, 3), activation='relu', padding='same')(MP3)
    #MP4 = MaxPooling2D((2, 2), padding='same')(CA4)
    #CA5 = Conv2D(16, (3, 3), activation='relu', padding='same')(MP4)
    #MP5 = MaxPooling2D((2, 2), padding='same')(CA5)
    #CA6 = Conv2D(4, (3, 3), activation='relu', padding='same')(MP5)
    #MP6 = MaxPooling2D((2, 2), padding='same')(CA6)
    #CA7 = Conv2D(4, (3, 3), activation='relu', padding='same')(MP6)
    
    
    encoded = MaxPooling2D((2, 2), padding='same')(CA4)
    
#decoder    
    
    CA6 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    UP1 = UpSampling2D((2, 2))(CA6)
    CA7 = Conv2D(64, (3, 3), activation='relu', padding='same')(UP1)
    UP2 = UpSampling2D((2, 2))(CA7)
    CA8 = Conv2D(128, (3, 3), activation='relu', padding='same')(UP2)
    UP3 = UpSampling2D((2, 2))(CA8)
    CA9 = Conv2D(256, (3, 3), activation='relu', padding='same')(UP3)
    UP4 = UpSampling2D((2, 2))(CA9)
    #CA10 = Conv2D(256, (3, 3), activation='relu', padding='same')(UP4)
    #UP5 = UpSampling2D((2, 2))(CA10)
    #CA11 = Conv2D(128, (3, 3), activation='relu', padding='same')(UP5)
    #UP6 = UpSampling2D((2, 2))(CA11)
    #CA12 = Conv2D(256, (3, 3), activation='relu', padding='same')(UP6)
    #UP7 = UpSampling2D((2, 2))(CA12)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(UP4)
    return decoded
#optimization network
autoencoder = Model(input_images, autoencoder(input_images))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
    
```


```python
autoencoder.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 256, 256, 1)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 256, 256, 256)     2560      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 128, 128, 256)     0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 128, 128, 128)     295040    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 64, 64, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 64, 64, 64)        73792     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 32, 32)        18464     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 32)        9248      
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 32, 32, 64)        18496     
    _________________________________________________________________
    up_sampling2d_2 (UpSampling2 (None, 64, 64, 64)        0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 64, 64, 128)       73856     
    _________________________________________________________________
    up_sampling2d_3 (UpSampling2 (None, 128, 128, 128)     0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 128, 128, 256)     295168    
    _________________________________________________________________
    up_sampling2d_4 (UpSampling2 (None, 256, 256, 256)     0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 256, 256, 1)       2305      
    =================================================================
    Total params: 788,929
    Trainable params: 788,929
    Non-trainable params: 0
    _________________________________________________________________
    


```python
from keras.preprocessing.image import ImageDataGenerator
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.2, height_shift_range=0.2,
    vertical_flip=True, fill_mode="nearest")
repeats = 3 # repeats*len(train_E) training set size
aug_E = np.empty((repeats*len(train_E),256,256,1))
aug_T = np.empty((repeats*len(train_T),256,256,1))
autoencoder_train = autoencoder.fit(aug_E, aug_T,batch_size=batch_size,epochs=number_epochs,verbose=1,validation_data=(valid_E, valid_T))
```

    Train on 1320 samples, validate on 110 samples
    Epoch 1/60
    1320/1320 [==============================] - 1713s 1s/step - loss: 0.2400 - val_loss: 0.2334
    Epoch 2/60
    1320/1320 [==============================] - 1587s 1s/step - loss: 0.2204 - val_loss: 0.2181
    Epoch 3/60
    1320/1320 [==============================] - 1600s 1s/step - loss: 0.2022 - val_loss: 0.2039
    Epoch 4/60
    1320/1320 [==============================] - 1591s 1s/step - loss: 0.1853 - val_loss: 0.1910
    Epoch 5/60
    1320/1320 [==============================] - 1595s 1s/step - loss: 0.1698 - val_loss: 0.1793
    Epoch 6/60
    1320/1320 [==============================] - 1643s 1s/step - loss: 0.1556 - val_loss: 0.1687
    Epoch 7/60
    1320/1320 [==============================] - 1596s 1s/step - loss: 0.1425 - val_loss: 0.1592
    Epoch 8/60
    1320/1320 [==============================] - 1640s 1s/step - loss: 0.1307 - val_loss: 0.1506
    Epoch 9/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.1198 - val_loss: 0.1429
    Epoch 10/60
    1320/1320 [==============================] - 1589s 1s/step - loss: 0.1100 - val_loss: 0.1360
    Epoch 11/60
    1320/1320 [==============================] - 1590s 1s/step - loss: 0.1011 - val_loss: 0.1298
    Epoch 12/60
    1320/1320 [==============================] - 1622s 1s/step - loss: 0.0930 - val_loss: 0.1243
    Epoch 13/60
    1320/1320 [==============================] - 1778s 1s/step - loss: 0.0856 - val_loss: 0.1194
    Epoch 14/60
    1320/1320 [==============================] - 1716s 1s/step - loss: 0.0789 - val_loss: 0.1150
    Epoch 15/60
    1320/1320 [==============================] - 1684s 1s/step - loss: 0.0728 - val_loss: 0.1111
    Epoch 16/60
    1320/1320 [==============================] - 1732s 1s/step - loss: 0.0673 - val_loss: 0.1076
    Epoch 17/60
    1320/1320 [==============================] - 1711s 1s/step - loss: 0.0623 - val_loss: 0.1045
    Epoch 18/60
    1320/1320 [==============================] - 1679s 1s/step - loss: 0.0577 - val_loss: 0.1017
    Epoch 19/60
    1320/1320 [==============================] - 1710s 1s/step - loss: 0.0535 - val_loss: 0.0993
    Epoch 20/60
    1320/1320 [==============================] - 1612s 1s/step - loss: 0.0497 - val_loss: 0.0971
    Epoch 21/60
    1320/1320 [==============================] - 1602s 1s/step - loss: 0.0463 - val_loss: 0.0951
    Epoch 22/60
    1320/1320 [==============================] - 1613s 1s/step - loss: 0.0431 - val_loss: 0.0933
    Epoch 23/60
    1320/1320 [==============================] - 1660s 1s/step - loss: 0.0402 - val_loss: 0.0917
    Epoch 24/60
    1320/1320 [==============================] - 1585s 1s/step - loss: 0.0375 - val_loss: 0.0903
    Epoch 25/60
    1320/1320 [==============================] - 1583s 1s/step - loss: 0.0350 - val_loss: 0.0891
    Epoch 26/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.0328 - val_loss: 0.0880
    Epoch 27/60
    1320/1320 [==============================] - 1597s 1s/step - loss: 0.0307 - val_loss: 0.0870
    Epoch 28/60
    1320/1320 [==============================] - 1584s 1s/step - loss: 0.0287 - val_loss: 0.0861
    Epoch 29/60
    1320/1320 [==============================] - 1580s 1s/step - loss: 0.0270 - val_loss: 0.0853
    Epoch 30/60
    1320/1320 [==============================] - 1582s 1s/step - loss: 0.0253 - val_loss: 0.0846
    Epoch 31/60
    1320/1320 [==============================] - 1582s 1s/step - loss: 0.0238 - val_loss: 0.0839
    Epoch 32/60
    1320/1320 [==============================] - 1584s 1s/step - loss: 0.0224 - val_loss: 0.0833
    Epoch 33/60
    1320/1320 [==============================] - 1585s 1s/step - loss: 0.0211 - val_loss: 0.0828
    Epoch 34/60
    1320/1320 [==============================] - 1582s 1s/step - loss: 0.0199 - val_loss: 0.0824
    Epoch 35/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.0187 - val_loss: 0.0820
    Epoch 36/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.0177 - val_loss: 0.0817
    Epoch 37/60
    1320/1320 [==============================] - 1585s 1s/step - loss: 0.0167 - val_loss: 0.0814
    Epoch 38/60
    1320/1320 [==============================] - 1581s 1s/step - loss: 0.0158 - val_loss: 0.0811
    Epoch 39/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.0149 - val_loss: 0.0808
    Epoch 40/60
    1320/1320 [==============================] - 1583s 1s/step - loss: 0.0141 - val_loss: 0.0806
    Epoch 41/60
    1320/1320 [==============================] - 1588s 1s/step - loss: 0.0133 - val_loss: 0.0805
    Epoch 42/60
    1320/1320 [==============================] - 1587s 1s/step - loss: 0.0126 - val_loss: 0.0803
    Epoch 43/60
    1320/1320 [==============================] - 1588s 1s/step - loss: 0.0120 - val_loss: 0.0802
    Epoch 44/60
    1320/1320 [==============================] - 1585s 1s/step - loss: 0.0114 - val_loss: 0.0801
    Epoch 45/60
    1320/1320 [==============================] - 1580s 1s/step - loss: 0.0108 - val_loss: 0.0800
    Epoch 46/60
    1320/1320 [==============================] - 1584s 1s/step - loss: 0.0102 - val_loss: 0.0799
    Epoch 47/60
    1320/1320 [==============================] - 1586s 1s/step - loss: 0.0097 - val_loss: 0.0799
    Epoch 48/60
    1320/1320 [==============================] - 1608s 1s/step - loss: 0.0092 - val_loss: 0.0798
    Epoch 49/60
    1320/1320 [==============================] - 1641s 1s/step - loss: 0.0088 - val_loss: 0.0798
    Epoch 50/60
    1320/1320 [==============================] - 1582s 1s/step - loss: 0.0083 - val_loss: 0.0798
    Epoch 51/60
    1320/1320 [==============================] - 1585s 1s/step - loss: 0.0079 - val_loss: 0.0798
    Epoch 52/60
    1320/1320 [==============================] - 1584s 1s/step - loss: 0.0075 - val_loss: 0.0798
    Epoch 53/60
    1320/1320 [==============================] - 1582s 1s/step - loss: 0.0072 - val_loss: 0.0798
    Epoch 54/60
    1320/1320 [==============================] - 1662s 1s/step - loss: 0.0068 - val_loss: 0.0798
    Epoch 55/60
    1320/1320 [==============================] - 1596s 1s/step - loss: 0.0065 - val_loss: 0.0798
    Epoch 56/60
    1320/1320 [==============================] - 1626s 1s/step - loss: 0.0062 - val_loss: 0.0798
    Epoch 57/60
    1320/1320 [==============================] - 1656s 1s/step - loss: 0.0059 - val_loss: 0.0799
    Epoch 58/60
    1320/1320 [==============================] - 1666s 1s/step - loss: 0.0056 - val_loss: 0.0799
    Epoch 59/60
    1320/1320 [==============================] - 1739s 1s/step - loss: 0.0054 - val_loss: 0.0800
    Epoch 60/60
    1320/1320 [==============================] - 1610s 1s/step - loss: 0.0051 - val_loss: 0.0800
    


```python
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(number_epochs)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()
```


![png](output_18_0.png)



```python
prediction = autoencoder.predict(M)
```


```python
plt.figure(figsize=(20,4))
print("Test Images")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(M[i, ..., 0], cmap='gray')
    plt.gray()
    
plt.figure(figsize=(20,4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(prediction[i, ..., 0], cmap='gray')
    plt.gray()
plt.show()
```

    Test Images
    Reconstruction of Test Images
    


![png](output_20_1.png)



![png](output_20_2.png)



```python
preds_0 = prediction[0]*255
preds_0 = preds_0.reshape(256,256)
x_test_0 = M[0]*255
x_test_0 = x_test_0.reshape(256,256)
plt.imshow(x_test_0, cmap='gray')
plt.gray()
```


![png](output_21_0.png)



```python
plt.imshow(preds_0, cmap='gray')
plt.gray()
```


![png](output_22_0.png)



```python

```


```python

```


```python

```
