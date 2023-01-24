import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

dir = 'VoiceCommands/demo_kit/images/'
img_path = dir + 'left.png'
print(img_path)


print(os.listdir(dir))

# plot an image from the directory dir
img = mpimg.imread(img_path)
plt.imshow(img)
plt.show()
