import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dir = 'project/VoiceCommands/demo_kit/images/'
img_path = dir + 'arrow_left.png'
print(img_path)

# plot an image from the directory dir
img = mpimg.imread(img_path)
plt.imshow(img)
plt.show()
