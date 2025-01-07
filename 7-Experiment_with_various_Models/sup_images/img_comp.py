from PIL import Image
import matplotlib.pyplot as plt

image_256 = Image.open("./img(5)_n_ori.png")
image_512 = Image.open("./img(5)_n_2x.png")
image_1024 = Image.open("./img(5)_n_4x.png")

crop_box = (130, 130, 200, 200)  # Adjust coordinates for a visible area
crop_box_2x = tuple( value * 2 for value in crop_box) 
crop_box_4x = tuple( value * 2 for value in crop_box_2x) 

crop_256 = image_256.crop(crop_box)
crop_512 = image_512.crop(crop_box_2x)
crop_1024 = image_1024.crop(crop_box_4x)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(crop_256, cmap='gray')
axes[0].set_title("Original (256x256)")
axes[0].axis("off")

axes[1].imshow(crop_512, cmap='gray')
axes[1].set_title("Super Res. 2x")
axes[1].axis("off")

axes[2].imshow(crop_1024, cmap='gray')
axes[2].set_title("Super Res. 4x")
axes[2].axis("off")

plt.subplots_adjust(wspace=0.02)
plt.tight_layout()

plt.show()


