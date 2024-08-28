from SSD import SSD
from PIL import Image

if __name__ == '__main__':
    ssd = SSD()

    image_path = 'img/street.jpg'
    image = Image.open(image_path)

    r_image = ssd.detect_image(image, crop=False, count=False)
    r_image.show()