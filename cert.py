from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def cert(id, face, name, class_tag):
    face = face.resize((300, 300))
    fontsize = 36
    font = ImageFont.truetype("arial.ttf", fontsize, encoding="unic")
    text_position_1 = (300, 350)
    face_position = (250, 425)
    text_position_2 = (250, 750)
    if class_tag == 'средней красоты':
        text_position_3 = (200, 850)
    elif class_tag == 'красивый':
        text_position_3 = (250, 850)
    else:
        text_position_3 = (230, 850)
    text_color = (0, 0, 0)
    img = Image.open('certificate.jpg')
    img.paste(face, face_position)
    # img.show()
    draw = ImageDraw.Draw(img)
    text_1 = "О том, что"
    text_2 = name
    text_3 = "{0} человек".format(class_tag)
    draw.text(text_position_1, text_1, text_color, font)
    draw.text(text_position_2, text_2, text_color, font)
    draw.text(text_position_3, text_3, text_color, font)
    # img.save('certificate_{0}.jpg'.format(id))
    return img

# cert(0, 'Антон Смирнов', 'красивый')
