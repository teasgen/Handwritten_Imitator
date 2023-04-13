from PIL import Image

from generate_image_by_font_text import draw_font, init_model
from remove_background import remove_background_from_image_with_text

init_model()

text = "f text"
font_removed_background = remove_background_from_image_with_text("got.jpg")
result = Image.fromarray(font_removed_background).convert('RGB')
result.save('removed_background.jpg')
generated = draw_font(font_removed_background, text)
result = Image.fromarray(generated).convert('RGB')
result.save("generated.jpg")
