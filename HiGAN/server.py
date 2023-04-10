from io import BytesIO
import flask
import numpy as np
from flask import request, send_file
from PIL import Image

from generate_image_by_font_text import draw_font, init_model
from remove_background import remove_background_from_image_with_text

app = flask.Flask(__name__)


@app.route('/upload', methods=['POST'])
def make_handwritten():
    text = request.values['text']
    font = request.files['font']
    print(text)

    font_bytes = font.read()
    font_io = BytesIO(font_bytes)

    font = Image.open(font_io).convert("L")
    # font_np = np.asarray(font)
    # print(font_np.shape)
    font.save("got.jpg")
    font_removed_background = remove_background_from_image_with_text("got.jpg")
    result = Image.fromarray(font_removed_background).convert('RGB')
    result.save('removed_background.jpg')
    generated = draw_font(font_removed_background, text)
    result = Image.fromarray(generated).convert('RGB')
    result.save("generated.jpg")

    # Output
    img_byte = BytesIO()
    result.save(img_byte, format='JPEG')
    img_byte.seek(0)
    return send_file(img_byte, download_name="received.jpg")


if __name__ == '__main__':
    init_model()
    app.run(host="0.0.0.0", port=5000, debug=True)