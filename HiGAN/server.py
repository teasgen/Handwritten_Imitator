from io import BytesIO
import flask
from flask import request, send_file
from PIL import Image
import numpy as np


from generate_image_by_font_text import draw_font, init_model
from remove_background import remove_background_from_image_with_text

app = flask.Flask(__name__)


@app.route('/upload', methods=['POST'])
def make_handwritten():
    text = request.values['text']
    number_of_symbols = int(request.values['number_of_symbols'])
    font = request.files['font']

    font_bytes = font.read()
    font_io = BytesIO(font_bytes)

    font = Image.open(font_io).convert("L")
    font.save("got.jpg")
    font_removed_background = remove_background_from_image_with_text("got.jpg")
    generated_rows = []

    for i in range(len(text)):
        j = i
        last_space = -1
        current_text = ""
        while j < len(text) and text[j] == " ":
            j += 1
        while j < len(text) and j - i < number_of_symbols:
            if text[j] == " ":
                last_space = j
            if text[j] == "\n":
                j += 1
                break
            current_text += text[j]
            j += 1
        if j < len(text) and j - 1 >= 0 and last_space != -1 and text[j - 1].isalpha() and current_text[-1].isalpha():
            j = last_space
            current_text = text[i: last_space + 1]
        i = j
    
        generated_rows.append(draw_font(font_removed_background, current_text))
    
    # stack rows into one image
    result = Image.fromarray(np.vstack(generated_rows)).convert('RGB')
    result.save("generated.jpg")

    # Output
    img_byte = BytesIO()
    result.save(img_byte, format='JPEG')
    img_byte.seek(0)
    return send_file(img_byte, download_name="received.jpg")


if __name__ == '__main__':
    init_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
