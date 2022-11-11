from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import util
import wikipedia

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'F:/facial recogition/server/static/uploadedimg'
configure_uploads(app, photos)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        print(filename)
        path = f'F:/facial recogition/server/static/uploadedimg/{filename}'
        print(path)
        util.load_saved_artifacts()

        try:
            actor_name=util.classify_image(None, path)
            for x in range(len(actor_name)):
                print(actor_name[x])
                actor_name=actor_name[x]
        except:
            actor_name = "Not identified"
        
        try:
            actor_desc = wikipedia.summary(actor_name, sentences=2)
        except:
            actor_desc = ""

    return render_template("index.html",actor_name=actor_name,actor_desc=actor_desc)

if __name__ == '__main__':
    app.run(debug=True)