from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import util
import wikipedia, os

app = Flask(__name__)  #creates flask instance

photos = UploadSet('photos', IMAGES) #use for uploading images to flask

app.config['UPLOADED_PHOTOS_DEST'] = './server/static/uploadedimg'
configure_uploads(app, photos)

#default route
@app.route('/')
def home():
    return render_template("index.html")


#route when clicked on upload button on web
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo']) #saving uploaded photos till model is evaluating it
        #print(filename)
        path = f'./server/static/uploadedimg/{filename}'
        #print(path)
        util.load_saved_artifacts()  #calling function to load saved pickled model

        try:
            actor_name=util.classify_image(None, path) #actually evaluating that image
            for x in range(len(actor_name)):
                #print(actor_name[x])
                actor_name=actor_name[x]  #converting actor name from dict to string for display
        except:
            actor_name = "Not identified"
        
        try:
            actor_desc = wikipedia.summary(actor_name, sentences=2, auto_suggest=False)
        except:
            actor_desc = "No description"

        os.remove(path)  #deleting uploaded image from our server
    return render_template("index.html",actor_name=actor_name,actor_desc=actor_desc)

if __name__ == '__main__':
    app.run(debug=True)