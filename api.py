import os
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
from PIL import Image
import torch
import requests


allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
app = Flask(__name__)

def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

model2 = torch.hub.load(
	"AK391/animegan2-pytorch:main",
	"generator",
	pretrained=True,
	device="cpu",
	progress=False
)

model1 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1",  device="cpu")
face2paint = torch.hub.load(
	'AK391/animegan2-pytorch:main', 'face2paint', 
	size=512, device="cpu",side_by_side=False
)

def inference(img, ver):
	if ver == 1:
		out = face2paint(model1, img)
	else:
		out = face2paint(model2, img)
	return out


@app.route("/",methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		print(request.form['file'])
		#url = "https://huggingface.co/spaces/akhaliq/AnimeGANv2/resolve/main/IU.png"
		url = request.form['file']
		img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
		#img = Image.open(file.stream)
		out = inference(img, 2)
		buffered = BytesIO()
		out.save(buffered, format="JPEG")
		encoded_string = base64.b64encode(buffered.getvalue()).decode()
		#encoded_string = base64.b64encode(bytearray(out)).decode()        
		return render_template('index.html', img_data=encoded_string), 200
	else:
		return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', port="5004")