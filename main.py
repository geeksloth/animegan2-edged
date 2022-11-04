#Step 1 prepare eng
from PIL import Image
import torch

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

#import matplotlib.pyplot as plt
from PIL import Image
import requests
'''
def display(before,after1,after2):
	plt.figure(figsize=(30,10))
	plt.subplot(1, 3, 1)
	plt.imshow(before)
	plt.subplot(1, 3, 2)
	plt.imshow(after1)
	plt.subplot(1, 3, 3)
	plt.imshow(after2)
'''
url = "https://huggingface.co/spaces/akhaliq/AnimeGANv2/resolve/main/IU.png"
before = Image.open(requests.get(url, stream=True).raw).convert('RGB') #png to jpeg
after_v1 = inference(before,1)
after_v2 = inference(before,2)

after_v1.save("after_v1.jpg")
after_v2.save("after_v2.jpg")
#display(before,after_v1, after_v2)