from vgg_model import Model
import torch
from collections import OrderedDict
import coremltools as ct

vgg = Model(1, 256, 256, 97)

# download an url to file
import urllib.request

url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"

urllib.request.urlretrieve(url, "english_g2.zip")

# unzip the file
import zipfile

with zipfile.ZipFile("english_g2.zip", "r") as zip_ref:
    zip_ref.extractall(".")


# load file from local
state_dict = torch.load("english_g2.pth", map_location=torch.device("cpu"))

new_state_dict = OrderedDict()
for key, value in state_dict.items():
    new_key = key[7:]
    new_state_dict[new_key] = value
vgg.load_state_dict(new_state_dict)

vgg.eval()

# input is batch Luminance height width (which is multiple of 64)
example_input = torch.rand(1, 1, 64, 256)
traced_model = torch.jit.trace(vgg, example_input)
out = traced_model(example_input)
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
)
model.save("vgg_english_g2.mlpackage")
