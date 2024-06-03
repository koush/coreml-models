from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import coremltools as ct
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor

# mtcnn = MTCNN(keep_all=True)
# img = Image.open("/Users/koush/Desktop/rotate.jpg").convert('RGB')
# faces = mtcnn(img)

# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# prev = None
# for img_cropped in faces:
#     img_embedding = resnet(img_cropped.unsqueeze(0))
#     if prev is not None:
#         dist = (img_embedding - prev).norm()
#     prev = img_embedding

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained="vggface2").eval()
# preferred input according to facenet-pytorch docs
example_input = torch.rand(1, 3, 160, 160)
traced_model = torch.jit.trace(resnet, example_input)
out = traced_model(example_input)
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
)
model.save("inception_resnet_v1.mlpackage")

# mtcnn = MTCNN(image_size=320, thresholds=[0,0,0])
# mtcnn.eval()
# example_input = torch.rand(1, 320, 320, 3)
# traced_model = torch.jit.trace(mtcnn, example_input)
# out = traced_model(example_input)
# model = ct.convert(
#     traced_model,
#     convert_to="mlprogram",
#     inputs=[ct.TensorType(shape=example_input.shape)],
# )
# model.save("mtcnn.mlpackage")
