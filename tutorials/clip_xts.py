from corenet.options.opts import get_training_arguments
from corenet.modeling import get_model
from PIL import Image
import torch
import os
from torchvision.transforms import Compose, Resize, PILToTensor, CenterCrop
from corenet.data.text_tokenizer import build_tokenizer


# configuration file path
config_file = os.path.join(
    os.getcwd(), "..", "projects/range_augment/clip/clip_vit_base_test.yaml"
)
# pre-trained weights
pretrained_weights = "https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt"

opts = get_training_arguments(
    args=[
        "--common.config-file",
        config_file,
        "--model.multi-modal-image-text.prtrained",
        pretrained_weights,
    ]
)

# build the model
model = get_model(opts)
# set the model in evaluation mode.
model.eval()

# build the text tokenizer. It is useful to convert the text description into tokens.
text_tokenizer = build_tokenizer(opts)

## STEP 1: Choose Textual description

example_class_names = ["cat", "horse", "dog"]
input_text_templates = [
    "a photo of a {}".format(class_name) for class_name in example_class_names
]
# Context length is 7 because we have 5 words in text template + beginning of text token + end of text token
context_length = 7

tokenized_input_templates = [
    text_tokenizer(inp_template) for inp_template in input_text_templates
]
# The size of tokenized_input_templates after stacking would be [num_classes, context_length]. In this case,
# num_classes=3 (cat, horse, dog) and context_length=7
tokenized_input_templates = torch.stack(tokenized_input_templates, dim=0)
print(tokenized_input_templates.shape)

## STEP 2: Encode Textual description

# The expected input to text encoder is [batch_size, num_classes, num_captions, context_length]
# For this example, we have batch_size=1 and num_captions=1.
# So, we add dummy dimensions to tokenized_input_templates to convert its shape from [num_classes, context_length] to [batch_size, num_classes, num_captions, context_length]

tokenized_input_templates = tokenized_input_templates[None, :, None, :]

# produce text_embeddings
with torch.no_grad():
    text_embeddings = model.text_encoder(tokenized_input_templates)
# The shape of text embeddings is [hidden_dim, num_classes]
print(text_embeddings.shape)

## STEP 3: Encode image

## STEP 3.1: Read an image
img_path = os.path.join(os.getcwd(), "..", "assets", "dog.jpeg")
image = Image.open(img_path).convert("RGB")
print(image)

## STEP 3.2: Transform an image
# The model is pre-trained using 224x224 resolution. Therefore, we resize the input PIL image while maintaining an aspect ratio,
# such that shorter dimension is 224. We then crop 224x224 image from the center. After that, we convert the resized PIL image into a tensor.
# The values in tensor range between 0 and 255. We covert it to float and normalize it between 0.0 and 1.0 by dividing the tensor by 255.

img_transforms = Compose([Resize(size=224), CenterCrop(size=224), PILToTensor()])

# Transform the image and normalize it between 0 and 1
input_img_tensor = img_transforms(image)
input_img_tensor = input_img_tensor.to(torch.float).div(255.0)

# add dummy batch dimension
input_img_tensor = input_img_tensor[None, ...]
print(input_img_tensor.shape)

## STEP 3.3 Encode an image
with torch.no_grad():
    image_embeddings = model.image_encoder(input_img_tensor)["logits"]
# the shape of image_embeddings is [batch_size, hidden_dim]
print(image_embeddings.shape)

## STEP 4: Compute similarity score

# Compute a dot product between image and text  embeddings
# [batch_size, hidden_dim] x [hidden_dim, num_classes] --> [batch_size, num_classes]
similarity_score = image_embeddings @ text_embeddings
print(similarity_score.shape)

## STEP 5: Clasify image

predicted_class_prob, predicted_class_id = torch.max(similarity_score, dim=-1)
n_samples = predicted_class_prob.shape[0]
for batch_id in range(n_samples):
    print(
        f"Predicted class for sample {batch_id} is {example_class_names[predicted_class_id[batch_id]]} (p={predicted_class_prob[batch_id]:.2f})"
    )