#@title Invert your own image
from PIL import Image
import requests
import io
import os


download = lambda x: Image.open(io.BytesIO(requests.get(x).content))
#@markdown Image URL. Leave empty if you upload your own
url = "" #@param ["https://thispersondoesnotexist.com/image", ""] {allow-input: true}
#@markdown (Optional) image path if you upload your own
path = "face_image.png"  #@param {type: "string"}
if len(url):
    target = download(url)
else:
    if not os.path.exists(path):
        from google.colab import files
        print("Upload image:")
        uploaded = list(files.upload())
        if uploaded:
            path = uploaded[0]
        else:
            print("Please upload something. Erroring out soon...")
    target = Image.open(path)
target = torch.from_numpy(np.asarray(target.convert("RGB").resize(
    (512, 512))) / 255).to(device)
Image.fromarray((target.detach().cpu().numpy() * 255).astype(np.uint8))
