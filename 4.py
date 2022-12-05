#@title Example: generate random image
from matplotlib import pyplot as plt
from PIL import Image


c = torch.cat((torch.tensor([
                                [-1, 0,  0,  0],
                                [0,  -1, 0,  0],
                                [0,  0,  -1, 2.7],
                                [0,  0,  0,  1]
]).flatten(), torch.tensor([
                                        [3, 0, 0.5],
                                        [0, 3, 0.5],
                                        [0, 0, 5]
]).flatten()))  # TODO
#@markdown Psi truncation (0-1)
psi = 0.5  #@param {type: "number"}
def generate(z, c=c, device=device, psi=psi):
    return G(z, c.to(device).unsqueeze(0).repeat(z.shape[0], 1),
             truncation_psi=psi)


def weights(z, c=c, device=device):
    planes = G.backbone(z,
                        c.unsqueeze(0).to(device).repeat(z.shape[0], 1)
                        )[0][[34, 52]]  # TODO make batched
    planes -= planes.min(dim=1)[0].min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
    planes /= planes.max(dim=1)[0].max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
    return planes[0] * 0.8 + planes[1] * 0.2


def to_img(img, as_torch=False, as_numpy=False):
    img = img["image"][0].permute(1, 2, 0) / 3 + 0.5
    if as_torch:
        return img
    img = img.detach().cpu().numpy()
    if as_numpy:
        return img
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


z_random = torch.randn([1, G.z_dim]).cuda()
img = generate(z_random)
plt.axis("off")
plt.imshow(img["image_depth"].detach().cpu().numpy()[0, 0])
plt.colorbar()
plt.show()
to_img(img)
