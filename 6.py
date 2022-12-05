from IPython.display import display, clear_output
from tqdm.auto import tqdm, trange
from ipywidgets import Output
import camera_utils
import lpips
import os


#@markdown ## Geometry
#@markdown Default camera focal length (tune to match your camera's zoom, 3-6
#@markdown are good values)
focal_length =   4#@param {type: "number"}
focal_length = torch.nn.Parameter(torch.tensor([focal_length],
                                               dtype=torch.float32,
                                               device=device))
#@markdown Default offset (don't go below 2.8, you'll get body horror)
start_z = 3.2  #@param {type: "number"}
try:
    position
except NameError:
    position = torch.nn.Parameter(
        torch.tensor([0, 0, start_z], dtype=torch.float32, device=device))
try:
    rotation
except NameError:
    rotation = torch.nn.Parameter(
        torch.tensor([0, 0, 0], dtype=torch.float32, device=device))
#@markdown ## Init
#@markdown Random seed, 0 for no seed
seed =   25#@param {type: "integer"}
if seed == 0:
    import random
    seed = random.getrandbits(16)
print("Seed:", seed)
torch.manual_seed(seed)
#@markdown How much to scale down the intialisation by (more conservative)
init_scale = 0.9  #@param {type: "number"}
#@markdown Start from a new face latent instead of reusing (turn off if you
#@markdown know what you're doing)
generate_new = True  #@param {type: "boolean"}
try:
    z
    if generate_new:
        abracadabra  # What if someone makes a variable with that name?
except NameError:
    try:
        z_random
        if generate_new:
            abracadabra
    except NameError:
        z_random = torch.randn([1, G.z_dim], dtype=torch.float32, device=device)
    z = torch.nn.Parameter(z_random[0] * init_scale)


# It would be shorter to just hardcode
def rotation_matrix(angle, dim=0):
    s = angle.sin()
    c = angle.cos()
    dims = [d for d in range(3) if d != dim]
    return torch.zeros(9).to(angle).index_add(
        0,
        torch.LongTensor([dims[0] + dims[0] * 3,
                          dims[1] + dims[0] * 3,
                          dims[0] + dims[1] * 3,
                          dims[1] + dims[1] * 3, dim + dim * 3
                          ]).to(angle.device), torch.cat([c, s, -s, c,
                                                          angle * 0 + 1
                                                          ])).reshape(3, 3)


def rotate(euler):
    return (
        rotation_matrix(euler[0:1], 0)
        @ rotation_matrix(euler[1:2], 1)
        @ rotation_matrix(euler[2:3], 2))


def get_cam_matrix():
    return torch.tensor(
        [
            [0, 0, 0.5],
            [0, 0, 0.5],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).flatten().index_add(
            0,
            torch.LongTensor([0, 4]).to(device),
            torch.cat((focal_length,) * 2)).reshape(3, 3)


def make_c():
    matrix = torch.tensor([
                        [-1, 0,  0],
                        [0, -1,  0],
                        [0,  0, -1]
    ]).float().to(device)
    matrix = matrix @ rotate(rotation)
    # Position is not rotated
    pos = torch.maximum(torch.tensor([-10, -10, 2.6]).to(position),
                        position)
    matrix = torch.cat((matrix, pos.unsqueeze(-1)), dim=1)
    matrix = torch.cat((matrix, torch.eye(4)[-1:].to(device)), dim=0)
    cam_matrix = get_cam_matrix().flatten()
    return torch.cat((matrix.flatten(), cam_matrix)).float().to(device)


def render():
    return generate(z.unsqueeze(0), make_c()), weights(z.unsqueeze(0), make_c())


#@markdown ## Training
#@markdown (note: you can always interrupt execution and continue with different
#@markdown parameters)
# TODO Make the interrupt begin the fly-around iterations
# TODO show a higher-res image at the end? Or separate images
#@markdown Number of training interations
iterations =   1000  #@param {type: "integer"}
#@markdown Iterations in the end to just fly around
show_iterations = 200  #@param {type: "integer"}
#@markdown Display images every N training steps (make this higher if you see
#@markdown flickering or the notebook slows down)
show_every = 1  #@param {type: "integer"}
#@markdown Save every N steps
save_every = 60  #@param {type: "integer"}
#@markdown Save to this file
save_to = "EG3D.pt"  #@param {type: "string"}
#@markdown Master learning rate
lr =   0.4  #@param {type: "number"}
#@markdown Position, rotation and focal length learning rate
p_lr = 4e-2  #@param {type: "number"}
#@markdown Generator latent learning rate
z_lr = 2e-1  #@param {type: "number"}
#@markdown PTI learning rate
g_lr = 0.0  #@param {type: "number"}
#@,arldpwm Gradient clipping norm
clip_grad_norm = 10.0  #@param {type: "number"}
#@markdown How much to take position into account for loss (experimental)
weight_weight = 0.0  #@param {type: "number"}
#@markdown Loss weight for L2 regularization on latents
reg_weight = 0.01  #@param {type: "number"}
#@markdown Network to be used for LPIPS
lpips_net = "vgg"  #@param {type: "string"} ["vgg", "alex"]
#@markdown LPIPS loss weight
perceptual_weight = 0.2  #@param {type: "number"}
# TODO: add face landmark alignment loss

perceptual_loss = lpips.LPIPS(net=lpips_net).to(device)
optim = torch.optim.Adam([{"params": [position, rotation, focal_length],
                           "lr": p_lr * lr},
                          {"params": [z], "lr": z_lr * lr},
                          {"params": G.parameters(), "lr": g_lr * lr}])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations,
                                                       eta_min=1e-4)
out = Output()
display(out)
frames = []
try:
    with trange((iterations + show_iterations)) as bar:
        for i in bar:
            pic, weight = render()
            im = to_img(pic, as_torch=True)
            losses = ((im - target) ** 2).mean(dim=-1)
            weight = torch.nn.functional.interpolate(
                weight.unsqueeze(0).unsqueeze(0), (im.shape[0], im.shape[1]))[0, 0]
            weight = weight * weight_weight + (1 - weight_weight)
            mse = (losses * weight.detach()).mean()
            reg = torch.norm(z)
            perceptual = perceptual_loss(
                im.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1,
                target.unsqueeze(0).permute(0, 3, 1, 2).float() * 2 - 1)
            loss = (
                mse
                + reg * reg_weight
                + perceptual * perceptual_weight)
            bar.set_postfix(loss=loss.item(), mse=mse.item(), reg=reg.item(),
                            perceptual=perceptual.item())
            if i < iterations:
                loss.backward()
                for group in optim.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], clip_grad_norm)
                optim.step()
                optim.zero_grad()
                scheduler.step()
            frame = to_img(pic)
            angle = i / 10
            angled_c = camera_utils.LookAtPoseSampler().sample(
                np.sin(angle) / 2 + np.pi / 2,
                np.cos(np.sin(angle / 2) / 2) / 2 * np.pi,
                torch.zeros(3), radius=start_z)
            pic2 = generate(z.unsqueeze(0),
                            torch.cat((angled_c.flatten().to(device),
                                    get_cam_matrix().flatten())).float().to(
                                        device))
            depth = torch.nn.functional.interpolate(pic2["image_depth"], im.shape[:2]
                                            )[0, 0].cpu().detach().numpy()
            depth -= depth.min()
            depth *= depth.max()
            depth = np.tile((depth[:, :, np.newaxis] * 255).astype(np.uint8),
                            (1, 1, 3))
            frame = Image.fromarray(np.concatenate((
                np.concatenate((
                    np.asarray(frame),
                    depth
                ), axis=0),
                np.concatenate((
                    (np.asarray(target.detach().cpu().numpy())
                    * 255).astype(np.uint8), np.asarray(to_img(pic2))
                ), axis=0)
            ), axis=1))
            frames.append(frame)
            if i % show_every == show_every - 1:
                with out:
                    clear_output(wait=True)
                    display(frame)
            if i % save_every == save_every - 1:
                torch.save((z, (position, rotation, focal_length)), save_to)
except KeyboardInterrupt:
    print("Early exit")
