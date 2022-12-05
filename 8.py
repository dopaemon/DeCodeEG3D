#@title (Experimental) 3D Export
import trimesh


im = to_img(pic, as_numpy=True)
im = im[::4, ::4]
focus = (1 / focal_length).item()
step = focus / im.shape[0] * 2
rays = np.mgrid[-focus:focus:step,
                -focus:focus:step]
pic = generate(z.unsqueeze(0), make_c())
depth = torch.nn.functional.interpolate(pic["image_depth"], im.shape[:2]
                                        )[0, 0].cpu().detach().numpy()
plt.axis("off")
plt.imshow(depth)
plt.colorbar()
plt.show()
xyz = np.concatenate(((rays * depth).transpose(1, 2, 0),
                      depth[..., np.newaxis]),
                     axis=-1).reshape(-1, 3)
colors = im.reshape(-1, 3)
faces = []
c = lambda x, y: y * im.shape[1] + x
for y in trange(im.shape[0]):
    for x in range(im.shape[1]):
        if x > 0 and y > 0:
            faces.append([c(x, y), c(x, y - 1), c(x - 1, y)])
        if x < im.shape[1] - 1 and y < im.shape[0] - 1:
            faces.append([c(x, y), c(x, y + 1), c(x + 1, y)])
face_colors = np.asarray([colors[i[0]] for i in faces])
texture = Image.fromarray((im * 255).astype(np.uint8))
uv = np.mgrid[0:im.shape[0], 0:im.shape[1]].reshape(2, -1).T
tri_mesh = trimesh.Trimesh(vertices=xyz[..., [1, 0, 2]] * -1,
                           faces=faces,
                           vertex_colors=colors,
                           smooth=True)
