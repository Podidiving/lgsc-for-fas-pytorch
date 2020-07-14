from torchvision.utils import make_grid


class GridMaker:
    def __init__(self):
        pass

    def __call__(self, images, cues):
        b, c, h, w = images.shape
        images_min = images.view(b, -1).min(axis=1)[0][:, None]
        images_max = images.view(b, -1).max(axis=1)[0][:, None]
        images = (images.view(b, -1) - images_min) / (images_max - images_min)
        images = images.reshape(b, c, h, w)

        b, c, h, w = cues.shape
        cues_min = cues.view(b, -1).min(axis=1)[0][:, None]
        cues_max = cues.view(b, -1).max(axis=1)[0][:, None]
        cues = (cues.view(b, -1) - cues_min) / (cues_max - cues_min)
        cues = cues.reshape(b, c, h, w)

        return make_grid(images), make_grid(cues)
