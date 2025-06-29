import torch
from jaxtyping import Int, Bool


class Analyzer:
    data: list[tuple[Int, Int, Bool[torch.Tensor, "H W"]]]
    # data: list of frames, each frame is a tuple of (frame_index, object_index, mask tensor)
    # TODO turn data in __init__ into list, i.e. remove frame_index?

    def __init__(self, data: list[tuple[Int, Int, Bool[torch.Tensor, "H W"]]]):
        # TODO initialization to torch tensor
        self.data = data

    def zeroth_image_moment(self) -> int:
        """Calculate the zeroth moment (area) of a binary mask."""
        raise NotImplementedError
        # TODO return list of zeroth moment for each frame?
        return int(torch.sum(mask.float()).item())

    def first_image_moment(self) -> tuple[int, int]:
        """Calculate the centroid (first moment) of a binary mask."""
        raise NotImplementedError
        mask = mask.float()
        area = torch.sum(mask)
        if area == 0:
            return (0, 0)
        height, width = mask.shape
        x_coords = (
            torch.arange(width, device=mask.device).view(1, -1).expand(height, width)
        )
        y_coords = (
            torch.arange(height, device=mask.device).view(-1, 1).expand(height, width)
        )
        x = (x_coords * mask).sum() / area
        y = (y_coords * mask).sum() / area
        return int(round(x.item())), int(round(y.item()))

    def second_image_moment(self) -> tuple[int, int, int]:
        """Calculate the second moment of inertia of a binary mask."""
        raise NotImplementedError
        mask = self.data.float()
        area = torch.sum(mask)
        if area == 0:
            return (0, 0, 0)
        height, width = mask.shape
        y_coords = (
            torch.arange(height, device=mask.device).view(-1, 1).expand(height, width)
        )
        x_coords = (
            torch.arange(width, device=mask.device).view(1, -1).expand(height, width)
        )
        x = (x_coords * mask).sum() / area
        y = (y_coords * mask).sum() / area
        x_diff = x_coords - x
        y_diff = y_coords - y
        xx = (x_diff**2 * mask).sum() / area
        yy = (y_diff**2 * mask).sum() / area
        xy = (x_diff * y_diff * mask).sum() / area
        return int(round(xx.item())), int(round(yy.item())), int(round(xy.item()))
