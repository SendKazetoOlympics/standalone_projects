from jaxtyping import Int, Bool
import torch


class Analyzer:
    data: list[tuple[Int, Int, Bool[torch.Tensor, "H W"]]]
    # data: list of frames, each frame is a tuple of (frame_index, object_index, mask tensor)

    def __init__(self, data: list[tuple[Int, Int, Bool[torch.Tensor, "H W"]]]):
        self.data = data

    @staticmethod
    def zeroth_image_moment(mask: Bool[torch.Tensor, "H W"]) -> int:
        """Calculate the zeroth moment (area) of a binary mask."""
        return int(torch.sum(mask.float()).item())

    @staticmethod
    def first_image_moment(mask: Bool[torch.Tensor, "H W"]) -> tuple[int, int]:
        """Calculate the centroid (first moment) of a binary mask."""
        mask = mask.float()
        area = torch.sum(mask)
        if area == 0:
            return (0, 0)
        height, width = mask.shape
        x_coords = torch.arange(width, device=mask.device).view(1, -1).expand(height, width)
        y_coords = torch.arange(height, device=mask.device).view(-1, 1).expand(height, width)
        x = (x_coords * mask).sum() / area
        y = (y_coords * mask).sum() / area
        return int(round(x.item())), int(round(y.item()))

    @staticmethod
    def second_image_moment(mask: Bool[torch.Tensor, "H W"]) -> tuple[int, int, int]:
        """Calculate the second moment of inertia of a binary mask."""
        mask = mask.float()
        area = torch.sum(mask)
        if area == 0:
            return (0, 0, 0)
        height, width = mask.shape
        y_coords = torch.arange(height, device=mask.device).view(-1, 1).expand(height, width)
        x_coords = torch.arange(width, device=mask.device).view(1, -1).expand(height, width)
        x = (x_coords * mask).sum() / area
        y = (y_coords * mask).sum() / area
        x_diff = x_coords - x
        y_diff = y_coords - y
        xx = (x_diff ** 2 * mask).sum() / area
        yy = (y_diff ** 2 * mask).sum() / area
        xy = (x_diff * y_diff * mask).sum() / area
        return int(round(xx.item())), int(round(yy.item())), int(round(xy.item()))
