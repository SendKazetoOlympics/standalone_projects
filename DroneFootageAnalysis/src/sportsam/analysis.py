from pathlib import Path

import torch
from jaxtyping import Int, Bool


class Analyzer:
    # data: dict[Int, dict[Int, dict[Int, Bool[torch.Tensor, "H W"]]]]
    # data: dict of videos, which is a dict of frames, each frame is a dict of (object_index, mask tensor)

    # def __init__(
    #     self,
    #     data: dict[Int, dict[Int, Bool[torch.Tensor, "H W"]]],
    #     videos: dict[Int, Int],
    # ):
    #     # TODO split by video frames from io_handler
    #     self.data = data

    @staticmethod
    def _load_data_from_dir(
        frames_dir: Path,
    ) -> dict[Int, dict[Int, Bool[torch.Tensor, "H W"]]]:
        raise NotImplementedError

    @staticmethod
    def zeroth_image_moment(frames_dir: Path, object_id: Int = 1) -> dict[Int, Int]:
        """Calculate the zeroth moment (area) of binary masks for each frame.

        Args:
            frames_dir (Path): Path object to the current directory of where all the data to be analyzed is stored.
            object_id (Int): Object ID to calculate area for. Defaults to 1.

        Returns:
            A dictionary mapping frame indices to the area of the specified object in that frame.
            Area is calculated by summing all True values in the binary mask.
            Only includes frames where the specified object exists.
        """
        data = Analyzer._load_data_from_dir(frames_dir)

        result = {}
        for frame_idx, objects in data.items():
            # Calculate area only for specified object if it exists in this frame
            if object_id in objects:
                mask = objects[object_id]
                result[frame_idx] = int(torch.sum(mask.float()).item())
        return result

    # TODO match first and second image moments
    def first_image_moment(
        self, video_idx: Int, object_id: Int = 1
    ) -> dict[Int, tuple[Int, Int]]:
        """Calculate the first moment (centroid) of binary masks for each frame.

        Args:
            video_idx: Index of video in io_handler dict.
            object_id: Object ID to calculate centroid for. Defaults to 1.

        Returns:
            A dictionary mapping frame indices to (x,y) centroid coordinates of the specified object.
            Centroid is calculated as the weighted average of x,y coordinates using the binary mask.
            Only includes frames where the specified object exists.
        """
        result = {}
        for frame_idx, objects in self.data[video_idx].items():
            # Calculate centroid only for specified object if it exists in this frame
            if object_id in objects:
                mask = objects[object_id].float()  # Convert to float for calculations
                area = torch.sum(mask)

                # Skip if mask is empty
                if area == 0:
                    continue

                height, width = mask.shape
                x_coords = (
                    torch.arange(width, device=mask.device)
                    .view(1, -1)
                    .expand(height, width)
                )
                y_coords = (
                    torch.arange(height, device=mask.device)
                    .view(-1, 1)
                    .expand(height, width)
                )

                # Calculate weighted average of coordinates
                x = (x_coords * mask).sum() / area
                y = (y_coords * mask).sum() / area

                # Store rounded integer coordinates
                result[frame_idx] = (int(round(x.item())), int(round(y.item())))
        return result

    def second_image_moment(
        self, video_idx: Int, object_id: Int = 1
    ) -> dict[Int, tuple[Int, Int, Int]]:
        """Calculate the second moment of inertia of binary masks for each frame.

        Args:
            video_idx: Index of video in io_handler dict.
            object_id: Object ID to calculate moments for. Defaults to 1.

        Returns:
            A dictionary mapping frame indices to (xx, yy, xy) moments of inertia:
            - xx: spread of mass in x direction
            - yy: spread of mass in y direction
            - xy: correlation between x and y spread
            Only includes frames where the specified object exists.
        """
        result = {}
        for frame_idx, objects in self.data[video_idx].items():
            # Calculate moments only for specified object if it exists in this frame
            if object_id in objects:
                mask = objects[object_id].float()  # Convert to float for calculations
                area = torch.sum(mask)

                # Skip if mask is empty
                if area == 0:
                    continue

                height, width = mask.shape
                x_coords = (
                    torch.arange(width, device=mask.device)
                    .view(1, -1)
                    .expand(height, width)
                )
                y_coords = (
                    torch.arange(height, device=mask.device)
                    .view(-1, 1)
                    .expand(height, width)
                )

                # Calculate centroid first
                x = (x_coords * mask).sum() / area
                y = (y_coords * mask).sum() / area

                # Calculate distances from centroid
                x_diff = x_coords - x
                y_diff = y_coords - y

                # Calculate moments of inertia
                xx = (x_diff**2 * mask).sum() / area
                yy = (y_diff**2 * mask).sum() / area
                xy = (x_diff * y_diff * mask).sum() / area

                # Store rounded integer moments
                result[frame_idx] = (
                    int(round(xx.item())),
                    int(round(yy.item())),
                    int(round(xy.item())),
                )
        return result
