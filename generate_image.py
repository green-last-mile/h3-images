from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import click
import requests
from io import BytesIO
import numpy as np
import cv2
import h3
from shapely.geometry import Polygon
from shapely.ops import transform


@dataclass
class Tile:
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    mapbox_style: str = "satellite-v9"
    image_height: int = 512
    image_width: int = 512

    def __post_init__(self):
        self.lat_min, self.lat_max = self.lat_range
        self.lon_min, self.lon_max = self.lon_range

    def _stringify(self):
        return f"{self.lon_min},{self.lat_min},{self.lon_max},{self.lat_max}"

    def fetch_image(self):
        assert os.environ.get("MAPBOX_KEY", False), "MAPBOX_KEY not set"

        url = f"https://api.mapbox.com/styles/v1/mapbox/{self.mapbox_style}/static/[{self._stringify()}]/{self.image_height}x{self.image_width}?access_token={os.environ.get('MAPBOX_KEY')}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error fetching image: {response.content}")  # Error information
            return None

        try:
            return Image.open(BytesIO(response.content)).convert("RGB")
        except UnidentifiedImageError as e:
            print(f"Error in image decoding: {e}")
            return None


@dataclass
class Hex:
    h3: str

    def __post_init__(self):
        self.poly: Polygon = Polygon(h3.cell_to_boundary(self.h3, geo_json=True))

    def build_tile(self, style: str, size: int) -> Tile:
        min_lon, min_lat, max_lon, max_lat = self.poly.bounds
        return Tile(
            lat_range=(min_lat, max_lat),
            lon_range=(min_lon, max_lon),
            mapbox_style=style,
            image_height=size,
            image_width=size,
        )

    def get_image_frame_polygon(self, img_size: Tuple[int, int]) -> Polygon:
        """
        Transforms the polygon to the image frame coordinates.
        """
        min_x, min_y, max_x, max_y = self.poly.bounds
        polygon_width = max_x - min_x
        polygon_height = max_y - min_y

        return transform(
            lambda x, y: (
                int(((x - min_x) / polygon_width) * img_size[0]),
                int(((y - min_y) / polygon_height) * img_size[1]),
            ),
            self.poly,
        )


def mask_image(img_array: np.array, polygon: Polygon):
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    points = np.array(polygon.exterior.coords, dtype=np.int32)
    mask = cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img_array, img_array, mask=mask)
    wbg = np.ones_like(img_array, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    res += wbg
    return res


def run(
    h3_index: str, style: str = "satellite-v9", tile_res: int = 512, output: str = None
):
    hex_ = Hex(h3=h3_index)
    tile = hex_.build_tile(style, tile_res)
    img = tile.fetch_image()
    if img is not None:
        img_array = np.array(img)
        frame_polygon = hex_.get_image_frame_polygon(img_array.shape[:2])
        masked_img = mask_image(img_array, frame_polygon)

        if output:
            output_path = Path(output)
            if output_path.is_dir():
                output_path = output_path / f"{h3_index}.png"
            cv2.imwrite(str(output_path), masked_img)


@click.command()
@click.option("--h3", required=True, help="The h3 index")
@click.option("--output", required=True, help="The output file")
@click.option("--style", default="satellite-v9", help="The mapbox style")
@click.option("--tile-res", default=512, help="The resolution of the tile (square)")
def main(h3: str, output: str, style: str, tile_res: int):
    run(h3, style, tile_res, output)


if __name__ == "__main__":
    main()
