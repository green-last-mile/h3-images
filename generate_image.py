from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import click
import requests
from io import BytesIO
import numpy as np
import cv2

import h3

from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import Transformer


@dataclass
class Tile:
    """
    A tile in a map.

    #TODO: switch this to xyzservice or whatever that library is called
    """

    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    mapbox_style: str = "satellite-v9"
    image_height: int = 512
    image_width: int = 512

    def __post_init__(self):
        self.lat_min, self.lat_max = self.lat_range
        self.lon_min, self.lon_max = self.lon_range

        # self.mapbox_key 

    def _stringify(self):
        """
        Stringify the tile
        """
        return f"{self.lon_min},{self.lat_min},{self.lon_max},{self.lat_max}"

    def fetch_image(
        self,
    ):
        """
        Fetch the tile
        """
        assert os.environ.get("MAPBOX_KEY", False), "MAPBOX_KEY not set"

        url = f"https://api.mapbox.com/styles/v1/mapbox/{self.mapbox_style}/static/[{self._stringify()}]/{self.image_height}x{self.image_width}?access_token={os.environ.get('MAPBOX_KEY')}"
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")


@dataclass
class Hex:
    """
    An extended hexagon
    """

    h3: str

    def __post_init__(self):
        self.poly: Polygon = Polygon(h3.h3_to_geo_boundary(self.h3))

        self.utm_poly: Polygon = self._transform_to_utm()

    def build_tile(
        self,
        style: str,
        size: int
    ) -> Tile:
        """
        Build a tile from the hexagon
        """
        min_lat, min_lon, max_lat, max_lon = self.poly.bounds
        return Tile(
            lat_range=(min_lat, max_lat),
            lon_range=(min_lon, max_lon),
            mapbox_style=style,
            image_height=size,
            image_width=size,
        )

    def _transform_to_utm(
        self,
    ):
        transformer = Transformer.from_crs("EPSG:4326", self._estimate_crs())

        return transform(transformer.transform, self.poly)

    def _estimate_crs(
        self,
    ) -> CRS:
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                *self.poly.bounds,
            ),
        )

        if len(utm_crs_list) == 0:
            raise ValueError("No UTM CRS found for the given area of interest.")

        return CRS.from_epsg(utm_crs_list[0].code)

    @property
    def polygon_height(
        self,
    ) -> float:
        """
        Get the height of the polygon in meters
        """
        return self.utm_poly.bounds[3] - self.utm_poly.bounds[1]

    @property
    def polygon_width(
        self,
    ) -> float:
        """
        Get the width of the polygon in meters
        """
        return self.utm_poly.bounds[2] - self.utm_poly.bounds[0]

    @property
    def ref_y(
        self,
    ) -> float:
        """
        Get the reference y coordinate
        """
        return self.utm_poly.bounds[1]

    @property
    def ref_x(
        self,
    ) -> float:
        """
        Get the reference x coordinate
        """
        return self.utm_poly.bounds[0]

    def get_image_frame_polygon(
        self,
        img_size: Tuple[int, int],
    ) -> Polygon:
        """
        Get the polygon of the image frame
        """
        return transform(
            lambda x, y: (
                int(((x - self.ref_x) / self.polygon_width) * img_size[0]),
                int(((y - self.ref_y) / self.polygon_height) * img_size[1]),
            ),
            self.utm_poly,
        )


def mask_image(
    img_array: np.array,
    polygon: Polygon,
):
    """
    Mask an image with a polygon
    """
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    points = np.array(polygon.exterior.coords, dtype=np.int32)
    
    # this modifies the mask in place
    mask = cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img_array, img_array, mask=mask)

    wbg = np.ones_like(img_array, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    res += wbg
    return res


@click.command()
@click.option("--h3", required=True, help="The h3 index")
@click.option("--output", required=True, help="The output file")
@click.option("--style", default="satellite-v9", help="The mapbox style")
@click.option("--tile-res", default=512, help="The resolution of the tile (square)")
def main(h3: str, output: str, style: str, tile_res: int) -> None:
    
    # if output is a directory, then we need to create a filename (just use the h3 index
    # for now)
    output = Path(output)
    if output.is_dir():
        output = output / f"{h3}.png"
    
    hex_ = Hex(h3=h3)

    tile = hex_.build_tile(style=style, size=tile_res)

    img = tile.fetch_image()
    img_array = np.array(img)

    polygon = hex_.get_image_frame_polygon(img_array.shape[:2])

    res = mask_image(img_array, polygon)

    cv2.imwrite(str(output), res)


if __name__ == "__main__":
    main()
