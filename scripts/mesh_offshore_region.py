import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
import numpy as np
import pandas as pd
from _helpers import REGION_COLS, configure_logging

def calculate_area(shape, ellipsoid="WGS84"):
    geod = Geod(ellps=ellipsoid)
    return abs(geod.geometry_area_perimeter(shape)[0]) / 1e6

def fill_shape_with_points(shape, num=50):
    """
    Fills the shape of the offshore region with points. This is needed for
    splitting the regions into smaller regions.

    Parameters
    ----------
    shape :
        Shape of the region.
    oversize_factor : int
        Factor by which the original region is oversized.
    num : int, optional
        Number of points added in the x and y direction.

    Returns
    -------
    inner_points :
        Returns a list of points lying inside the shape.
    """

    inner_points = list()
    x_min, y_min, x_max, y_max = shape.bounds
    iteration = 0
    for x in np.linspace(x_min, x_max, num=num):
        for y in np.linspace(y_min, y_max, num=num):
            if Point(x, y).within(shape):
                inner_points.append((x, y))

    return inner_points

def transform_points(points, source="4326", target="3035"):
    points = gpd.GeoSeries.from_xy(points[:, 0], points[:, 1], crs=source).to_crs(
        target
    )
    points = np.asarray([points.x, points.y]).T
    return points


def cluster_points(n_clusters, point_list):
    """
    Clusters the inner points of a region into n_clusters.

    Parameters
    ----------
    n_clusters :
        Number of clusters
    point_list :
        List of inner points.

    Returns
    -------
        Returns list of cluster centers.
    """
    point_list = transform_points(np.array(point_list), source="4326", target="3035")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(point_list)
    cluster_centers = transform_points(
        np.array(kmeans.cluster_centers_), source="3035", target="4326"
    )
    return cluster_centers

def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py
    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    # Convert shapes to equidistant projection shapes
    outline = gpd.GeoSeries(outline, crs="4326").to_crs("3035")[0]
    points = transform_points(points, source="4326", target="3035")

    if len(points) == 1:
        polygons = gpd.GeoSeries(outline, crs="3035").to_crs(4326).values
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = poly.intersection(outline)

            polygons.append(poly)

        polygons = gpd.GeoSeries(polygons, crs="3035").to_crs(4326).values

    return polygons

def build_voronoi_cells(shape, points):
    """
    Builds Voronoi cells from given points in the given shape.

    Parameters
    ----------
    shape :
        Shape where to build the cells in 4326 crs.
    points :
        List of points in 4326 crs.

    Returns
    -------
    split region
        Geopandas DataFrame containing the split regions.
    """
    cells = voronoi_partition_pts(points, shape)
    regions = gpd.GeoDataFrame(
        {
            "geometry": cells,
        }
    )
    return regions

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("mesh_offshore_region")
    configure_logging(snakemake)

    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_region = gpd.read_file(snakemake.input.offshore_region)
    offshore_shapes_sea = gpd.GeoDataFrame.copy(offshore_shapes)
    offshore_shapes_sea['geometry'] = offshore_shapes_sea.intersection(offshore_region.geometry.unary_union, align=True)
    offshore_shapes_rest = gpd.GeoDataFrame.copy(offshore_shapes)
    offshore_shapes_rest['geometry'] = offshore_shapes.difference(offshore_region.geometry.unary_union, align=True)
    c = offshore_shapes_rest.geometry.is_empty
    offshore_shapes_rest = offshore_shapes_rest.drop(offshore_shapes_rest.loc[c].index)

    countries = snakemake.config["countries"]

    offshore_shapes_sea['area'] = offshore_shapes_sea.geometry.map(
        lambda x: calculate_area(x)
    )

    offshore_shapes_sea = offshore_shapes_sea.set_index("name")
    offshore_shapes_rest = offshore_shapes_rest.set_index("name")

    threshold_area = snakemake.config["mesh_offshore_region"]["threshold"]
    offshore_regions = []

    for country in countries:
        shape = offshore_shapes_sea.loc[country]
        inner_points = fill_shape_with_points(shape.geometry)
        oversize_factor = shape.area/threshold_area
        n_regions = int(np.ceil(oversize_factor))
        region_centers = cluster_points(n_regions, inner_points)
        inner_regions = build_voronoi_cells(shape.geometry, region_centers)
        if country in offshore_shapes_rest.index:
            inner_regions = pd.concat([gpd.GeoDataFrame({"geometry": offshore_shapes_rest.loc[country]}), inner_regions], ignore_index=True)
        inner_regions.set_index(
            pd.Index([f"off_{country}_{i}" for i in inner_regions.index], name="region"),
            inplace=True,
        )
        inner_regions["name"] = inner_regions.index
        inner_regions["country"] = country
        offshore_regions.append(inner_regions)
        
    offshore_regions = pd.concat(offshore_regions, ignore_index=True)
    centroid = offshore_regions.to_crs(3035).centroid.to_crs(4326)
    offshore_regions["x"] = centroid.x
    offshore_regions["y"] = centroid.y

    offshore_regions.plot()

    offshore_regions.to_file(snakemake.output.meshed_offshore_shapes)