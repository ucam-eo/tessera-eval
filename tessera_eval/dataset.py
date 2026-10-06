"""Which TESSERA embeddings dataset tessera-eval reads.

Pinned explicitly rather than left to geotessera's default, because that
default changes between releases: geotessera 0.11.0 made a bare
``GeoTessera()`` mean v1.1-cambridge (a sparse test run -- e.g. 0 tiles for
Austria 2022, where v1.0 has 70) and a bare ``GeoTesseraZarr()`` mean
v1.1-dclimate. With only ``geotessera>=0.10.1`` as the requirement, a fresh
install silently switched every evaluation to different embeddings.

Moving to another dataset (e.g. the wall-to-wall v1.1-dclimate) should be a
deliberate change here, not a side effect of a dependency upgrade.
"""

EMBEDDINGS_DATASET_VERSION = "v1.0"


def zarr_store_url():
    """The Zarr store URL for EMBEDDINGS_DATASET_VERSION."""
    from geotessera.registry import zarr_store_url as _url

    return _url(EMBEDDINGS_DATASET_VERSION)
