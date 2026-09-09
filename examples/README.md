# Example data

## `austria_crops.geojson`

A small worked-example dataset for the tessera-eval quickstart: **349 field
parcels near Vienna, Austria, labelled by crop type** (`crop` column, 10
classes — Winter Grain, Fallow, Soy, Beet, Sunflower, Corn, …). EPSG:4326,
~0.05° × 0.05°, so `load_embeddings_for_shapefile` pulls only a handful of
Tessera tiles.

Derived (spatial subset + light geometry simplification) from an openly
licensed Austrian agricultural parcel dataset.

```python
import geopandas as gpd
from geotessera import GeoTessera
from tessera_eval import load_embeddings_for_shapefile, run_kfold_cv

gdf = gpd.read_file("examples/austria_crops.geojson")

vectors, labels, class_names, stats = load_embeddings_for_shapefile(
    gdf, field="crop", year=2024, gt_instance=GeoTessera()
)
print(f"{len(labels):,} labelled pixels across {len(class_names)} classes")

for event in run_kfold_cv(vectors, labels, ["rf"], k=5, task="classification"):
    if event["type"] == "aggregate":
        m = event["models"]["rf"]
        print(f"macro-F1: {m['mean_f1']:.3f} ± {m['std_f1']:.3f}")
```
