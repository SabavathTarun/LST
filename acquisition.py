"""
acquisition.py — STC Universal LST Wizard (Tab 1 engine)
=========================================================
Python port of the "Landsat 8 & 9 Satellite Imagery Downloader / Band Exporter":

  * Scene search  -> Microsoft Planetary Computer STAC (collection landsat-c2-l2,
                     i.e. the official USGS Collection 2 Level-2 science products)
  * Asset signing -> Planetary Computer SAS endpoint
  * Download      -> either the full original USGS tile, or only the pixels inside the
                     study-area window (HTTP range reads on the Cloud-Optimised GeoTIFF),
                     which is far smaller/faster and gives identical LST results because
                     the LST tab clips to the same boundary anyway.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION = "landsat-c2-l2"


# --------------------------------------------------------------------------------------
# Band catalogue (same bands / asset keys as the original downloader)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    id: str
    asset_key: str
    name: str
    wavelength: str
    category: str


BANDS: list[Band] = [
    Band("B1", "coastal", "Band 1 – Coastal / Aerosol", "0.43–0.45 µm", "visible"),
    Band("B2", "blue", "Band 2 – Blue", "0.45–0.51 µm", "visible"),
    Band("B3", "green", "Band 3 – Green", "0.53–0.59 µm", "visible"),
    Band("B4", "red", "Band 4 – Red", "0.64–0.67 µm", "visible"),
    Band("B5", "nir08", "Band 5 – Near Infrared (NIR)", "0.85–0.88 µm", "infrared"),
    Band("B6", "swir16", "Band 6 – SWIR-1", "1.57–1.65 µm", "infrared"),
    Band("B7", "swir22", "Band 7 – SWIR-2", "2.11–2.29 µm", "infrared"),
    Band("B10", "lwir11", "Band 10 – Surface Temperature (Thermal)", "10.60–11.19 µm", "thermal"),
    Band("QA_PIXEL", "qa_pixel", "QA Pixel (cloud / shadow / snow flags)", "bitmask", "qa"),
    Band("QA_RADSAT", "qa_radsat", "Radiometric Saturation QA", "bitmask", "qa"),
    Band("QA_AEROSOL", "qa_aerosol", "Aerosol QA", "bitmask", "qa"),
    Band("MTL_TXT", "mtl.txt", "Scene metadata (MTL.txt)", "ASCII", "metadata"),
]
BAND_BY_ID = {b.id: b for b in BANDS}

# The three bands + metadata the LST tab needs (SR_B4, SR_B5, ST_B10, MTL)
LST_ESSENTIALS = ["B4", "B5", "B10", "MTL_TXT"]

PRESETS: dict[str, list[str]] = {
    "🌡️ LST Essentials — B4, B5, B10 + MTL (recommended)": LST_ESSENTIALS,
    "Natural Color (B4, B3, B2)": ["B4", "B3", "B2", "MTL_TXT"],
    "Color Infrared / Vegetation (B5, B4, B3)": ["B5", "B4", "B3", "MTL_TXT"],
    "Agriculture & Moisture (B6, B5, B2)": ["B6", "B5", "B2", "MTL_TXT"],
    "All Surface Reflectance (B1–B7 + MTL)": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "MTL_TXT"],
    "Full Science Suite (B1–B7, B10, QA_PIXEL + MTL)": [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "QA_PIXEL", "MTL_TXT",
    ],
}

ProgressCB = Callable[[float, str], None]


# --------------------------------------------------------------------------------------
# AOI handling
# --------------------------------------------------------------------------------------
def load_aoi_from_upload(uploaded) -> gpd.GeoDataFrame:
    """Read a zipped shapefile / GeoJSON / KML upload into a GeoDataFrame."""
    import tempfile
    import zipfile

    name = uploaded.name
    ext = name.rsplit(".", 1)[-1].lower()
    tmp = tempfile.mkdtemp(prefix="stc_aoi_")
    path = os.path.join(tmp, name)
    with open(path, "wb") as fh:
        fh.write(uploaded.getbuffer())

    if ext == "zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        shp = next((os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f.lower().endswith(".shp")), None)
        if shp is None:
            raise ValueError("No .shp file found inside the zip. Zip must contain .shp, .shx, .dbf (and .prj).")
        gdf = gpd.read_file(shp)
    elif ext in ("geojson", "json", "kml"):
        gdf = gpd.read_file(path)
    else:
        raise ValueError("Unsupported format. Upload a zipped shapefile, .geojson or .kml.")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("The file contains no polygon features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)  # assume WGS84 when the .prj is missing
    return gdf.reset_index(drop=True)


def aoi_from_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> gpd.GeoDataFrame:
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Bounding box needs min < max for both longitude and latitude.")
    return gpd.GeoDataFrame({"name": ["Bounding-box AOI"]}, geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs=4326)


def aoi_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(4326)


def aoi_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = aoi_wgs84(gdf).total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def aoi_area_km2(gdf: gpd.GeoDataFrame) -> float:
    g = aoi_wgs84(gdf)
    utm = g.estimate_utm_crs()
    return float(g.to_crs(utm).geometry.area.sum() / 1e6)


def aoi_key(gdf: gpd.GeoDataFrame) -> str:
    """Short stable id for an AOI (used to name the clipped-data folder)."""
    b = ",".join(f"{v:.5f}" for v in aoi_bounds(gdf))
    return hashlib.md5(b.encode()).hexdigest()[:8]


# --------------------------------------------------------------------------------------
# STAC search
# --------------------------------------------------------------------------------------
def search_scenes(
    gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    max_cloud: float,
    platform: str = "all",
    limit: int = 30,
    timeout: int = 60,
) -> list[dict]:
    """Search Landsat 8/9 Collection-2 Level-2 scenes intersecting the AOI.

    Returns a list of plain dicts (newest first) incl. AOI coverage % for each footprint.
    """
    minx, miny, maxx, maxy = aoi_bounds(gdf)
    query: dict = {"eo:cloud_cover": {"lte": max_cloud}}
    if platform in ("landsat-8", "landsat-9"):
        query["platform"] = {"eq": platform}
    else:
        query["platform"] = {"in": ["landsat-8", "landsat-9"]}

    body = {
        "collections": [COLLECTION],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": query,
        "limit": int(limit),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    resp = requests.post(STAC_SEARCH_URL, json=body, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"Planetary Computer catalogue error {resp.status_code}: {resp.text[:300]}")

    aoi_union = aoi_wgs84(gdf).geometry.union_all() if hasattr(gdf.geometry, "union_all") else aoi_wgs84(gdf).geometry.unary_union
    aoi_area = aoi_union.area or 1e-12

    scenes: list[dict] = []
    for f in resp.json().get("features", []):
        p = f.get("properties", {})
        try:
            foot = shape(f["geometry"])
            coverage = 100.0 * foot.intersection(aoi_union).area / aoi_area
        except Exception:
            coverage = float("nan")
        if coverage == coverage and coverage <= 0:
            continue  # bbox overlapped but the real boundary does not

        assets = {k: {"href": v.get("href"), "size": v.get("file:size")} for k, v in f.get("assets", {}).items()}
        scenes.append(
            {
                "id": f["id"],
                "platform": p.get("platform", ""),
                "datetime": p.get("datetime", ""),
                "date": (p.get("datetime", "") or "")[:10],
                "cloud": round(float(p.get("eo:cloud_cover", 0.0)), 2),
                "path": p.get("landsat:wrs_path"),
                "row": p.get("landsat:wrs_row"),
                "sun_elevation": p.get("view:sun_elevation"),
                "coverage": None if coverage != coverage else round(coverage, 1),
                "bbox": f.get("bbox"),
                "geometry": f.get("geometry"),
                "preview": f.get("assets", {}).get("rendered_preview", {}).get("href"),
                "assets": assets,
            }
        )
    return scenes


# --------------------------------------------------------------------------------------
# Signing + download
# --------------------------------------------------------------------------------------
def sign_href(href: str, timeout: int = 30) -> str:
    """Planetary Computer SAS signing (no-op for non-PC URLs and local files)."""
    if not href or not (href.startswith("http") and ("planetarycomputer" in href or "blob.core.windows.net" in href)):
        return href
    r = requests.get(SAS_SIGN_URL, params={"href": href}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("href", href)


def authentic_filename(href: str) -> str:
    return os.path.basename(href.split("?")[0])


def _download_full(url: str, out_path: Path, cb: Optional[Callable[[float], None]] = None) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    if cb and total:
                        cb(done / total)
    tmp_path.replace(out_path)


def _aoi_window(src: rasterio.DatasetReader, gdf: gpd.GeoDataFrame, pad_px: int = 2) -> Window:
    """Pixel window of `src` covering the AOI (rounded outwards + small pad)."""
    minx, miny, maxx, maxy = gdf.to_crs(src.crs).total_bounds
    w = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    c0 = math.floor(w.col_off) - pad_px
    r0 = math.floor(w.row_off) - pad_px
    c1 = math.ceil(w.col_off + w.width) + pad_px
    r1 = math.ceil(w.row_off + w.height) + pad_px
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, src.width), min(r1, src.height)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("The study area does not overlap this scene's raster extent.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def _download_clipped(url: str, out_path: Path, gdf: gpd.GeoDataFrame) -> None:
    """Read only the AOI window of a remote COG and save it as a georeferenced GeoTIFF.
    Pixel values, dtype, nodata, CRS and the pixel grid are preserved exactly."""
    env = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_TIMEOUT="60",
    )
    with rasterio.Env(**env):
        with rasterio.open(url) as src:
            win = _aoi_window(src, gdf)
            data = src.read(1, window=win)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=int(win.height),
                width=int(win.width),
                transform=src.window_transform(win),
                count=1,
                compress="deflate",
                tiled=False,
                blockxsize=None,
                blockysize=None,
            )
            profile = {k: v for k, v in profile.items() if v is not None}
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(data, 1)
    tmp_path.replace(out_path)


def acquire_scene(
    scene: dict,
    band_ids: list[str],
    gdf: gpd.GeoDataFrame,
    base_dir: str | Path,
    mode: str = "clipped",  # "clipped" | "full"
    progress: Optional[ProgressCB] = None,
) -> dict:
    """Download the selected bands for one scene. Returns an 'acquisition' record:
       {scene_id, platform, date, cloud, mode, dir, files:{band_id: path}, aoi_name}
    Files that already exist are re-used (so re-running is instant)."""
    base_dir = Path(base_dir)
    sub = f"aoi_{aoi_key(gdf)}" if mode == "clipped" else "full_scene"
    out_dir = base_dir / scene["id"] / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep the AOI next to the data so the LST tab / a later session can re-use it
    aoi_path = out_dir / "aoi_boundary.geojson"
    aoi_wgs84(gdf).to_file(aoi_path, driver="GeoJSON")

    files: dict[str, str] = {}
    n = len(band_ids)
    for i, bid in enumerate(band_ids):
        band = BAND_BY_ID[bid]
        asset = scene["assets"].get(band.asset_key)
        if not asset or not asset.get("href"):
            raise RuntimeError(f"{band.name} is not available for scene {scene['id']}.")
        fname = authentic_filename(asset["href"])
        target = out_dir / fname

        def _p(frac: float, _i=i, _name=fname):
            if progress:
                progress((_i + frac) / n, f"Downloading {_i + 1}/{n}: {_name}")

        if target.exists() and target.stat().st_size > 0:
            _p(1.0)
            files[bid] = str(target)
            continue

        _p(0.0)
        url = sign_href(asset["href"])
        if band.category == "metadata" or mode == "full":
            _download_full(url, target, cb=_p)
        else:
            _download_clipped(url, target, gdf)
        _p(1.0)
        files[bid] = str(target)

    if progress:
        progress(1.0, "All bands ready.")
    return {
        "scene_id": scene["id"],
        "platform": scene.get("platform", ""),
        "date": scene.get("date", ""),
        "cloud": scene.get("cloud"),
        "mode": mode,
        "dir": str(out_dir),
        "files": files,
    }


# --------------------------------------------------------------------------------------
# Re-discover previously acquired scenes on disk
# --------------------------------------------------------------------------------------
def discover_acquisitions(base_dir: str | Path) -> list[dict]:
    """Find scene folders that already hold the 4 LST inputs (SR_B4, SR_B5, ST_B10, MTL)."""
    base = Path(base_dir)
    found: list[dict] = []
    if not base.exists():
        return found
    for d in sorted({p.parent for p in base.rglob("*_MTL.txt")}):
        def pick(pattern: str):
            m = sorted(d.glob(pattern))
            return str(m[0]) if m else None

        files = {
            "B4": pick("*_SR_B4.TIF"),
            "B5": pick("*_SR_B5.TIF"),
            "B10": pick("*_ST_B10.TIF"),
            "MTL_TXT": pick("*_MTL.txt"),
        }
        if all(files.values()):
            scene_id = Path(files["MTL_TXT"]).name.replace("_MTL.txt", "")
            found.append(
                {
                    "scene_id": scene_id,
                    "platform": "landsat-9" if scene_id.startswith("LC09") else "landsat-8",
                    "date": "",
                    "cloud": None,
                    "mode": "full" if d.name == "full_scene" else "clipped",
                    "dir": str(d),
                    "files": files,
                    "aoi_path": str(d / "aoi_boundary.geojson") if (d / "aoi_boundary.geojson").exists() else None,
                }
            )
    return found


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
