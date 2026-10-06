"""
lst_core.py — STC Universal LST Wizard
=======================================
The LST / UHI calculation engine, carried over unchanged from the validated standalone
"India LST & UHI Dashboard" (LST_Streamlit_app.py). The MTL parsing, Level-1 / Level-2
pipelines, UHI z-score and GeoTIFF export below are the original functions, word for word.

The only edit is in `process_uploaded_raster`, where locally acquired bands are opened in
place instead of being copied to a temp file first (I/O only — no change to the maths).
"""
import rasterio
from rasterio.mask import mask
import numpy as np
import zipfile
import os
import tempfile
import re


class LocalFile:
    """Duck-types a Streamlit UploadedFile for a file already on disk, so bands acquired in
    Tab 1 flow through the exact same functions as manually uploaded bands."""

    def __init__(self, path):
        self.path = str(path)
        self.name = os.path.basename(self.path)

    def getbuffer(self):
        with open(self.path, "rb") as fh:
            return fh.read()

    def getvalue(self):
        return self.getbuffer()


def extract_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".shp"):
                return os.path.join(root, file)
    return None

def parse_mtl_nested(uploaded_mtl_file):
    """Parses a Landsat Collection 2 MTL.txt file into a NESTED dict that
    respects the file's actual GROUP / END_GROUP structure.

    This matters because Level-2 MTL files embed a full copy of the original
    Level-1 processing record for traceability. That means keys like
    PROCESSING_LEVEL and REFLECTANCE_MULT_BAND_4 appear TWICE in the same
    file — once under the real Level-2 groups, once under the nested
    LEVEL1_PROCESSING_RECORD / LEVEL1_RADIOMETRIC_RESCALING groups — with
    genuinely different values. A flat parser silently lets the later
    duplicate overwrite the earlier one, which produces wrong constants.
    Preserving group structure lets us always pull each constant from the
    correct, unambiguous group."""
    content = uploaded_mtl_file.getvalue().decode("utf-8", errors="ignore")
    root = {}
    stack = [root]
    group_re = re.compile(r'^\s*GROUP\s*=\s*(\S+)\s*$')
    endgroup_re = re.compile(r'^\s*END_GROUP\s*=\s*(\S+)\s*$')
    kv_re = re.compile(r'^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$')

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line == "END":
            continue
        gm = group_re.match(line)
        if gm:
            new_group = {}
            stack[-1][gm.group(1)] = new_group
            stack.append(new_group)
            continue
        eg = endgroup_re.match(line)
        if eg:
            if len(stack) > 1:
                stack.pop()
            continue
        kvm = kv_re.match(line)
        if kvm:
            key, val = kvm.group(1), kvm.group(2).strip().strip('"')
            try:
                val = float(val)
            except ValueError:
                pass
            stack[-1][key] = val
    return root

def find_group(node, group_name):
    """Depth-first search for a group by name anywhere in the nested MTL tree."""
    if group_name in node and isinstance(node[group_name], dict):
        return node[group_name]
    for v in node.values():
        if isinstance(v, dict):
            found = find_group(v, group_name)
            if found is not None:
                return found
    return None

def build_scoped_mtl(root):
    """Builds two clean, unambiguous flat dicts — one for Level-1 math, one
    for Level-2 math — by pulling each constant from its correct group.
    This is the fix for the key-collision bug above."""
    image_attrs = find_group(root, "IMAGE_ATTRIBUTES") or {}
    product_contents = find_group(root, "PRODUCT_CONTENTS") or {}
    l1_rescaling = find_group(root, "LEVEL1_RADIOMETRIC_RESCALING") or {}
    l1_thermal = find_group(root, "LEVEL1_THERMAL_CONSTANTS") or {}
    l2_refl = find_group(root, "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS") or {}
    l2_temp = find_group(root, "LEVEL2_SURFACE_TEMPERATURE_PARAMETERS") or {}

    common = {
        "SPACECRAFT_ID": image_attrs.get("SPACECRAFT_ID"),
        "DATE_ACQUIRED": image_attrs.get("DATE_ACQUIRED"),
        "SCENE_CENTER_TIME": image_attrs.get("SCENE_CENTER_TIME"),
        "SUN_ELEVATION": image_attrs.get("SUN_ELEVATION"),
        # Authoritative processing level: PRODUCT_CONTENTS describes the
        # actual delivered product, not any embedded provenance record.
        "PROCESSING_LEVEL": product_contents.get("PROCESSING_LEVEL"),
    }

    mtl_l1 = dict(common)
    mtl_l1.update({
        "RADIANCE_MULT_BAND_10": l1_rescaling.get("RADIANCE_MULT_BAND_10"),
        "RADIANCE_ADD_BAND_10": l1_rescaling.get("RADIANCE_ADD_BAND_10"),
        "REFLECTANCE_MULT_BAND_4": l1_rescaling.get("REFLECTANCE_MULT_BAND_4"),
        "REFLECTANCE_ADD_BAND_4": l1_rescaling.get("REFLECTANCE_ADD_BAND_4"),
        "REFLECTANCE_MULT_BAND_5": l1_rescaling.get("REFLECTANCE_MULT_BAND_5"),
        "REFLECTANCE_ADD_BAND_5": l1_rescaling.get("REFLECTANCE_ADD_BAND_5"),
        "K1_CONSTANT_BAND_10": l1_thermal.get("K1_CONSTANT_BAND_10"),
        "K2_CONSTANT_BAND_10": l1_thermal.get("K2_CONSTANT_BAND_10"),
    })

    mtl_l2 = dict(common)
    mtl_l2.update({
        "REFLECTANCE_MULT_BAND_4": l2_refl.get("REFLECTANCE_MULT_BAND_4"),
        "REFLECTANCE_ADD_BAND_4": l2_refl.get("REFLECTANCE_ADD_BAND_4"),
        "REFLECTANCE_MULT_BAND_5": l2_refl.get("REFLECTANCE_MULT_BAND_5"),
        "REFLECTANCE_ADD_BAND_5": l2_refl.get("REFLECTANCE_ADD_BAND_5"),
        "TEMPERATURE_MULT_BAND_ST_B10": l2_temp.get("TEMPERATURE_MULT_BAND_ST_B10"),
        "TEMPERATURE_ADD_BAND_ST_B10": l2_temp.get("TEMPERATURE_ADD_BAND_ST_B10"),
    })

    return mtl_l1, mtl_l2, common

def detect_processing_level(common, filenames):
    """Determines Level-1 vs Level-2 from PRODUCT_CONTENTS.PROCESSING_LEVEL
    (the single authoritative field — never the embedded provenance copy),
    with filename patterns (SR_B*/ST_B* = Level-2) as a cross-check."""
    level_field = str(common.get("PROCESSING_LEVEL") or "").upper()
    if "L2" in level_field:
        mtl_says = "L2"
    elif "L1" in level_field:
        mtl_says = "L1"
    else:
        mtl_says = None

    joined_names = " ".join(filenames).upper()
    filenames_say_l2 = ("SR_B" in joined_names) or ("ST_B" in joined_names)

    detected = mtl_says if mtl_says is not None else ("L2" if filenames_say_l2 else "L1")
    return detected, mtl_says, filenames_say_l2

L1_REQUIRED_KEYS = [
    "RADIANCE_MULT_BAND_10", "RADIANCE_ADD_BAND_10",
    "K1_CONSTANT_BAND_10", "K2_CONSTANT_BAND_10",
    "REFLECTANCE_MULT_BAND_4", "REFLECTANCE_ADD_BAND_4",
    "REFLECTANCE_MULT_BAND_5", "REFLECTANCE_ADD_BAND_5",
    "SUN_ELEVATION",
]

L2_REQUIRED_KEYS = [
    "TEMPERATURE_MULT_BAND_ST_B10", "TEMPERATURE_ADD_BAND_ST_B10",
    "REFLECTANCE_MULT_BAND_4", "REFLECTANCE_ADD_BAND_4",
    "REFLECTANCE_MULT_BAND_5", "REFLECTANCE_ADD_BAND_5",
]

def validate_mtl(mtl_dict, level):
    required = L1_REQUIRED_KEYS if level == "L1" else L2_REQUIRED_KEYS
    missing = [k for k in required if mtl_dict.get(k) is None]
    return missing

def process_uploaded_raster(uploaded_file, geometries=None, return_meta=False, get_crs_only=False):
    # I/O ONLY CHANGE vs. the standalone tool: bands acquired in the Acquisition tab are already on
    # disk (LocalFile has a .path), so they are opened in place instead of being copied to a temp file.
    # Browser uploads (Streamlit UploadedFile) follow the original temp-file route unchanged.
    local_path = getattr(uploaded_file, "path", None)
    if local_path:
        tmp_path, is_temp = str(local_path), False
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        is_temp = True
    try:
        with rasterio.open(tmp_path) as src:
            if get_crs_only:
                return src.crs
            out_image, out_transform = mask(src, geometries, crop=True)
            array = out_image[0].astype('float32')
            array[array == 0] = np.nan  # Mask background clipping artifacts
            if return_meta:
                return array, out_transform, src.crs
            return array
    finally:
        if is_temp:
            os.remove(tmp_path)

def calculate_lst_level1(red_dn, nir_dn, thermal_dn, mtl, ndvi_min_override=None, ndvi_max_override=None):
    """LEVEL-1 (raw TOA DN) pipeline: radiance -> brightness temperature ->
    NDVI-based emissivity -> Planck-corrected LST. Uses only constants read
    from THIS scene's MTL file."""
    np.seterr(divide='ignore', invalid='ignore')

    # 1. TOA Spectral Radiance (thermal band)
    ML = mtl["RADIANCE_MULT_BAND_10"]
    AL = mtl["RADIANCE_ADD_BAND_10"]
    radiance = (thermal_dn * ML) + AL

    # 2. At-Satellite Brightness Temperature
    K1 = mtl["K1_CONSTANT_BAND_10"]
    K2 = mtl["K2_CONSTANT_BAND_10"]
    bt_celsius = (K2 / np.log((K1 / radiance) + 1)) - 273.15

    # 3. TOA Reflectance (Red & NIR) — requires sun-angle correction for Level-1
    MR4 = mtl["REFLECTANCE_MULT_BAND_4"]
    AR4 = mtl["REFLECTANCE_ADD_BAND_4"]
    MR5 = mtl["REFLECTANCE_MULT_BAND_5"]
    AR5 = mtl["REFLECTANCE_ADD_BAND_5"]
    sun_elev_rad = np.radians(mtl["SUN_ELEVATION"])

    red_refl = ((red_dn * MR4) + AR4) / np.sin(sun_elev_rad)
    nir_refl = ((nir_dn * MR5) + AR5) / np.sin(sun_elev_rad)

    # 4. NDVI from calibrated TOA reflectance
    ndvi = (nir_refl - red_refl) / (nir_refl + red_refl)

    # 5. NDVI min/max derived from THIS scene, not assumed
    auto_derived = False
    if ndvi_min_override is not None and ndvi_max_override is not None:
        ndvi_min, ndvi_max = ndvi_min_override, ndvi_max_override
    else:
        valid_ndvi = ndvi[np.isfinite(ndvi)]
        ndvi_min = float(np.nanpercentile(valid_ndvi, 2))
        ndvi_max = float(np.nanpercentile(valid_ndvi, 98))
        auto_derived = True
        if (ndvi_max - ndvi_min) < 0.05:
            ndvi_min, ndvi_max = 0.2, 0.5

    # 6. Proportion of Vegetation
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    pv = np.clip(pv, 0, 1)

    # 7. Land Surface Emissivity
    emissivity = 0.004 * pv + 0.986

    # 8. Final LST (Planck-based correction)
    lam = 10.895
    rho = 14388
    lst = bt_celsius / (1 + ((lam * bt_celsius / rho) * np.log(emissivity)))
    lst[(lst < -10) | (lst > 65)] = np.nan

    return {
        "lst": lst,
        "ndvi": ndvi,
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "ndvi_auto_derived": auto_derived,
        "level": "L1",
    }

def calculate_lst_level2(red_dn, nir_dn, thermal_dn, mtl):
    """LEVEL-2 (Science Product) pipeline. ST_B10 is already USGS's fully
    atmospherically- and emissivity-corrected surface temperature — it is
    converted straight to Celsius via its own scale factors, with NO Planck
    re-derivation and NO NDVI-based emissivity step (that correction is
    already baked into the product). SR_B4/SR_B5 are already surface
    reflectance, so NDVI uses them directly with no sun-angle correction
    (that only applies to raw TOA quantities, not atmospherically-corrected
    surface reflectance)."""
    np.seterr(divide='ignore', invalid='ignore')

    # 1. Surface Temperature directly from ST_B10 scaling
    TM = mtl["TEMPERATURE_MULT_BAND_ST_B10"]
    TA = mtl["TEMPERATURE_ADD_BAND_ST_B10"]
    lst_kelvin = (thermal_dn * TM) + TA
    lst = lst_kelvin - 273.15
    lst[(lst < -60) | (lst > 70)] = np.nan  # physically plausible LST bounds

    # 2. Surface Reflectance directly from SR_B4/SR_B5 scaling (no sun-angle division)
    MR4 = mtl["REFLECTANCE_MULT_BAND_4"]
    AR4 = mtl["REFLECTANCE_ADD_BAND_4"]
    MR5 = mtl["REFLECTANCE_MULT_BAND_5"]
    AR5 = mtl["REFLECTANCE_ADD_BAND_5"]

    red_refl = (red_dn * MR4) + AR4
    nir_refl = (nir_dn * MR5) + AR5
    red_refl[(red_refl < 0) | (red_refl > 1)] = np.nan
    nir_refl[(nir_refl < 0) | (nir_refl > 1)] = np.nan

    # 3. NDVI
    ndvi = (nir_refl - red_refl) / (nir_refl + red_refl)

    return {
        "lst": lst,
        "ndvi": ndvi,
        "ndvi_min": None,
        "ndvi_max": None,
        "ndvi_auto_derived": False,
        "level": "L2",
    }

def compute_uhi_standardized(lst_array):
    """Calculates the UHI Index using standard deviation normalization (Z-score)."""
    mean_lst = np.nanmean(lst_array)
    std_lst = np.nanstd(lst_array)
    uhi_index = (lst_array - mean_lst) / std_lst
    return uhi_index, mean_lst, std_lst

def create_download_tiff(array, transform, crs):
    profile = {
        'driver': 'GTiff',
        'height': array.shape[0],
        'width': array.shape[1],
        'count': 1,
        'dtype': str(array.dtype),
        'crs': crs,
        'transform': transform,
        'nodata': np.nan
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, 1)
        return memfile.read()
