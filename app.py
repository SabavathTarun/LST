"""
STC Universal LST Wizard
========================
One web tool, two tabs:
  1. Satellite Image Acquisition  — find a Landsat 8/9 scene and pull only the bands LST needs
  2. LST & UHI Analysis           — the validated STC Land Surface Temperature / Urban Heat Island engine

Run:  streamlit run app.py
"""
import base64
import datetime as dt
import io
import os
import zipfile
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from streamlit_folium import st_folium

import acquisition as acq
from lst_core import (
    LocalFile,
    build_scoped_mtl,
    calculate_lst_level1,
    calculate_lst_level2,
    compute_uhi_standardized,
    create_download_tiff,
    detect_processing_level,
    extract_zip,
    parse_mtl_nested,
    process_uploaded_raster,
    validate_mtl,
)

# st.image / st.dataframe: "stretch" width on new Streamlit, use_container_width on older ones
_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

APP_DIR = Path(__file__).parent
ASSETS = APP_DIR / "assets"
DEFAULT_DATA_DIR = APP_DIR / "STC_LST_Wizard_Data"

# STC brand palette (sampled from the logo)
NAVY, NAVY_2, GOLD, GOLD_HI, GOLD_LO, CREAM = "#0d1829", "#14233a", "#d6ba70", "#e4d280", "#a38139", "#efe6cc"

st.set_page_config(
    page_title="STC Universal LST Wizard",
    page_icon=str(ASSETS / "stc_icon.png"),
    layout="wide",
)


# ======================================================================================
# Branding
# ======================================================================================
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def inject_css() -> None:
    st.markdown(
        f"""
<style>
.stApp {{
  background: radial-gradient(1100px 520px at 88% -8%, rgba(214,186,112,.10), transparent 60%),
              radial-gradient(900px 500px at -5% 105%, rgba(163,129,57,.08), transparent 55%), {NAVY};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}

.stc-hero {{ display:flex; align-items:center; gap:28px; padding:18px 26px; margin-bottom:10px;
  background: linear-gradient(135deg, {NAVY_2}, #101d33); border:1px solid rgba(214,186,112,.28);
  border-radius:18px; box-shadow: 0 8px 30px rgba(0,0,0,.35); }}
.stc-hero img {{ height:112px; width:auto; }}
.stc-hero h1 {{ margin:0; font-size:2.15rem; line-height:1.1; letter-spacing:.3px;
  background: linear-gradient(90deg, {GOLD_HI}, {GOLD} 55%, {GOLD_LO}); -webkit-background-clip:text;
  background-clip:text; color:transparent; font-weight:800; }}
.stc-hero p {{ margin:.45rem 0 0; color:#c9c1a8; font-size:.98rem; max-width:760px; }}
.stc-pill {{ display:inline-block; margin-top:.7rem; margin-right:.4rem; padding:3px 11px; font-size:.74rem;
  color:{GOLD}; border:1px solid rgba(214,186,112,.45); border-radius:999px; letter-spacing:.4px; }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid rgba(214,186,112,.25); }}
.stTabs [data-baseweb="tab"] {{ padding:10px 20px; font-weight:600; color:#b9b199; border-radius:10px 10px 0 0; }}
.stTabs [aria-selected="true"] {{ color:{GOLD_HI} !important; background:rgba(214,186,112,.08); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:{GOLD} !important; height:3px; }}

.stc-step {{ display:flex; align-items:center; gap:12px; margin:1.3rem 0 .4rem; }}
.stc-step .n {{ width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,{GOLD_HI},{GOLD_LO}); color:{NAVY}; font-weight:800; font-size:.9rem; flex:none; }}
.stc-step .t {{ font-size:1.12rem; font-weight:700; color:{CREAM}; }}
.stc-step .s {{ font-size:.82rem; color:#a79f87; margin-left:2px; }}

button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg,{GOLD_HI},#c9a85c) !important; color:{NAVY} !important;
  border:none !important; font-weight:700 !important; box-shadow:0 3px 14px rgba(214,186,112,.25); }}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ filter:brightness(1.07); }}
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {{
  border:1px solid rgba(214,186,112,.55) !important; color:{GOLD_HI} !important; background:transparent !important; }}

[data-testid="stMetric"] {{ background:{NAVY_2}; border:1px solid rgba(214,186,112,.22); border-radius:12px; padding:12px 16px; }}
[data-testid="stMetricLabel"] p {{ color:#b9b199; }}
[data-testid="stExpander"] {{ border:1px solid rgba(214,186,112,.22); border-radius:12px; background:rgba(20,35,58,.55); }}
[data-testid="stFileUploader"] section {{ background:{NAVY_2}; border:1px dashed rgba(214,186,112,.45); }}
.stc-callout {{ border-left:4px solid {GOLD}; background:rgba(214,186,112,.08); padding:12px 16px; border-radius:0 10px 10px 0; margin:.6rem 0; }}
.stc-footer {{ text-align:center; color:#8d8670; font-size:.8rem; margin:2.2rem 0 .6rem;
  padding-top:1rem; border-top:1px solid rgba(214,186,112,.18); }}
[data-testid="stSidebar"] {{ background:#0a1322; border-right:1px solid rgba(214,186,112,.2); }}
</style>
""",
        unsafe_allow_html=True,
    )


def header() -> None:
    logo = _b64(ASSETS / "stc_logo.png")
    st.markdown(
        f"""
<div class="stc-hero">
  <img src="data:image/png;base64,{logo}" alt="STC Consultancy"/>
  <div>
    <h1>STC Universal LST Wizard</h1>
    <p>From satellite scene to Land Surface Temperature and Urban Heat Island maps in one place —
       acquire Landsat 8/9 bands, then analyse them without leaving the tool.</p>
    <span class="stc-pill">LANDSAT 8 · 9</span><span class="stc-pill">LST</span>
    <span class="stc-pill">UHI Z-SCORE</span><span class="stc-pill">GIS-READY GeoTIFF</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def step(n: int, title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="stc-step"><div class="n">{n}</div><div><span class="t">{title}</span>'
        f'<div class="s">{sub}</div></div></div>',
        unsafe_allow_html=True,
    )


inject_css()
header()

with st.sidebar:
    st.image(str(ASSETS / "stc_logo.png"), **STRETCH)
    st.markdown("### Workflow")
    st.markdown(
        "**1 · Acquire** — draw your study area, pick a clear Landsat scene, download B4, B5, B10 + MTL.\n\n"
        "**2 · Analyse** — the same bands feed the LST / UHI engine automatically."
    )
    with st.expander("⚙️ Data folder"):
        data_dir_str = st.text_input(
            "Where acquired bands are saved",
            value=st.session_state.get("data_dir", str(DEFAULT_DATA_DIR)),
            help="Files already downloaded here are re-used, so repeating a scene is instant.",
        )
        st.session_state["data_dir"] = data_dir_str
    st.caption("Data: USGS Landsat Collection 2 Level-2 via Microsoft Planetary Computer.")

DATA_DIR = Path(st.session_state.get("data_dir", str(DEFAULT_DATA_DIR)))


# ======================================================================================
# TAB 1 — SATELLITE IMAGE ACQUISITION
# ======================================================================================
def _scene_label(s: dict) -> str:
    cov = "?" if s["coverage"] is None else f'{s["coverage"]:.0f}%'
    return f'{s["date"]} · {s["platform"].replace("landsat-", "Landsat ")} · cloud {s["cloud"]:.1f}% · AOI cover {cov} · P{s["path"]}/R{s["row"]}'


def _footprint_map(aoi: gpd.GeoDataFrame, scene: dict | None) -> folium.Map:
    minx, miny, maxx, maxy = acq.aoi_bounds(aoi)
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)
    if scene and scene.get("geometry"):
        folium.GeoJson(
            scene["geometry"], name="Scene footprint",
            style_function=lambda _: {"color": "#5fd3ff", "weight": 2, "dashArray": "6 6", "fillOpacity": 0.04},
        ).add_to(m)
    folium.GeoJson(
        acq.aoi_wgs84(aoi).__geo_interface__, name="Study area",
        style_function=lambda _: {"color": GOLD, "weight": 3, "fillColor": GOLD, "fillOpacity": 0.18},
    ).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(30, 30))
    folium.LayerControl(collapsed=True).add_to(m)
    return m


@st.fragment
def acquisition_tab() -> None:
    st.markdown(
        '<div class="stc-callout">Pick your study area and a scene — the wizard downloads <b>only the pixels inside your '
        "boundary</b> for Band 4, Band 5, Band 10 and the MTL file, then hands them straight to the LST tab. "
        "No USGS EarthExplorer login, no manual re-upload.</div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- 1. study area
    step(1, "Study area", "Boundary used to search scenes, clip the bands and later mask the LST map")
    mode = st.radio(
        "Define study area by",
        ["Upload boundary", "Enter bounding box"],
        horizontal=True,
        label_visibility="collapsed",
        key="aoi_mode",
    )
    if mode == "Upload boundary":
        up = st.file_uploader(
            "Zipped shapefile (.zip), GeoJSON or KML",
            type=["zip", "geojson", "json", "kml"],
            key="aoi_upload",
        )
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get("aoi_sig") != sig:
                try:
                    gdf = acq.load_aoi_from_upload(up)
                    st.session_state["aoi_gdf"], st.session_state["aoi_sig"] = gdf, sig
                    st.session_state["aoi_name"] = up.name
                    st.session_state.pop("scenes", None)
                except Exception as e:
                    st.session_state.pop("aoi_gdf", None)
                    st.error(f"Could not read the boundary: {e}")
    else:
        c = st.columns(4)
        lon0 = c[0].number_input("Min longitude", value=None, min_value=-180.0, max_value=180.0, format="%.5f", key="bb_lon0")
        lat0 = c[1].number_input("Min latitude", value=None, min_value=-90.0, max_value=90.0, format="%.5f", key="bb_lat0")
        lon1 = c[2].number_input("Max longitude", value=None, min_value=-180.0, max_value=180.0, format="%.5f", key="bb_lon1")
        lat1 = c[3].number_input("Max latitude", value=None, min_value=-90.0, max_value=90.0, format="%.5f", key="bb_lat1")
        if None not in (lon0, lat0, lon1, lat1):
            try:
                st.session_state["aoi_gdf"] = acq.aoi_from_bbox(lon0, lat0, lon1, lat1)
                st.session_state["aoi_name"] = "Bounding box"
                st.session_state["aoi_sig"] = ("bbox", lon0, lat0, lon1, lat1)
            except ValueError as e:
                st.session_state.pop("aoi_gdf", None)
                st.error(str(e))

    aoi = st.session_state.get("aoi_gdf")
    if aoi is None:
        st.info("Upload a boundary or enter a bounding box to continue.")
        return
    area = acq.aoi_area_km2(aoi)
    m1, m2, m3 = st.columns(3)
    m1.metric("Study area", st.session_state.get("aoi_name", "AOI"))
    m2.metric("Area", f"{area:,.1f} km²")
    m3.metric("Features", f"{len(aoi)}")
    if area > 5000:
        st.warning("This is a large study area — a single Landsat tile is ~185 × 180 km. Consider a smaller boundary.")

    # ---------------------------------------------------------------- 2. search
    step(2, "Find a scene", "Landsat 8 / 9 Collection 2 Level-2 (surface reflectance + surface temperature)")
    f1, f2, f3 = st.columns([1.1, 1.4, 1.2])
    platform_label = f1.radio("Satellite", ["Landsat 8 + 9", "Landsat 9", "Landsat 8"], key="f_platform")
    platform = {"Landsat 8 + 9": "all", "Landsat 9": "landsat-9", "Landsat 8": "landsat-8"}[platform_label]
    today = dt.date.today()
    rng = f2.date_input("Acquisition date range", value=(today - dt.timedelta(days=90), today), max_value=today, key="f_dates")
    cloud = f3.slider("Max cloud cover (%)", 0, 100, 20, key="f_cloud")
    limit = f3.number_input("Max scenes to list", 5, 100, 30, step=5, key="f_limit")

    if st.button("🔍 Search scenes", type="primary", key="btn_search"):
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            st.error("Pick both a start and an end date.")
        else:
            with st.spinner("Querying the Landsat catalogue…"):
                try:
                    found = acq.search_scenes(aoi, rng[0].isoformat(), rng[1].isoformat(), cloud, platform, int(limit))
                    st.session_state["scenes"] = found
                    st.session_state.pop("acq_done", None)
                except Exception as e:
                    st.session_state["scenes"] = []
                    st.error(f"Search failed: {e}")

    scenes = st.session_state.get("scenes")
    if scenes is None:
        return
    if not scenes:
        st.warning("No scenes matched. Try a wider date range or a higher cloud-cover limit.")
        return

    # ---------------------------------------------------------------- 3. pick scene
    step(3, "Choose the scene", f"{len(scenes)} scenes found")
    sort_by = st.radio("Sort by", ["Newest first", "Lowest cloud", "Best AOI coverage"], horizontal=True, key="f_sort")
    ordered = list(scenes)
    if sort_by == "Lowest cloud":
        ordered.sort(key=lambda s: (s["cloud"], -(s["coverage"] or 0)))
    elif sort_by == "Best AOI coverage":
        ordered.sort(key=lambda s: (-(s["coverage"] or 0), s["cloud"]))

    df = pd.DataFrame(
        [
            {
                "Date": s["date"],
                "Satellite": s["platform"].replace("landsat-", "Landsat "),
                "Cloud %": s["cloud"],
                "AOI coverage %": s["coverage"],
                "Path/Row": f'{s["path"]}/{s["row"]}',
                "Sun elev. °": s["sun_elevation"],
                "Scene ID": s["id"],
            }
            for s in ordered
        ]
    )
    st.dataframe(df, hide_index=True, **STRETCH, height=min(300, 38 + 35 * len(df)))

    idx = st.selectbox("Scene to acquire", range(len(ordered)), format_func=lambda i: _scene_label(ordered[i]), key="f_scene")
    scene = ordered[idx]

    left, right = st.columns([1.5, 1])
    with left:
        st_folium(_footprint_map(aoi, scene), height=360, use_container_width=True, returned_objects=[], key="fp_map")
    with right:
        if scene.get("preview"):
            st.image(scene["preview"], caption=f'Preview · {scene["id"]}', **STRETCH)
        if scene["coverage"] is not None and scene["coverage"] < 99:
            st.warning(
                f'This scene covers only **{scene["coverage"]:.0f}%** of your study area. '
                "Pick another scene (or another path/row) for full coverage."
            )
        if scene["cloud"] > 10:
            st.caption("Cloud % is for the whole scene — check the preview over your own study area.")

    # ---------------------------------------------------------------- 4. bands
    step(4, "Bands to download", "LST needs Band 4 (Red), Band 5 (NIR), Band 10 (Thermal) and the MTL file")
    preset = st.selectbox("Preset", list(acq.PRESETS.keys()), key="f_preset")
    chosen = st.multiselect(
        "Bands / files",
        [b.id for b in acq.BANDS],
        default=acq.PRESETS[preset],
        format_func=lambda i: f"{i} — {acq.BAND_BY_ID[i].name}",
        key=f"f_bands_{preset}",
    )
    missing_lst = [b for b in acq.LST_ESSENTIALS if b not in chosen]
    if missing_lst:
        st.info(f"The LST tab will not be able to use this download — it also needs: {', '.join(missing_lst)}.")

    dl_mode = st.radio(
        "Download extent",
        ["Clip to my study area (fast, small files — recommended)", "Full scene (original USGS tile, large)"],
        key="f_dlmode",
    )
    mode_key = "clipped" if dl_mode.startswith("Clip") else "full"

    if st.button("⬇️ Acquire bands", type="primary", disabled=not chosen, key="btn_acquire"):
        order = [b.id for b in acq.BANDS if b.id in chosen]
        bar = st.progress(0.0, text="Starting…")

        def _cb(frac: float, text: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=text)

        try:
            rec = acq.acquire_scene(scene, order, aoi, DATA_DIR, mode=mode_key, progress=_cb)
            rec["aoi_path"] = str(Path(rec["dir"]) / "aoi_boundary.geojson")
            st.session_state["acq"] = rec
            st.session_state["acq_done"] = rec
            st.session_state.pop("lst_out", None)
        except Exception as e:
            bar.empty()
            st.error(f"Download failed: {e}")
        else:
            st.rerun()  # full-app rerun so the LST tab sees the new scene immediately

    done = st.session_state.get("acq_done")
    if done:
        st.success(f"✅ Scene **{done['scene_id']}** acquired — {len(done['files'])} files ready.")
        rows = [
            {"Band": bid, "File": Path(p).name, "Size": acq.human_size(Path(p).stat().st_size)}
            for bid, p in done["files"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
        st.markdown(
            '<div class="stc-callout"><b>Next →</b> open the <b>“2 · LST &amp; UHI Analysis”</b> tab. '
            "This scene and your boundary are already loaded.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Save a copy / keep the files"):
            st.caption(f"Files are saved in: `{done['dir']}`")
            if st.button("📦 Prepare ZIP", key="btn_zip"):
                zpath = Path(done["dir"]) / f"{done['scene_id']}_bands.zip"
                with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in done["files"].values():
                        zf.write(p, arcname=Path(p).name)
                    zf.write(done["aoi_path"], arcname="aoi_boundary.geojson")
                st.session_state["zip_path"] = str(zpath)
            zp = st.session_state.get("zip_path")
            if zp and Path(zp).exists() and Path(zp).name.startswith(done["scene_id"]):
                with open(zp, "rb") as fh:
                    st.download_button("⬇️ Download ZIP", fh, file_name=Path(zp).name, mime="application/zip", key="dl_zip")


# ======================================================================================
# TAB 2 — LST & UHI ANALYSIS
# ======================================================================================
def run_lst_analysis(red_file, nir_file, thermal_file, mtl_file, gdf, satellite_choice,
                     manual_override, manual_ndvi_min, manual_ndvi_max) -> dict:
    """The original dashboard pipeline, unchanged in logic. Returns everything the results view needs."""
    out = {"messages": [], "error": None}
    try:
        # 1. Parse MTL (group-aware) and detect processing level
        mtl_root = parse_mtl_nested(mtl_file)
        mtl_l1, mtl_l2, mtl_common = build_scoped_mtl(mtl_root)
        filenames = [red_file.name, nir_file.name, thermal_file.name]
        level, mtl_says, filenames_say_l2 = detect_processing_level(mtl_common, filenames)
        mtl = mtl_l1 if level == "L1" else mtl_l2

        if mtl_says is not None and (
            (mtl_says == "L1" and filenames_say_l2) or (mtl_says == "L2" and not filenames_say_l2)
        ):
            out["messages"].append((
                "warning",
                f"⚠️ The MTL file's PROCESSING_LEVEL says **{mtl_says}**, but your band "
                f"filenames look like the other level ({', '.join(filenames)}). Proceeding "
                f"with **{level}** processing based on the MTL field — double-check you "
                "uploaded matching bands and metadata for the same download.",
            ))

        missing_keys = validate_mtl(mtl, level)
        if missing_keys:
            out["error"] = (
                f"Detected **{level}** processing, but the uploaded MTL file is missing "
                f"required fields for it: {', '.join(missing_keys)}. Please upload the "
                "original, unmodified *_MTL.txt file that came with this exact band download."
            )
            return out

        spacecraft_id = str(mtl.get("SPACECRAFT_ID", "")).upper()
        expected_id = "LANDSAT_8" if satellite_choice.startswith("Landsat 8") else "LANDSAT_9"
        if expected_id not in spacecraft_id:
            out["messages"].append((
                "warning",
                f"⚠️ You selected **{satellite_choice}**, but the MTL file reports "
                f"**SPACECRAFT_ID = {mtl.get('SPACECRAFT_ID', 'unknown')}**. "
                "The calculation will still proceed using the constants actually found "
                "in the MTL file (which are always scene-accurate), but double-check "
                "you uploaded the matching bands and metadata for the same scene.",
            ))

        # 3. CRS Alignment & Clipping
        raster_crs = process_uploaded_raster(red_file, get_crs_only=True)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        geometries = gdf.geometry.values

        red_band, out_transform, out_crs = process_uploaded_raster(red_file, geometries, return_meta=True)
        nir_band = process_uploaded_raster(nir_file, geometries)
        thermal_band = process_uploaded_raster(thermal_file, geometries)

        # 4. Mathematical Calculations — branch by detected processing level
        if level == "L1":
            result = calculate_lst_level1(
                red_band, nir_band, thermal_band, mtl,
                ndvi_min_override=manual_ndvi_min,
                ndvi_max_override=manual_ndvi_max,
            )
        else:
            result = calculate_lst_level2(red_band, nir_band, thermal_band, mtl)
            if manual_override:
                out["messages"].append((
                    "info",
                    "ℹ️ NDVI min/max override is ignored for Level-2 data — the ST band "
                    "already includes USGS's own emissivity correction, so there's no "
                    "Pv/emissivity step here to override.",
                ))

        lst_map = result["lst"]
        ndvi_map = result["ndvi"]
        uhi_map, mean_lst, std_lst = compute_uhi_standardized(lst_map)

        valid_pixel_frac = np.isfinite(lst_map).sum() / lst_map.size
        if valid_pixel_frac < 0.02:
            out["messages"].append((
                "error",
                "⚠️ Fewer than 2% of pixels produced a valid LST value. This almost always "
                "means the uploaded bands don't match the detected processing level, the "
                "boundary shapefile doesn't overlap the scene, or a band was uploaded in "
                "the wrong slot. Double-check your uploads before trusting the results below.",
            ))

        # ---- figures (same maps / colour ramps as the standalone tool, on the STC navy canvas)
        theme = {
            "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
            "text.color": CREAM, "axes.labelcolor": CREAM, "axes.titlecolor": GOLD_HI,
            "xtick.color": CREAM, "ytick.color": CREAM, "axes.edgecolor": GOLD_LO,
        }
        with plt.rc_context(theme):
            fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
            lst_cmap = LinearSegmentedColormap.from_list("LST_Ramp", ["#228B22", "#FFFF00", "#FFA500", "#FF0000"])
            im1 = ax1[0].imshow(lst_map, cmap=lst_cmap, vmin=np.nanpercentile(lst_map, 2), vmax=np.nanpercentile(lst_map, 98))
            ax1[0].set_title("Land Surface Temperature (°C)")
            fig1.colorbar(im1, ax=ax1[0], label="Temp (°C)", fraction=0.046, pad=0.04)
            ax1[0].axis('off')
            im2 = ax1[1].imshow(ndvi_map, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            ax1[1].set_title("Vegetation Density Index (NDVI)")
            fig1.colorbar(im2, ax=ax1[1], label="NDVI Index", fraction=0.046, pad=0.04)
            ax1[1].axis('off')

            fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
            im3 = ax2[0].imshow(uhi_map, cmap='RdYlBu_r', vmin=-3, vmax=3)
            ax2[0].set_title("UHI Index (Standard Deviations from Mean)")
            fig2.colorbar(im3, ax=ax2[0], label="UHI Score (Z-Value)", fraction=0.046, pad=0.04)
            ax2[0].axis('off')

            action_map = np.zeros_like(uhi_map)
            action_map[np.isnan(uhi_map)] = np.nan
            action_map[(uhi_map >= 1.0) & (uhi_map < 2.0)] = 1
            action_map[uhi_map >= 2.0] = 2
            cmap_action = plt.matplotlib.colors.ListedColormap(['#e0e0e0', '#ff9999', '#cc0000'])
            ax2[1].imshow(action_map, cmap=cmap_action)
            ax2[1].set_title("Vulnerability Zoning (Based on σ)")
            ax2[1].axis('off')
            labels = ['Average/Cool (UHI < 1)', 'Moderate Heat (UHI 1 to 2)', 'Severe Heat (UHI > 2)']
            colors = ['#e0e0e0', '#ff9999', '#cc0000']
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
            ax2[1].legend(handles=patches, loc='lower right', fontsize='small', facecolor=NAVY_2, labelcolor=CREAM, edgecolor=GOLD_LO)

        pixel_sqkm = (30 * 30) / 1_000_000
        out.update(
            level=level, mtl=mtl, result=result, mean_lst=mean_lst, std_lst=std_lst,
            max_lst=float(np.nanmax(lst_map)), max_uhi=float(np.nanmax(uhi_map)),
            total_sqkm=float(np.sum(~np.isnan(lst_map)) * pixel_sqkm),
            moderate_uhi_sqkm=float(np.sum((uhi_map >= 1.0) & (uhi_map < 2.0)) * pixel_sqkm),
            severe_uhi_sqkm=float(np.sum(uhi_map >= 2.0) * pixel_sqkm),
            fig1=fig1, fig2=fig2,
            lst_tiff=create_download_tiff(lst_map, out_transform, out_crs),
            uhi_tiff=create_download_tiff(uhi_map, out_transform, out_crs),
            manual_override=manual_override,
        )
        return out
    except Exception as e:  # same behaviour as the standalone tool
        out["error"] = f"An error occurred: {e}"
        return out


def render_lst_results(r: dict) -> None:
    for kind, msg in r["messages"]:
        getattr(st, kind)(msg)
    if r["error"]:
        st.error(r["error"])
        return

    level, mtl, result = r["level"], r["mtl"], r["result"]
    mean_lst, std_lst = r["mean_lst"], r["std_lst"]

    st.success("Spatial Processing Complete!")
    with st.expander("📄 Scene metadata used for this calculation", expanded=False):
        level_label = "Level-1 (TOA radiance-based)" if level == "L1" else "Level-2 (Science Product: SR + ST)"
        st.write(f"**Detected processing level:** {level_label}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Satellite", str(mtl.get("SPACECRAFT_ID", "N/A")))
        m2.metric("Date Acquired", str(mtl.get("DATE_ACQUIRED", "N/A")))
        m3.metric("Scene Center Time", str(mtl.get("SCENE_CENTER_TIME", "N/A")).strip('"'))
        m4.metric("Sun Elevation", f'{mtl.get("SUN_ELEVATION", "N/A")}°')

        if level == "L1":
            st.write(
                f"**Thermal constants (Band 10):** RADIANCE_MULT = {mtl['RADIANCE_MULT_BAND_10']}, "
                f"RADIANCE_ADD = {mtl['RADIANCE_ADD_BAND_10']}, K1 = {mtl['K1_CONSTANT_BAND_10']}, "
                f"K2 = {mtl['K2_CONSTANT_BAND_10']}"
            )
            ndvi_source = "auto-derived from this scene (2nd–98th percentile)" if result["ndvi_auto_derived"] else "manually overridden by user"
            st.write(f"**NDVI min/max used for emissivity ({ndvi_source}):** {result['ndvi_min']:.3f} to {result['ndvi_max']:.3f}")
        else:
            st.write(
                f"**Surface temperature scaling (ST_B10):** TEMPERATURE_MULT = "
                f"{mtl['TEMPERATURE_MULT_BAND_ST_B10']}, TEMPERATURE_ADD = "
                f"{mtl['TEMPERATURE_ADD_BAND_ST_B10']}"
            )
            st.write(
                "LST comes directly from the ST_B10 product — USGS has already applied "
                "atmospheric and emissivity correction upstream, so no further Planck-based "
                "correction is applied here."
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study Area Mean (μ)", f"{mean_lst:.2f} °C")
    c2.metric("Standard Deviation (σ)", f"{std_lst:.2f} °C")
    c3.metric("Max Urban Temp", f"{r['max_lst']:.2f} °C")
    c4.metric("Max UHI Index Score", f"+{r['max_uhi']:.2f} σ", delta_color="inverse")

    st.subheader("🗺️ Base Layers: LST & Vegetation")
    st.pyplot(r["fig1"])

    st.markdown("---")
    st.header("📊 Standardized Urban Heat Island Analysis")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total Analyzed Area", f"{r['total_sqkm']:.2f} km²")
    sc2.metric("Moderate UHI Area (1σ to 2σ)", f"{r['moderate_uhi_sqkm']:.2f} km²")
    sc3.metric("Severe UHI Crisis Area (> 2σ)", f"{r['severe_uhi_sqkm']:.2f} km²", "Requires Intervention")
    st.pyplot(r["fig2"])

    st.info(f"""
            **📋 Planner's Standardization Insight:**
            * The UHI Index is calculated as a **Z-Score** ($UHI = \\frac{{LST - \\mu}}{{\\sigma}}$).
            * A score of `0` represents the exact average temperature of the study area ({mean_lst:.2f}°C).
            * Red zones on the vulnerability map indicate areas that are **more than 2 standard deviations hotter** than the rest of the city.
            * This scene was processed as **{"Level-1 (TOA radiance-based)" if level == "L1" else "Level-2 (Science Product)"}** using constants read from its own MTL file — calibrated specifically for {mtl.get('SPACECRAFT_ID', 'this satellite')} on {mtl.get('DATE_ACQUIRED', 'this date')}.
            """)

    st.markdown("---")
    st.subheader("💾 Export GIS Layers")
    st.write("Download the maps as georeferenced .TIF files for QGIS/ArcGIS.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("⬇️ Download Absolute LST Map (.tif)", data=r["lst_tiff"], file_name="LST_Map.tif", mime="image/tiff", key="dl_lst")
    with col_d2:
        st.download_button("⬇️ Download UHI Standardized Index Map (.tif)", data=r["uhi_tiff"], file_name="Standardized_UHI_Map.tif", mime="image/tiff", key="dl_uhi")

    with st.expander("🛰️ Optional: Compare against MODIS LST (MOD11A1/MYD11A1)"):
        st.write(
            "This is not wired up yet because it needs a live data source with your "
            "own credentials. The two practical routes are:\n\n"
            "1. **Google Earth Engine** — pull the MODIS MOD11A1.061 (Terra) or "
            "MYD11A1.061 (Aqua) collection for the same date and boundary, resample "
            "to compare against this Landsat-derived LST. Requires a GEE service "
            "account and the `earthengine-api` package.\n"
            "2. **NASA AppEEARS** — request a MODIS LST point/area extraction via "
            "their REST API for the same date/AOI. Requires a NASA Earthdata login "
            "and API token.\n\n"
            "Tell me which one you'd like and I'll wire it into this app."
        )


@st.fragment
def lst_tab() -> None:
    current = st.session_state.get("acq")
    options: list[dict] = []
    if current:
        options.append(current)
    for d in acq.discover_acquisitions(DATA_DIR):
        if not current or d["dir"] != current["dir"]:
            options.append(d)

    step(1, "Data source", "Use the bands from the Acquisition tab, or upload files manually (Level-1 or Level-2)")
    src_choices = (["Use acquired scene"] if options else []) + ["Upload files manually"]
    source = st.radio("Data source", src_choices, horizontal=True, label_visibility="collapsed", key="lst_source")

    red_file = nir_file = thermal_file = mtl_file = None
    gdf = None
    satellite_choice = "Landsat 9 (OLI-2/TIRS-2)"
    study_area_ready = False

    if source == "Use acquired scene":
        pick = st.selectbox(
            "Scene", range(len(options)), key="lst_scene_pick",
            format_func=lambda i: f"{options[i]['scene_id']}  ·  {'clipped to AOI' if options[i]['mode'] == 'clipped' else 'full scene'}"
                                  + ("  ·  just acquired" if options[i] is current else ""),
        )
        rec = options[pick]
        f = rec["files"]
        red_file, nir_file, thermal_file, mtl_file = (LocalFile(f["B4"]), LocalFile(f["B5"]), LocalFile(f["B10"]), LocalFile(f["MTL_TXT"]))
        satellite_choice = "Landsat 9 (OLI-2/TIRS-2)" if rec["platform"] == "landsat-9" else "Landsat 8 (OLI/TIRS)"
        st.success(f"Loaded **{rec['scene_id']}** — {satellite_choice}. Bands: " + ", ".join(Path(v).name for k, v in f.items() if k != 'MTL_TXT'))

        override = st.file_uploader("Optional: use a different boundary (zipped shapefile)", type=["zip"], key="lst_shape_override")
        if override is not None:
            shp_path = extract_zip(override)
            if shp_path:
                gdf = gpd.read_file(shp_path)
            else:
                st.error("Invalid shapefile inside zip.")
        elif rec.get("aoi_path") and Path(rec["aoi_path"]).exists():
            gdf = gpd.read_file(rec["aoi_path"])
        elif st.session_state.get("aoi_gdf") is not None:
            gdf = st.session_state["aoi_gdf"]
        study_area_ready = gdf is not None
        if gdf is None:
            st.warning("No boundary found for this scene — upload a zipped shapefile above.")
    else:
        st.markdown(
            "Upload **either** Level-1 bands (`..._B4.TIF`, `..._B5.TIF`, `..._B10.TIF`) **or** Level-2 bands "
            "(`..._SR_B4.TIF`, `..._SR_B5.TIF`, `..._ST_B10.TIF`). Don't mix bands from two different scenes."
        )
        satellite_choice = st.radio(
            "Which dataset did you download?",
            ["Landsat 8 (OLI/TIRS)", "Landsat 9 (OLI-2/TIRS-2)"], horizontal=True, key="lst_sat",
            help="Used to sanity-check the MTL file and label outputs — the maths always uses the constants read from your MTL file.",
        )
        u1, u2 = st.columns(2)
        red_file = u1.file_uploader("Red Band (B4 or SR_B4)", type=["tif", "tiff"], key="up_red")
        nir_file = u2.file_uploader("NIR Band (B5 or SR_B5)", type=["tif", "tiff"], key="up_nir")
        thermal_file = u1.file_uploader("Thermal Band (B10 or ST_B10)", type=["tif", "tiff"], key="up_thermal")
        mtl_file = u2.file_uploader("MTL metadata file (_MTL.txt)", type=["txt"], key="up_mtl")
        shape_up = st.file_uploader("Study area boundary (zipped shapefile)", type=["zip"], key="up_shape")
        if shape_up is not None:
            shp_path = extract_zip(shape_up)
            if shp_path:
                gdf = gpd.read_file(shp_path)
                study_area_ready = True
            else:
                st.error("Invalid shapefile inside zip.")
        elif st.session_state.get("aoi_gdf") is not None:
            if st.checkbox("Use the study area from the Acquisition tab", value=True, key="use_tab1_aoi"):
                gdf = st.session_state["aoi_gdf"]
                study_area_ready = True

    step(2, "Advanced (optional)")
    with st.expander("NDVI emissivity override — Level-1 only"):
        manual_ndvi_override = st.checkbox(
            "Manually override NDVI min/max for emissivity (Level-1 only)", value=False, key="ndvi_override",
            help="Only applies to Level-1 processing, where NDVI drives the Proportion-of-Vegetation "
                 "emissivity correction. By default these are derived automatically from this scene's "
                 "own NDVI distribution (2nd/98th percentile). Level-2 ST already includes USGS's own "
                 "emissivity correction, so this has no effect there.",
        )
        manual_ndvi_min = manual_ndvi_max = None
        if manual_ndvi_override:
            manual_ndvi_min = st.number_input("NDVI min (bare soil)", value=0.2, step=0.01, format="%.2f", key="ndvi_min")
            manual_ndvi_max = st.number_input("NDVI max (full vegetation)", value=0.5, step=0.01, format="%.2f", key="ndvi_max")

    ready = all([red_file, nir_file, thermal_file, mtl_file]) and study_area_ready
    step(3, "Run analysis")
    if not ready:
        st.info(
            "💡 Ready when you are. Provide the 3 bands (Level-1 or Level-2), the scene's MTL.txt "
            "metadata file and your boundary — or acquire a scene in the first tab."
        )
    if st.button("▶ Run LST & UHI analysis", type="primary", disabled=not ready, key="btn_run"):
        with st.spinner("Reading scene metadata and running scene-calibrated LST algorithm..."):
            if "lst_out" in st.session_state:
                for k in ("fig1", "fig2"):
                    if st.session_state["lst_out"].get(k) is not None:
                        plt.close(st.session_state["lst_out"][k])
            st.session_state["lst_out"] = run_lst_analysis(
                red_file, nir_file, thermal_file, mtl_file, gdf, satellite_choice,
                manual_ndvi_override, manual_ndvi_min, manual_ndvi_max,
            )

    if "lst_out" in st.session_state:
        st.markdown("---")
        render_lst_results(st.session_state["lst_out"])


tab_acq, tab_lst = st.tabs(["🛰️  1 · Satellite Image Acquisition", "🌡️  2 · LST & UHI Analysis"])
with tab_acq:
    acquisition_tab()
with tab_lst:
    lst_tab()

st.markdown(
    '<div class="stc-footer">STC Consultants &amp; Service Provider · Universal LST Wizard · '
    "Landsat data courtesy of USGS / NASA, served via Microsoft Planetary Computer</div>",
    unsafe_allow_html=True,
)
