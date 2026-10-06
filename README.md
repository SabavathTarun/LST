# STC Universal LST Wizard

One Streamlit tool, two tabs:

1. **Satellite Image Acquisition** – upload your boundary (zipped shapefile / GeoJSON / KML) or enter a bounding box,
   search Landsat 8/9 Collection 2 Level-2 scenes (date, cloud cover, satellite, AOI coverage %), and download
   **B4 (SR_B4), B5 (SR_B5), B10 (ST_B10) + MTL** – clipped to your study area (fast) or as full original tiles.
2. **LST & UHI Analysis** – the validated STC engine. Acquired bands are loaded automatically (no re-upload),
   or you can upload Level-1 / Level-2 files manually as before.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Needs internet access to `planetarycomputer.microsoft.com` (no login / API key required).

## Files
| File | Purpose |
|---|---|
| `app.py` | UI, branding, both tabs |
| `acquisition.py` | STAC search, URL signing, AOI-window / full-scene download |
| `lst_core.py` | LST/UHI functions from the standalone tool (unchanged maths) |
| `assets/` | STC logo (transparent PNG, icon, original JPEG) |
| `.streamlit/config.toml` | Navy & gold theme |

Downloaded bands go to `STC_LST_Wizard_Data/<scene_id>/…` (change in the sidebar). Re-running a scene re-uses existing files.

## Notes
* Planetary Computer serves **Level-2** products only (SR + ST). Level-1 workflows remain available via manual upload in Tab 2.
* Scene cloud % covers the whole ~185 km tile – check the preview over your own boundary.
