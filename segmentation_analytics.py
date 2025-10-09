# import json
# import plotly.io as pio
# import streamlit as st

# st.title("3D Lesion Topography")

# with open("images/3d_heatmap.json", "r") as f:
#     fig = pio.from_json(f.read())

# st.plotly_chart(fig, use_container_width=True)


# #translate in english 
# st.title("Segmentation et analyse des lésions cornéennes par tomographie en cohérence optique ")
# st.header("ABOUT THE PROJECT")
# st.write("This project focuses on the segmentation and analysis of corneal lesions using Optical Coherence Tomography (OCT) images. The goal is to develop a robust deep learning model that can accurately identify and segment corneal ulcers, facilitating better diagnosis and treatment planning for ophthalmologists.")
# st.divider()
# st.header("DATASET")
# st.write("How its done")
# st.image("images/poster_image_oct.png", use_column_width=True)
# st.divider()
# person1, person2 = st.tabs(["Corneal Ulcer 1", "Corneal Ulcer 2"])
# with person1:
#     st.header("Corneal Ulcer 1")
#     st.write("Description of Corneal Ulcer 1")
#     st.image("images/ulcus_1_foto_uakrostock_2.jpg", use_column_width=True)
# with person2:
#     st.header("Corneal Ulcer 2")
#     st.write("Description of Corneal Ulcer 2")
#     st.image("images/ulcus_1_foto_uakrostock_2.jpg", use_column_width=True)
# st.divider()

# segmentation_analytics.py
# Streamlit app: 3D Lesion Topography + OCT Segmentation Carousel

# segmentation_analytics.py
# Streamlit app: 3D Lesion Topography + OCT Segmentation Carousel (fast)

# segmentation_analytics.py
# Fast Streamlit app: 3D Lesion Topography + super-snappy carousel via WebP previews

from pathlib import Path
from typing import List
import json
import plotly.io as pio
from PIL import Image, ImageOps
import streamlit as st

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="3D Lesion Topography",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== PATHS =====
BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "images"
JSON_MAP= IMG_DIR / "heatmaps_json"
SEG_DIR = IMG_DIR / "oct_segmentation1" 
SEG_DIR2 = IMG_DIR / "oct_segmentation2"
SEG_DIR3 = IMG_DIR / "oct_segmentation3"
PREV_DIR = IMG_DIR / "oct_segmentation_previews1"      # auto-generated WebP previews
PREV_DIR2 = IMG_DIR / "oct_segmentation_previews2" 
PREV_DIR3 = IMG_DIR / "oct_segmentation_previews3" 
PREV_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
FIG_JSON = JSON_MAP / "3d_heatmap.json"
FIG_JSON2 = JSON_MAP / "3d_heatmap2.json"
FIG_JSON3 = JSON_MAP / "3d_heatmap3.json"
FIG_JSON4 = JSON_MAP / "3d_heatmap4.json"
FIG_JSON5 = JSON_MAP / "3d_heatmap5.json"
FIG_JSON6 = JSON_MAP / "3d_heatmap6.json"



# ===== CACHED LOADERS =====
@st.cache_data(show_spinner=True)
def load_plotly_fig(json_path: Path):
    with open(json_path, "r") as f:
        return pio.from_json(f.read())

@st.cache_data(show_spinner=True)
def list_source_slices(seg_dir: Path) -> List[Path]:
    if not seg_dir.exists():
        return []
    files = [p for p in seg_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTS]
    files.sort(key=lambda p: p.name)
    return files

@st.cache_data(show_spinner=True)
def build_previews(seg_dir: Path, prev_dir: Path, max_dim: int = 1000, quality: int = 80) -> List[Path]:
    """
    Ensure every source image has a WebP preview; return sorted list of preview paths.
    This only runs when new files appear; otherwise it’s a no-op (cached).
    """
    prev_dir.mkdir(parents=True, exist_ok=True)
    srcs = list_source_slices(seg_dir)

    preview_paths: List[Path] = []
    for src in srcs:
        # One-to-one mapping: <name>.webp
        dest = prev_dir / (src.stem + ".webp")
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            # (Re)generate preview
            img = Image.open(src)

            # Multi-frame TIFF: use first frame for speed
            if getattr(img, "n_frames", 1) > 1:
                try:
                    img.seek(0)
                except Exception:
                    pass

            # Fix orientation, convert to RGB
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Downscale (keeps aspect ratio)
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            # Save as WebP for tiny, fast loads
            img.save(dest, format="WEBP", quality=quality, method=0)  # method=6 is slower once, faster to load
        preview_paths.append(dest)

    preview_paths.sort(key=lambda p: p.name)
    return preview_paths



# ===== MAIN TITLE =====
st.title("Automated OCT-based corneal ulcers segmentation and 3D mapping")

# ---- About ----
st.header("About the Project")
st.write("""
This project introduces a **Python-based algorithm** for the segmentation and 
analysis of corneal lesions from OCT B-scans. By combining the Sobel operator with a weighted moving average, 
the algorithm detects corneal contours, estimates curvature, and calculates lesion area and volume, **generating both 2D and 3D topographic maps** for each patient. 
This method aims to enhance corneal **3D-bioprinting**  to stimulate the regeneration by enabling personalized treatments and minimizing injection inaccuracies. 
The long-term goal is to train an AI model capable of estimating ulcer severity using smartphone images.
""")
st.divider()

# ---- Dataset ----
st.header("What is an Optical Coherence Tomography (OCT) Scan?")
st.write("""
A corneal OCT scan is a quick, non-invasive test that uses light to 
capture micrometer-resolution cross-section images of the cornea, letting clinicians see its layers and measure thickness 
to detect problems like scars, ulcers, or swelling.
""")
img_intact = IMG_DIR / "intact_oct_scan.png"
if img_intact.exists():
    st.image(
        str(img_intact),
        caption="Normal cornea on OCT: cross-sectional scan selected from a 21-scan acquisition",
        use_container_width=True
    )

img_intact = IMG_DIR / "wound_oct_scan.png"
if img_intact.exists():
    st.image(
        str(img_intact),
        caption="Wounded cornea on OCT: cross-sectional scan selected from a 21-scan acquisition",
        use_container_width=True
    )

st.divider()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Contour Detection")
    st.write("""
    A Sobel-Y gradient is computed to detect vertical intensity transitions (black → white). Only positive 
    gradients are retained, corresponding to the upper corneal surface.
    → **This edge-detection step isolates the anterior corneal contour by identifying rapid brightness 
    changes along each column**.
    """)

with col2:
    cornea_scan = IMG_DIR / "corneal_original.png"
    if cornea_scan.exists():
        st.image(
            str(cornea_scan),
            caption="(1) Initial OCT scan with **no corneal contour highlighted**.",
            use_container_width=False
        )
    sobel_op = IMG_DIR / "sobel_final1.png"
    if sobel_op.exists():
        st.image(
            str(sobel_op),
            caption="(2) Edge detection using the Sobel operator to highlight the **anterior corneal contour**.",
            use_container_width=False
        )
    detection_op = IMG_DIR / "detection_sobel.png"
    if detection_op.exists():
        st.image(
            str(detection_op),
            caption="(3) Detected contour after applying the Sobel operator.",
            use_container_width=False
        )    

# st.divider()
# detection_op = IMG_DIR / "detection_sobel.png"
# if poster.exists():
#     st.image(str(poster), caption="Example of corneal B-scan OCT", use_container_width=True)
# else:
#     st.info(f"Add an image at: {poster}")
st.divider()

# ---- Lesion Examples ----
st.header("Lesion Examples")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Corneal Ulcer 1", "Corneal Ulcer 2", "Corneal Ulcer 3", "Corneal Ulcer 4", "Corneal Ulcer 5", "Corneal Ulcer 6", "Validation"])

with tab1:
    st.subheader("Estimated volume of 0.55 µL using 21 scans")
    col1, col2 = st.columns([1, 2])

    # Left column: clinical photos (optional)
    with col1:
        img1 = IMG_DIR / "ulcer_fluo.png"
        if img1.exists():
            st.image(str(img1), caption="Slit-lamp photograph showing a corneal ulcer highlighted with fluorescein dye under cobalt blue illumination.", use_container_width=True)
        img2 = IMG_DIR / "slit_lamp.png"
        if img2.exists():
            st.image(str(img2), caption="Corneal ulcer observed under slit-lamp", use_container_width=True)

    # Right column: 3D + Carousel
    with col2:
        st.header("🌐 3D Lesion Topography")
        if FIG_JSON.exists():
            fig = load_plotly_fig(FIG_JSON)
            fig.update_layout(template=None)
            fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1.2, y=1, z=0.15)
                ),
                width=800,
                height=600
            )
            st.plotly_chart(fig, theme=None, use_container_width=False)

        else:
            st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

        st.divider()
    st.subheader("🖼️ Segmented OCT slices")
        # Build/get previews once (cached)
    previews = build_previews(SEG_DIR, PREV_DIR, max_dim=600, quality=70)

    if not previews:
        st.info(
            f"No source images found in `{SEG_DIR}`.\n"
            f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
        )
    else:
        # --- PARAMETERS ---
        n_cols = 3   # number of images per row
        max_to_show = 30  # safety cap so you don't overload the page
        previews = previews[:max_to_show]

        # Split list of images into rows of n_cols
        rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

        # Display each row
        for row in rows:
            cols = st.columns(len(row))
            for col, img_path in zip(cols, row):
                col.image(str(img_path),
                        caption=img_path.name,
                        use_container_width=True)


with tab2:
    st.subheader("Estimated volume of 0.21 µL using 41 scans")
    col1, col2 = st.columns([1, 2])

    # Left column: clinical photos (optional)
    with col1:
        img1 = IMG_DIR / "ulcer_fluo2.png"
        if img1.exists():
            st.image(str(img1), caption="Slit-lamp photograph showing a corneal ulcer highlighted with fluorescein dye under cobalt blue illumination.", use_container_width=True)
        img2 = IMG_DIR / "slit_lamp2.png"
        if img2.exists():
            st.image(str(img2), caption="Corneal ulcer observed under slit-lamp", use_container_width=True)

    # Right column: 3D + Carousel
    with col2:
        st.header("🌐 3D Lesion Topography")
        if FIG_JSON.exists():
            fig = load_plotly_fig(FIG_JSON2)
            fig.update_layout(template=None)
            fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=2, y=1, z=0.15)
                ),
                width=800,
                height=600
            )
            st.plotly_chart(fig, theme=None, use_container_width=False)

        else:
            st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

        st.divider()
    st.subheader("🖼️ Segmented OCT slices")
        # Build/get previews once (cached)
    previews = build_previews(SEG_DIR2, PREV_DIR2, max_dim=600, quality=70)

    if not previews:
        st.info(
            f"No source images found in `{SEG_DIR}`.\n"
            f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
        )
    else:
        # --- PARAMETERS ---
        n_cols = 3   # number of images per row
        max_to_show = 30  # safety cap so you don't overload the page
        previews = previews[:max_to_show]

        # Split list of images into rows of n_cols
        rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

        # Display each row
        for row in rows:
            cols = st.columns(len(row))
            for col, img_path in zip(cols, row):
                col.image(str(img_path),
                        caption=img_path.name,
                        use_container_width=True)
with tab3:
    st.subheader("Estimated volume of 1.03 µL using 11 scans")
    col1, col2 = st.columns([1, 2])

    # Left column: clinical photos (optional)
    with col1:
        img1 = IMG_DIR / "oct_section3.png"
        if img1.exists():
            st.image(str(img1), use_container_width=True)


    # Right column: 3D + Carousel
    with col2:
        st.header("🌐 3D Lesion Topography")
        if FIG_JSON.exists():
            fig = load_plotly_fig(FIG_JSON3)
            fig.update_layout(template=None)
            fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=2, y=1, z=0.15)
                ),
                width=800,
                height=600
            )
            st.plotly_chart(fig, theme=None, use_container_width=False)

        else:
            st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

        st.divider()
    st.subheader("🖼️ Segmented OCT slices")
        # Build/get previews once (cached)
    previews = build_previews(SEG_DIR3, PREV_DIR3, max_dim=600, quality=70)

    if not previews:
        st.info(
            f"No source images found in `{SEG_DIR}`.\n"
            f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
        )
    else:
        # --- PARAMETERS ---
        n_cols = 3   # number of images per row
        max_to_show = 30  # safety cap so you don't overload the page
        previews = previews[:max_to_show]

        # Split list of images into rows of n_cols
        rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

        # Display each row
        for row in rows:
            cols = st.columns(len(row))
            for col, img_path in zip(cols, row):
                col.image(str(img_path),
                        caption=img_path.name,
                        use_container_width=True)
# with tab4:
#     st.subheader("Estimated volume of 0.17 µL using 22 scans")
#     col1, col2 = st.columns([1, 2])

#     # Left column: clinical photos (optional)
#     with col1:
#         img1 = IMG_DIR / "oct_section4.png"
#         if img1.exists():
#             st.image(str(img1), use_container_width=True)


#     # Right column: 3D + Carousel
#     with col2:
#         st.header("🌐 3D Lesion Topography")
#         if FIG_JSON.exists():
#             fig = load_plotly_fig(FIG_JSON3)
#             fig.update_layout(template=None)
#             fig.update_layout(
#                 scene=dict(
#                     aspectmode='manual',
#                     aspectratio=dict(x=2, y=1, z=0.15)
#                 ),
#                 width=800,
#                 height=600
#             )
#             st.plotly_chart(fig, theme=None, use_container_width=False)

#         else:
#             st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

#         st.divider()
#     st.subheader("🖼️ Segmented OCT slices")
#         # Build/get previews once (cached)
#     previews = build_previews(SEG_DIR3, PREV_DIR3, max_dim=600, quality=70)

#     if not previews:
#         st.info(
#             f"No source images found in `{SEG_DIR}`.\n"
#             f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
#         )
#     else:
#         # --- PARAMETERS ---
#         n_cols = 3   # number of images per row
#         max_to_show = 30  # safety cap so you don't overload the page
#         previews = previews[:max_to_show]

#         # Split list of images into rows of n_cols
#         rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

#         # Display each row
#         for row in rows:
#             cols = st.columns(len(row))
#             for col, img_path in zip(cols, row):
#                 col.image(str(img_path),
#                         caption=img_path.name,
#                         use_container_width=True)
# with tab5:
#     st.subheader("Estimated volume of 0.10 µL using 22 scans")
#     col1, col2 = st.columns([1, 2])

#     # Left column: clinical photos (optional)
#     with col1:
#         img1 = IMG_DIR / "oct_section5.png"
#         if img1.exists():
#             st.image(str(img1), use_container_width=True)


#     # Right column: 3D + Carousel
#     with col2:
#         st.header("🌐 3D Lesion Topography")
#         if FIG_JSON.exists():
#             fig = load_plotly_fig(FIG_JSON3)
#             fig.update_layout(template=None)
#             fig.update_layout(
#                 scene=dict(
#                     aspectmode='manual',
#                     aspectratio=dict(x=2, y=1, z=0.15)
#                 ),
#                 width=800,
#                 height=600
#             )
#             st.plotly_chart(fig, theme=None, use_container_width=False)

#         else:
#             st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

#         st.divider()
#     st.subheader("🖼️ Segmented OCT slices")
#         # Build/get previews once (cached)
#     previews = build_previews(SEG_DIR3, PREV_DIR3, max_dim=600, quality=70)

#     if not previews:
#         st.info(
#             f"No source images found in `{SEG_DIR}`.\n"
#             f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
#         )
#     else:
#         # --- PARAMETERS ---
#         n_cols = 3   # number of images per row
#         max_to_show = 30  # safety cap so you don't overload the page
#         previews = previews[:max_to_show]

#         # Split list of images into rows of n_cols
#         rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

#         # Display each row
#         for row in rows:
#             cols = st.columns(len(row))
#             for col, img_path in zip(cols, row):
#                 col.image(str(img_path),
#                         caption=img_path.name,
#                         use_container_width=True)
# with tab6:
#     st.subheader("Estimated volume of 0.21 µL using 81 scans")
#     col1, col2 = st.columns([1, 2])

#     # Left column: clinical photos (optional)
#     with col1:
#         img1 = IMG_DIR / "oct_section6.png"
#         if img1.exists():
#             st.image(str(img1), use_container_width=True)


#     # Right column: 3D + Carousel
#     with col2:
#         st.header("🌐 3D Lesion Topography")
#         if FIG_JSON.exists():
#             fig = load_plotly_fig(FIG_JSON3)
#             fig.update_layout(template=None)
#             fig.update_layout(
#                 scene=dict(
#                     aspectmode='manual',
#                     aspectratio=dict(x=2, y=1, z=0.15)
#                 ),
#                 width=800,
#                 height=600
#             )
#             st.plotly_chart(fig, theme=None, use_container_width=False)

#         else:
#             st.warning(f"3D visualization not found. Please add: {FIG_JSON}")

#         st.divider()
#     st.subheader("🖼️ Segmented OCT slices")
#         # Build/get previews once (cached)
#     previews = build_previews(SEG_DIR3, PREV_DIR3, max_dim=600, quality=70)

#     if not previews:
#         st.info(
#             f"No source images found in `{SEG_DIR}`.\n"
#             f"Add .tif/.tiff/.png/.jpg files there. Previews will appear in `{PREV_DIR}`."
#         )
#     else:
#         # --- PARAMETERS ---
#         n_cols = 3   # number of images per row
#         max_to_show = 30  # safety cap so you don't overload the page
#         previews = previews[:max_to_show]

#         # Split list of images into rows of n_cols
#         rows = [previews[i:i + n_cols] for i in range(0, len(previews), n_cols)]

#         # Display each row
#         for row in rows:
#             cols = st.columns(len(row))
#             for col, img_path in zip(cols, row):
#                 col.image(str(img_path),
#                         caption=img_path.name,
#                         use_container_width=True)

st.divider()

st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
© 2025 Mariam Jemaa | Université de Montréal
</div>
""", unsafe_allow_html=True)





