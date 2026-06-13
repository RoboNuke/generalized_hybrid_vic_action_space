"""Lightweight RGB coordinate-axis frame marker for eval-recording visualization.

Wraps Isaac Lab's :class:`VisualizationMarkers` with the prebuilt ``FRAME_MARKER_CFG`` (the
standard red/green/blue XYZ axis prim). One :class:`AxisFrameMarker` draws a single frame per
env at a commanded world pose; it is a real USD ``PointInstancer`` prim, so it is captured by
the offscreen recorder camera (unlike ``omni.isaac.debug_draw`` lines, which are not).

Isaac Lab markers only import cleanly after the Omniverse app has booted, so this module is
imported lazily (inside the control wrapper) rather than at module load.
"""

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class AxisFrameMarker:
    """A per-env RGB axis frame drawn at a commanded world pose.

    Args:
        prim_path: USD prim path for this marker set (must be unique per marker instance).
        scale: axis length in meters (the default ``frame_prim`` is ~1 m, far too big for a
            ~50 mm peg, so this is typically ~0.05).
    """

    def __init__(self, prim_path: str, scale: float):
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = prim_path
        # Keep only the XYZ axis prototype; drop the optional connecting-line cylinder.
        cfg.markers = {"frame": cfg.markers["frame"].replace(scale=(scale, scale, scale))}
        self._markers = VisualizationMarkers(cfg)

    def update(self, translations_w, orientations_wxyz):
        """Place the frame for every env.

        Args:
            translations_w: (E, 3) absolute world positions.
            orientations_wxyz: (E, 4) world-frame quaternions in (w, x, y, z) order.
        """
        self._markers.visualize(translations=translations_w, orientations=orientations_wxyz)
