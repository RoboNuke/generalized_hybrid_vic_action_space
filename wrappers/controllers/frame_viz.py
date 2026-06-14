"""Lightweight RGB coordinate-axis frame marker for eval-recording visualization.

Wraps Isaac Lab's :class:`VisualizationMarkers` with the prebuilt ``FRAME_MARKER_CFG`` (the
standard red/green/blue XYZ axis prim). One :class:`AxisFrameMarker` draws a single frame per
env at a commanded world pose; it is a real USD ``PointInstancer`` prim, so it is captured by
the offscreen recorder camera (unlike ``omni.isaac.debug_draw`` lines, which are not).

Isaac Lab markers only import cleanly after the Omniverse app has booted, so this module is
imported lazily (inside the control wrapper) rather than at module load.
"""

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg


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


class EllipsoidMarker:
    """A per-env translucent ellipsoid (a non-uniformly scaled sphere) at a commanded world pose.

    Used to draw the translational stiffness/compliance ellipsoid: a unit-radius sphere prototype
    scaled per-axis by the principal semi-axis lengths and oriented by the principal-axis frame.
    Like :class:`AxisFrameMarker` it is a real USD prim, so it is captured by the offscreen recorder
    camera. Note: ``PreviewSurfaceCfg.opacity`` is honored by the interactive/RTX viewport; if the
    recorded video shows it fully opaque, drop the opacity or switch to a wireframe prototype.

    Args:
        prim_path: USD prim path for this marker set (must be unique per marker instance).
        color: RGB diffuse color in [0, 1].
        opacity: surface opacity in [0, 1] (lower = more see-through).
    """

    def __init__(self, prim_path: str, color=(0.1, 0.5, 0.9), opacity: float = 0.35):
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "ellipsoid": sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, opacity=opacity
                    ),
                )
            },
        )
        self._markers = VisualizationMarkers(cfg)

    def update(self, translations_w, orientations_wxyz, scales):
        """Place + shape the ellipsoid for every env.

        Args:
            translations_w: (E, 3) absolute world positions (the ellipsoid center).
            orientations_wxyz: (E, 4) world-frame quaternions in (w, x, y, z) order; the columns of
                the corresponding rotation are the ellipsoid's principal axes.
            scales: (E, 3) per-axis semi-axis lengths in meters, aligned with those principal axes.
                Applied before the orientation (matches VisualizationMarkers semantics).
        """
        self._markers.visualize(
            translations=translations_w, orientations=orientations_wxyz, scales=scales
        )
