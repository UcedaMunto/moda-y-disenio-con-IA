import numpy as np

from fase2_1.core.tryon.overlay import (
    alpha_overlay_with_occlusion,
    build_upper_occlusion_mask,
    compose_tryon_layers,
)


def test_alpha_overlay_with_occlusion_respects_mask() -> None:
    background = np.zeros((4, 4, 3), dtype=np.uint8)

    # Overlay rojo opaco 2x2
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :, 0] = 255
    overlay[:, :, 3] = 255

    # Occlusion: bloquear pixel superior izquierdo del área objetivo
    occ = np.zeros((4, 4), dtype=np.uint8)
    occ[1, 1] = 255

    out = alpha_overlay_with_occlusion(background.copy(), overlay, occ, x=1, y=1)

    # Pixel ocluido se mantiene en negro
    assert (out[1, 1] == np.array([0, 0, 0], dtype=np.uint8)).all()

    # Pixel libre queda rojo
    assert (out[1, 2] == np.array([255, 0, 0], dtype=np.uint8)).all()
    assert (out[2, 1] == np.array([255, 0, 0], dtype=np.uint8)).all()
    assert (out[2, 2] == np.array([255, 0, 0], dtype=np.uint8)).all()


def test_alpha_overlay_with_occlusion_out_of_bounds_safe() -> None:
    background = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = np.zeros((3, 3, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 255
    occ = np.zeros((2, 2), dtype=np.uint8)

    out = alpha_overlay_with_occlusion(background.copy(), overlay, occ, x=1, y=1)
    assert out.shape == (2, 2, 3)


def test_build_upper_occlusion_mask_limits_to_upper_zone() -> None:
    person_mask = np.full((10, 4), 255, dtype=np.uint8)

    upper = build_upper_occlusion_mask(person_mask, upper_ratio=0.3)

    assert upper.shape == person_mask.shape
    assert (upper[:3, :] == 255).all()
    assert (upper[3:, :] == 0).all()


def test_compose_tryon_layers_applies_ordered_occlusion() -> None:
    background = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255
    overlay[:, :, 3] = 255

    occ = np.zeros((4, 4), dtype=np.uint8)
    occ[1, 1] = 255

    out = compose_tryon_layers(background, overlay, x=1, y=1, occlusion_mask=occ)

    assert (out[1, 1] == np.array([0, 0, 0], dtype=np.uint8)).all()
    assert (out[1, 2] == np.array([0, 0, 255], dtype=np.uint8)).all()
