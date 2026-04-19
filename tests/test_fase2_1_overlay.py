import numpy as np

from fase2_1.core.tryon.overlay import alpha_overlay_with_occlusion


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
