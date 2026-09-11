import math

import pytorch3d.transforms as transform
import torch


SPIN_DELTA_SCALE = 20.0


def _build_spin_basis(spin_axis):
    """Build a deterministic orthonormal basis around the spin axis."""
    axis = torch.nn.functional.normalize(spin_axis, dim=-1)
    ref = torch.zeros_like(axis)
    ref[:, 0] = 1.0
    alt_ref = torch.zeros_like(axis)
    alt_ref[:, 1] = 1.0
    use_alt = torch.abs(axis[:, 0]) > 0.9
    ref = torch.where(use_alt.unsqueeze(-1), alt_ref, ref)

    vector_1 = torch.cross(axis, ref, dim=-1)
    vector_1 = torch.nn.functional.normalize(vector_1, dim=-1)
    vector_2 = torch.cross(axis, vector_1, dim=-1)
    vector_2 = torch.nn.functional.normalize(vector_2, dim=-1)
    return axis, vector_1, vector_2


def compute_legacy_spin_deltas_from_rot_mats(prev_rot_mats, curr_rot_mats, spin_axis):
    """Legacy theta/dev-angle algorithm using rotation matrices."""
    axis, vector_1, vector_2 = _build_spin_basis(spin_axis)
    inverse_rotation_matrix = prev_rot_mats.transpose(1, 2)
    vector_1_new = torch.bmm(inverse_rotation_matrix, vector_1.unsqueeze(-1))
    vector_1_new = torch.bmm(curr_rot_mats, vector_1_new).squeeze(-1)

    rot_vec_coordinate_1 = (vector_1_new * vector_1).sum(-1).reshape(-1, 1)
    rot_vec_coordinate_2 = (vector_1_new * vector_2).sum(-1).reshape(-1, 1)
    rot_vec_coordinate_3 = (vector_1_new * axis).sum(-1)

    rot_vec_coordinate_3 = torch.clamp(rot_vec_coordinate_3, -1.0, 1.0)
    dev_angle = torch.abs(0.5 * math.pi - torch.arccos(rot_vec_coordinate_3))

    rot_vec = torch.cat((rot_vec_coordinate_1, rot_vec_coordinate_2), dim=-1)
    rot_vec = torch.nn.functional.normalize(rot_vec, dim=-1)

    inner_prod = torch.clamp(rot_vec[:, 0], -1.0, 1.0)
    theta_sign = torch.sign(rot_vec[:, 1])
    theta = theta_sign * torch.arccos(inner_prod)
    return theta * SPIN_DELTA_SCALE, dev_angle * SPIN_DELTA_SCALE


def _quaternion_to_axis_angle_stable(quat_wxyz):
    """Convert quaternion to axis-angle with stable q/-q handling."""
    quat = torch.nn.functional.normalize(quat_wxyz, dim=-1)
    sign = torch.where(quat[:, :1] < 0.0, -torch.ones_like(quat[:, :1]), torch.ones_like(quat[:, :1]))
    quat = quat * sign

    quat_vec = quat[:, 1:]
    sin_half = torch.norm(quat_vec, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, torch.clamp(quat[:, 0], min=1e-8))
    scale = torch.where(sin_half > 1e-8, angle / sin_half, torch.full_like(sin_half, 2.0))
    return quat_vec * scale.unsqueeze(-1)


def compute_spin_deltas_from_rot_mats(prev_rot_mats, curr_rot_mats, spin_axis):
    """Compute control-step spin deltas from relative rotation.

    Returns:
        spin_delta_axis: per-control-step rotation around target axis, unit = rad * 20.
        spin_delta_offaxis: per-control-step off-axis rotation magnitude, unit = rad * 20.
    """
    axis = torch.nn.functional.normalize(spin_axis, dim=-1)
    relative_rot = torch.bmm(curr_rot_mats, prev_rot_mats.transpose(1, 2))
    relative_quat = transform.matrix_to_quaternion(relative_rot)
    relative_axis_angle = _quaternion_to_axis_angle_stable(relative_quat)

    spin_delta_axis_raw = (relative_axis_angle * axis).sum(-1)
    spin_delta_offaxis_raw = torch.norm(
        relative_axis_angle - spin_delta_axis_raw.unsqueeze(-1) * axis, dim=-1
    )
    return spin_delta_axis_raw * SPIN_DELTA_SCALE, spin_delta_offaxis_raw * SPIN_DELTA_SCALE