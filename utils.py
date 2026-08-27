def set_seed(random_seed):
    import torch, random
    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def quaternion_multiply(q1, q2):
    import numpy as np
    """Multiply two quaternions (x, y, z, w format)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

def quaternion_to_rotation_matrix(quaternion):
    import numpy as np
    import tf_transformations
    quaternion_np = np.array(quaternion, dtype=np.float64)
    rotation_matrix_4x4 = tf_transformations.quaternion_matrix(quaternion_np)
    rotation_matrix_3x3 = rotation_matrix_4x4[:3, :3]
    return rotation_matrix_3x3
