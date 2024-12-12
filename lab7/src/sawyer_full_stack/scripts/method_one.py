def get_scale_factors(ex_image_x, ex_image_y, ):
    # Extract camera intrinsic parameters

    # Convert pixel to normalized camera coordinate
    width = 752
    height = 480

    new_x = ex_image_x - width/2
    new_y = height/2 - ex_image_y

    # 84, 60

    default_position = [0.78326561, 0.15723982, 0.46321379]  # TODO

    result = [0.754, 0.018, -0.108]

    depth = default_position[2] - result[2]

    scale_factor_x = (result[0] - default_position[0])/new_x/depth

    scale_factor_y = (result[1] - default_position[1])/new_y/depth

    print(f"scale factors: {scale_factor_x, scale_factor_y}")
    return scale_factor_x, scale_factor_y


def get_real_world_coordinates(image_x, image_y):

    scale_factor_x, scale_factor_y = get_scale_factors(450, 180)

    # Convert pixel to normalized camera coordinate
    width = 752
    height = 480

    new_x = image_x - width/2
    new_y = height/2 - image_y

    # 84, 60

    default_position = [0.681, 0.121, 0.426]  # TODO

    new_z = -0.108

    depth = default_position[2] - new_z

    new_x = (scale_factor_x * depth * new_x) + default_position[0]
    new_y = (scale_factor_y * depth * new_y) + default_position[1]

    return new_x, new_y, new_z


get_scale_factors(450, 180)

print(get_real_world_coordinates(438, 294))