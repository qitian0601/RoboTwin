import argparse
from pathlib import Path

import numpy as np
import sapien
from PIL import Image


def look_at_pose(camera_pos, target):
    camera_pos = np.array(camera_pos, dtype=float)
    target = np.array(target, dtype=float)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = camera_pos
    return sapien.Pose(mat)


def make_material(
    color,
    roughness=0.5,
    metallic=0.0,
    specular=0.5,
    transmission=0.0,
    ior=1.45,
):
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.roughness = roughness
    mat.metallic = metallic
    mat.specular = specular
    mat.transmission = transmission
    mat.ior = ior
    return mat


def add_box(scene, name, pose, half_size, material):
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=half_size, material=material)
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(pose))
    return actor


def add_sphere(scene, name, pose, radius, material):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=material)
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(pose))
    return actor


def add_cylinder(scene, name, pose, radius, half_length, material):
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(radius=radius, half_length=half_length, material=material)
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(pose))
    return actor


def save_color(camera, path):
    camera.take_picture()
    color = camera.get_picture("Color")
    rgb = (np.clip(color[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def render_scene(output_path, mode, denoiser):
    if mode == "rt":
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(256)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser(denoiser)
    else:
        sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.set_ambient_light([0.02, 0.02, 0.02])

    # Large emissive panel and RT area light make reflections and transmission visible.
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([0.0, -2.2, 4.0], [0.7071068, 0.7071068, 0.0, 0.0]),
        [5.0, 5.0, 5.0],
        2.0,
        2.0,
    )
    scene.add_directional_light([0.3, 0.5, -1.0], [0.8, 0.8, 0.8], shadow=True)

    floor_mat = make_material([0.82, 0.82, 0.78, 1.0], roughness=0.28, specular=0.65)
    mirror_mat = make_material([0.92, 0.84, 0.65, 1.0], roughness=0.06, metallic=1.0, specular=1.0)
    glass_mat = make_material(
        [0.78, 0.92, 1.0, 0.45],
        roughness=0.0,
        metallic=0.0,
        specular=1.0,
        transmission=0.95,
        ior=1.45,
    )
    rough_mat = make_material([0.95, 0.18, 0.08, 1.0], roughness=0.82, specular=0.25)
    blue_mat = make_material([0.05, 0.2, 0.9, 1.0], roughness=0.22, specular=0.8)

    add_box(scene, "floor", [0.0, 0.0, -0.05], [4.0, 3.0, 0.05], floor_mat)
    add_box(scene, "back_wall", [1.55, 0.0, 1.45], [0.05, 3.0, 1.5], floor_mat)
    add_box(scene, "left_wall", [0.0, 1.55, 1.45], [4.0, 0.05, 1.5], floor_mat)

    add_sphere(scene, "gold_mirror_sphere", [-0.55, -0.25, 0.55], 0.55, mirror_mat)
    add_sphere(scene, "glass_sphere", [0.65, -0.1, 0.55], 0.55, glass_mat)
    add_box(scene, "rough_red_box", [0.05, 0.75, 0.35], [0.35, 0.35, 0.35], rough_mat)
    add_cylinder(scene, "blue_cylinder", [-1.15, 0.7, 0.45], 0.28, 0.45, blue_mat)

    camera = scene.add_camera("camera", 1280, 720, np.deg2rad(45), 0.01, 100)
    camera.entity.set_pose(look_at_pose([-3.0, -2.2, 1.6], [0.05, 0.05, 0.55]))

    for _ in range(8):
        scene.step()
    scene.update_render()
    save_color(camera, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/sapien_rt_demo")
    parser.add_argument("--denoiser", default="optix")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_scene(out_dir / "material_raster.png", "raster", args.denoiser)
    render_scene(out_dir / "material_rt_256spp.png", "rt", args.denoiser)
    print(f"Saved demo images to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
