#!/usr/bin/env python3
"""
Parallel Synth - Blender Procedural Generator
Generates 3D rendering samples across all taxonomy categories
"""

import bpy
import bmesh
import math
import random
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

class ParallelSynthGenerator:
    """Main generator class for creating diverse 3D rendering samples"""

    def __init__(self, output_dir: str, taxonomy_path: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load taxonomy
        with open(taxonomy_path, 'r') as f:
            import yaml
            self.taxonomy = yaml.safe_load(f)['taxonomy']

        self.scene = bpy.context.scene
        self.setup_scene()

    def setup_scene(self):
        """Initialize scene with render settings"""
        # Clear default scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Render settings
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.samples = 128
        self.scene.cycles.use_denoising = True
        self.scene.render.resolution_x = 1920
        self.scene.render.resolution_y = 1080
        self.scene.render.film_transparent = False

        # Color management
        self.scene.view_settings.view_transform = 'Filmic'
        self.scene.sequencer_colorspace_settings.name = 'sRGB'

    def generate_sample(self, categories: List[str], seed: int = None) -> Dict[str, Any]:
        """Generate a single sample with specified categories"""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        random.seed(seed)
        np.random.seed(seed)

        sample_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'

        metadata = {
            'sample_id': sample_id,
            'version': '1.0.0',
            'timestamp': timestamp,
            'source': {
                'type': 'procedural',
                'generator': f'Blender {bpy.app.version_string}',
                'seed': seed
            },
            'files': {},
            'categories': {},
            'captions': {},
            'tags': [],
            'quality_metrics': {}
        }

        # Clear scene
        self.clear_scene()

        # Generate based on categories
        if 'geometry' in categories:
            metadata['categories']['geometry'] = self.generate_geometry()

        if 'materials' in categories:
            metadata['categories']['materials'] = self.generate_materials()

        if 'lighting' in categories:
            metadata['categories']['lighting'] = self.generate_lighting()

        if 'camera' in categories:
            metadata['categories']['camera'] = self.generate_camera()

        if 'liquids' in categories:
            metadata['categories']['liquids'] = self.generate_liquid_simulation()

        if 'gases' in categories:
            metadata['categories']['gases'] = self.generate_gas_simulation()

        # Generate captions
        metadata['captions'] = self.generate_captions(metadata['categories'])

        # Render
        render_start = datetime.now()
        output_path = self.output_dir / f"{sample_id}.png"
        self.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        render_time = (datetime.now() - render_start).total_seconds()

        metadata['files']['image'] = {
            'path': str(output_path),
            'format': 'png',
            'resolution': {
                'width': self.scene.render.resolution_x,
                'height': self.scene.render.resolution_y
            },
            'bit_depth': 8,
            'color_space': 'sRGB'
        }

        metadata['quality_metrics']['render_time'] = render_time
        metadata['quality_metrics']['sample_count'] = self.scene.cycles.samples

        # Save metadata
        metadata_path = self.output_dir / f"{sample_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save blend file
        blend_path = self.output_dir / f"{sample_id}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

        return metadata

    def clear_scene(self):
        """Clear all objects from scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def generate_geometry(self) -> Dict[str, Any]:
        """Generate random geometry"""
        geometry_types = ['cube', 'sphere', 'cylinder', 'torus', 'cone', 'monkey']
        geo_type = random.choice(geometry_types)

        if geo_type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=2)
        elif geo_type == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16)
        elif geo_type == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, vertices=32)
        elif geo_type == 'torus':
            bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.25)
        elif geo_type == 'cone':
            bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, vertices=32)
        elif geo_type == 'monkey':
            bpy.ops.mesh.primitive_monkey_add()

        obj = bpy.context.active_object

        # Random rotation
        obj.rotation_euler = (
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2)
        )

        # Subdivision
        subdiv_level = random.randint(0, 3)
        if subdiv_level > 0:
            modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            modifier.levels = subdiv_level
            modifier.render_levels = subdiv_level

        poly_count = len(obj.data.polygons)

        return {
            'type': geo_type,
            'polygon_count': poly_count,
            'subdivision_level': subdiv_level,
            'topology': 'quad'
        }

    def generate_materials(self) -> Dict[str, Any]:
        """Generate PBR materials"""
        obj = bpy.context.active_object

        if obj is None:
            return {}

        # Create material
        mat = bpy.data.materials.new(name="GeneratedMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add Principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Random PBR parameters
        material_type = random.choice(['metal', 'dielectric', 'glass', 'subsurface'])

        base_color = (random.random(), random.random(), random.random(), 1.0)
        bsdf.inputs['Base Color'].default_value = base_color

        if material_type == 'metal':
            bsdf.inputs['Metallic'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = random.uniform(0.0, 0.5)
        elif material_type == 'dielectric':
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 1.0)
        elif material_type == 'glass':
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = random.uniform(0.0, 0.3)
            bsdf.inputs['Transmission'].default_value = 1.0
            bsdf.inputs['IOR'].default_value = random.uniform(1.33, 2.0)
        elif material_type == 'subsurface':
            bsdf.inputs['Subsurface'].default_value = random.uniform(0.3, 1.0)
            bsdf.inputs['Subsurface Radius'].default_value = (
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0)
            )

        obj.data.materials.append(mat)

        return {
            'primary_material': material_type,
            'material_list': [{
                'name': 'GeneratedMaterial',
                'type': material_type,
                'pbr_parameters': {
                    'base_color': '#{:02x}{:02x}{:02x}'.format(
                        int(base_color[0] * 255),
                        int(base_color[1] * 255),
                        int(base_color[2] * 255)
                    ),
                    'metallic': bsdf.inputs['Metallic'].default_value,
                    'roughness': bsdf.inputs['Roughness'].default_value,
                    'ior': bsdf.inputs['IOR'].default_value if material_type == 'glass' else 1.45,
                    'transmission': bsdf.inputs['Transmission'].default_value if material_type == 'glass' else 0.0
                }
            }]
        }

    def generate_lighting(self) -> Dict[str, Any]:
        """Generate lighting setup"""
        setup_types = ['three_point', 'natural', 'studio', 'dramatic']
        setup_type = random.choice(setup_types)

        lights = []

        if setup_type == 'three_point':
            # Key light
            key_light = bpy.data.lights.new(name="KeyLight", type='AREA')
            key_light.energy = random.uniform(50, 150)
            key_obj = bpy.data.objects.new("KeyLight", key_light)
            bpy.context.collection.objects.link(key_obj)
            key_obj.location = (random.uniform(3, 5), random.uniform(-2, 2), random.uniform(2, 4))
            key_obj.rotation_euler = (math.radians(45), 0, math.radians(random.uniform(-45, 45)))

            lights.append({
                'type': 'area_light',
                'intensity': key_light.energy,
                'position': {'x': key_obj.location.x, 'y': key_obj.location.y, 'z': key_obj.location.z}
            })

            # Fill light
            fill_light = bpy.data.lights.new(name="FillLight", type='AREA')
            fill_light.energy = random.uniform(20, 60)
            fill_obj = bpy.data.objects.new("FillLight", fill_light)
            bpy.context.collection.objects.link(fill_obj)
            fill_obj.location = (random.uniform(-5, -3), random.uniform(-2, 2), random.uniform(1, 3))

            lights.append({
                'type': 'area_light',
                'intensity': fill_light.energy,
                'position': {'x': fill_obj.location.x, 'y': fill_obj.location.y, 'z': fill_obj.location.z}
            })

            # Rim light
            rim_light = bpy.data.lights.new(name="RimLight", type='POINT')
            rim_light.energy = random.uniform(30, 80)
            rim_obj = bpy.data.objects.new("RimLight", rim_light)
            bpy.context.collection.objects.link(rim_obj)
            rim_obj.location = (random.uniform(-3, 3), random.uniform(3, 5), random.uniform(2, 4))

            lights.append({
                'type': 'point_light',
                'intensity': rim_light.energy,
                'position': {'x': rim_obj.location.x, 'y': rim_obj.location.y, 'z': rim_obj.location.z}
            })

        elif setup_type == 'natural':
            # Sun light
            sun = bpy.data.lights.new(name="Sun", type='SUN')
            sun.energy = random.uniform(1, 3)
            sun_obj = bpy.data.objects.new("Sun", sun)
            bpy.context.collection.objects.link(sun_obj)
            sun_obj.rotation_euler = (math.radians(random.uniform(30, 60)), 0, math.radians(random.uniform(0, 360)))

            lights.append({
                'type': 'sun',
                'intensity': sun.energy,
                'position': {'x': 0, 'y': 0, 'z': 0}
            })

        return {
            'setup_type': setup_type,
            'lights': lights
        }

    def generate_camera(self) -> Dict[str, Any]:
        """Generate camera setup"""
        # Create camera
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        bpy.context.scene.camera = camera_obj

        # Random focal length
        focal_lengths = [16, 24, 35, 50, 85, 100, 135, 200]
        focal_length = random.choice(focal_lengths)
        camera_data.lens = focal_length

        # Determine lens category
        if focal_length < 17:
            lens_category = 'ultra_wide'
        elif focal_length < 36:
            lens_category = 'wide'
        elif focal_length < 56:
            lens_category = 'normal'
        elif focal_length < 86:
            lens_category = 'portrait'
        else:
            lens_category = 'telephoto'

        # Camera position - orbit around origin
        distance = random.uniform(5, 10)
        angle_h = random.uniform(0, math.pi * 2)
        angle_v = random.uniform(-math.pi / 6, math.pi / 3)

        x = distance * math.cos(angle_v) * math.cos(angle_h)
        y = distance * math.cos(angle_v) * math.sin(angle_h)
        z = distance * math.sin(angle_v)

        camera_obj.location = (x, y, z)

        # Point camera at origin
        direction = -np.array([x, y, z])
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        # Depth of field
        camera_data.dof.use_dof = random.choice([True, False])
        if camera_data.dof.use_dof:
            camera_data.dof.focus_distance = distance
            camera_data.dof.aperture_fstop = random.uniform(1.4, 8.0)
            dof_type = 'shallow' if camera_data.dof.aperture_fstop < 2.8 else 'medium' if camera_data.dof.aperture_fstop < 5.6 else 'deep'
        else:
            dof_type = 'deep'

        return {
            'angle': self.classify_camera_angle(angle_v),
            'lens': {
                'focal_length': focal_length,
                'type': 'prime',
                'aperture': camera_data.dof.aperture_fstop if camera_data.dof.use_dof else 8.0
            },
            'depth_of_field': dof_type,
            'position': {'x': x, 'y': y, 'z': z},
            'rotation': {
                'x': camera_obj.rotation_euler.x,
                'y': camera_obj.rotation_euler.y,
                'z': camera_obj.rotation_euler.z
            }
        }

    def classify_camera_angle(self, angle_v: float) -> str:
        """Classify vertical camera angle"""
        if angle_v > math.pi / 4:
            return 'birds_eye'
        elif angle_v > math.pi / 12:
            return 'high_angle'
        elif angle_v > -math.pi / 12:
            return 'eye_level'
        elif angle_v > -math.pi / 4:
            return 'low_angle'
        else:
            return 'worms_eye'

    def generate_liquid_simulation(self) -> Dict[str, Any]:
        """Generate liquid simulation (simplified)"""
        # Note: Full fluid simulation requires domain and flow objects
        # This is a simplified placeholder
        return {
            'type': 'water',
            'simulation': {
                'method': 'mantaflow',
                'resolution': 128,
                'viscosity': 1.0
            }
        }

    def generate_gas_simulation(self) -> Dict[str, Any]:
        """Generate gas/smoke simulation (simplified)"""
        # Note: Full smoke simulation requires domain and flow objects
        # This is a simplified placeholder
        return {
            'type': 'smoke',
            'simulation': {
                'method': 'mantaflow',
                'resolution': 128,
                'density': 1.0
            }
        }

    def generate_captions(self, categories: Dict[str, Any]) -> Dict[str, str]:
        """Generate descriptive captions"""
        elements = []

        if 'geometry' in categories:
            elements.append(f"{categories['geometry']['type']} geometry")

        if 'materials' in categories:
            mat_type = categories['materials']['primary_material']
            elements.append(f"{mat_type} material")

        if 'lighting' in categories:
            setup = categories['lighting']['setup_type']
            elements.append(f"{setup} lighting setup")

        if 'camera' in categories:
            focal = categories['camera']['lens']['focal_length']
            angle = categories['camera']['angle']
            elements.append(f"{focal}mm lens with {angle} camera angle")

        short_caption = f"3D render featuring {', '.join(elements[:2])}"
        medium_caption = f"A 3D rendered scene with {', '.join(elements)}. "
        medium_caption += "Rendered using Cycles path tracing with realistic lighting and materials."

        return {
            'short': short_caption,
            'medium': medium_caption,
            'long': medium_caption + " Generated procedurally for AI training dataset.",
            'technical': json.dumps(categories, indent=2),
            'artistic': f"A beautifully rendered 3D scene showcasing {elements[0] if elements else 'various elements'}"
        }

    def batch_generate(self, count: int, categories: List[str]):
        """Generate multiple samples"""
        print(f"Generating {count} samples with categories: {categories}")

        for i in range(count):
            print(f"\nGenerating sample {i+1}/{count}...")
            try:
                metadata = self.generate_sample(categories)
                print(f"✓ Generated: {metadata['sample_id']}")
                print(f"  Render time: {metadata['quality_metrics']['render_time']:.2f}s")
            except Exception as e:
                print(f"✗ Error generating sample {i+1}: {str(e)}")
                continue


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Blender Generator')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--taxonomy', type=str, required=True, help='Path to taxonomy YAML')
    parser.add_argument('--count', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--categories', nargs='+', default=['geometry', 'materials', 'lighting', 'camera'],
                      help='Categories to include')

    args = parser.parse_args()

    generator = ParallelSynthGenerator(args.output, args.taxonomy)
    generator.batch_generate(args.count, args.categories)


if __name__ == '__main__':
    main()
