from math import pi

def sphere_volume(diameter: float) -> float:
  return 4/3 * pi * (diameter/2)**3

def normalized_sphere_intensity(intensity: float, diameter: float) -> float:
  return intensity / sphere_volume(diameter)