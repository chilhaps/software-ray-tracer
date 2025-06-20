from myshapes import Sphere, Triangle, Plane
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys, os, json

MAX_BOUNCES = 4

cwd = os.getcwd()

if not os.path.exists(os.path.join(cwd, sys.argv[1])):
	print('"{}" not found.'.format(sys.argv[1]))
	sys.exit()

scene_fn = sys.argv[1]

try:
	res = int(sys.argv[2])
except:
	print('"{}" not a valid resolution.'.format(sys.argv[2]))
	sys.exit()

# Scene Loader (provided by instructor)
def loadScene(scene_fn):
	with open(scene_fn) as f:
		data = json.load(f)

	spheres = []

	for sphere in data["Spheres"]:
		spheres.append(
			Sphere(sphere["Center"], sphere["Radius"], 
		 	sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
		 	sphere["Kd"], sphere["Ks"], sphere["Ka"]))
		
	triangles = []

	for triangle in data["Triangles"]:
		triangles.append(
			Triangle(triangle["A"], triangle["B"], triangle["C"],
			triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
			triangle["Kd"], triangle["Ks"], triangle["Ka"]))
	
	planes = []

	for plane in data["Planes"]:
		planes.append(
			Plane(plane["Normal"], plane["Distance"],
			plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
			plane["Kd"], plane["Ks"], plane["Ka"]))
	
	objects = spheres + triangles + planes

	camera = {
		"LookAt": np.array(data["Camera"]["LookAt"],),
		"LookFrom": np.array(data["Camera"]["LookFrom"]),
		"Up": np.array(data["Camera"]["Up"]),
		"FieldOfView": data["Camera"]["FieldOfView"]
	}

	light = {
		"DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
		"LightColor": np.array(data["Light"]["LightColor"]),
		"AmbientLight": np.array(data["Light"]["AmbientLight"]),
		"BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
	}

	return camera, light, objects

# Compute Gram-Schmidt Orthogonalization
def gso(p_at, p_from, v_up):
    e3 = (p_at - p_from) / np.linalg.norm(p_at - p_from)
    e1 = np.cross(e3, v_up) / np.linalg.norm(np.cross(e3, v_up))
    e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3))

    return e1, e2, e3

# Calculate window points
def gen_window_points(res_h, res_w, fov_x, fov_y, p_at, p_from, e1, e2):
	# Initilaize empty window
	window = np.zeros((res_h, res_w, 3), dtype=np.float32)

	# Calculate all values
	d = np.linalg.norm(p_at - p_from)
	u_max = d * np.tan(fov_x / 2)
	u_min = -u_max
	v_max = d * np.tan(fov_y / 2)
	v_min = -v_max
	du = (u_max - u_min) / (res_w + 1)
	dv = (v_max - v_min) / (res_h + 1)

	# Debug print statement
	# print('d: {}, u_max: {}, u_min: {}, v_max: {}, v_min: {}, du: {}, dv: {}'.format(d, u_max, u_min, v_max, v_min, du, dv))

	# Calculate index and s-coordinate for each pixel
	for i in range(0, len(window)):
		i2 = -i + res_h / 2
		if i2 <= 0: i2 -= 1

		for j in range(0, len(window[i])):
			j2 = j - res_w / 2
			if j2 >= 0: j2 += 1

			# Calculate s-coordinate at pixel (i, j)
			window[i][j] = p_at + du * (j2 + 0.5) * e1 + dv * (i2 + 0.5) * e2

	return window

# Calculate intersections for ray
def cast_ray(r_o, r_d, obj_list):
	t_min = -1
	closest_object = obj_list[0]

	# Calculate t-value for each object in scene
	for object in obj_list:
		t = object.intersect(r_o, r_d)

		# Update minimum t-value and closest object
		if ((t > 0 and t < t_min) or (t_min <= 0 and t > 0)):
			t_min = t
			closest_object = object

	return t_min, closest_object

# Define function to calculate reflection of given vector at normal
def calc_refl(normal, vector):
	r = 2 * np.dot(normal, vector) * normal - vector
	r_hat = r / np.linalg.norm(r)

	return r, r_hat

# Calculate Phong lighting
def phong(r_o, r_d, t, obj, obj_list, light, max_bounces, current_bounce=0):
	# Stop shading when bounce limit is reached
	if current_bounce == max_bounces: return 0

	# Don't shade empty pixels
	if t == -1: return light['BackgroundColor']

	# Calculate intersection point in world space
	p = r_o + r_d * t

	# Get object material attributes
	k_d = obj.getKd()
	k_s = obj.getKs()
	k_a = obj.getKa()
	k_r = obj.getRefl()
	
	# Calculate normal based on object type
	if isinstance(obj, Sphere):
		n = obj.getNormal(p)
	else:
		n = obj.getNormal()

	n_hat = n / np.linalg.norm(n)

	# Get direction to light
	l = light['DirectionToLight']
	l_hat = l / np.linalg.norm(l)

	# Calculate reflection vector
	r_hat = calc_refl(n, l)[1]

	# Calculate direction from point to ray origin
	v = -r_d * t
	v_hat = v / np.linalg.norm(v)

	# Get light color and ambient value
	s = light['LightColor']
	s_amb = light['AmbientLight']

	# Get material diffusion, specular, ambient, and gloss value
	m_diff = obj.getDiffuse()
	m_spec = obj.getSpecular()
	m_amb = m_diff
	m_gls = obj.getGloss()

	# Calculate diffusion, specular, and ambient value at point
	c_diff = (s * m_diff) * np.clip(np.dot(n_hat, l_hat), 0, None)
	c_spec = (s * m_spec) * np.clip(np.dot(v_hat, r_hat), 0, None)**m_gls
	c_amb = s_amb * m_amb

	# Calculate offset value using normal value at point
	offset = 0.001 * n_hat

	# Cast ray to light to determine whether or not point is in shadow
	shade_t = cast_ray(p + offset, light['DirectionToLight'], obj_list)[0]

	# Set diffusion and specular coefficients to 0 if point is in shadow
	if shade_t != -1:
		k_d = 0
		k_s = 0

	# Calculate reflection vector at point and cast ray
	refl_v_hat = calc_refl(n, -r_d)[1]
	refl_t, refl_obj = cast_ray(p + offset, refl_v_hat, obj_list)

	# Increment bounce count
	current_bounce += 1

	# Calculate final color at point, clip to 0..1
	return np.clip(k_d * c_diff + k_s * c_spec + k_a * c_amb + k_r * phong(p + offset, refl_v_hat, refl_t, refl_obj, obj_list, light, max_bounces, current_bounce), 0, 1)

# Initialize scene objects and window
camera, light, objects = loadScene(scene_fn)
image = np.zeros((res,res,3), dtype=np.float32)

print('Calculating window coordinates...')

# Perform GSO and define window points in world space
e1, e2, e3 = gso(camera['LookAt'], camera['LookFrom'], camera['Up'])
window = gen_window_points(res, res, np.deg2rad(camera['FieldOfView']), np.deg2rad(camera['FieldOfView']), camera['LookAt'], camera['LookFrom'], e1, e2)

print('Done!')

# Initialize origin for camera rays
ray_origin = camera["LookFrom"]

# Calculate directions for all rays shot from camera through window points
ray_directions = (window - ray_origin) / (np.linalg.norm(window - ray_origin))

print('Rendering...')

# Define progress bar
with tqdm(total=len(ray_directions) * len(ray_directions[0])) as pbar:
	# Cast each ray, perform phong shading at intersection points, update progress bar
	for i in range(0, len(ray_directions)):
		for j in range(0, len(ray_directions[i])):
			t, obj = cast_ray(ray_origin, ray_directions[i][j], objects)
			image[i][j] = phong(ray_origin, ray_directions[i][j], t, obj, objects, light, MAX_BOUNCES)
			pbar.update()

print('Done!')

# Save and Display Output
plt.imsave("output.png", image)
plt.imshow(image);plt.show()
