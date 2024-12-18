from pydantic import BaseModel

import cv2
import math
import colorsys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface


class ColorsPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):  
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.add_action_easy(self.compute_colors, ['execute'])

    async def compute_colors(self, context: ActionContext):
        instances = await self.project.get_instances(ids=context.instance_ids)
        uniques = list({i.sha1: i for i in instances}.values())
        res = {}
        for i in uniques:
            # values = get_main_color(i.url)
            rgb_array = get_main_color(i.url)
            # values = get_dominant(i.url)
            values = step(rgb_array, 8)
            res[i.sha1] = values
        res = await self.save_values(res)
        return ActionResult(commit=res)
        
    async def save_values(self, colors):
        commit = DbCommit()
        # first create the props in DB
        properties = []
        for letter in ['R', 'G', 'B', 'H', 'S', 'V', 'L']:
            properties.append(self.project.create_property("color_" + letter, PropertyType.number, PropertyMode.sha1))
        commit.properties.extend(properties)

        for sha1 in colors:
            for index, prop in enumerate(properties):
                commit.image_values.append(ImageProperty(property_id=prop.id, sha1=sha1, value=colors[sha1][index]))
            
        res = await self.project.do(commit)
        return res


def get_main_color(image_path, n_colors=1):
    # Lire l'image avec OpenCV
    image = cv2.imread(image_path)
    # Convertir en espace de couleurs RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionner l'image pour accélérer le traitement
    resized_image = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_AREA)

    # Reshape pour préparer l'image au clustering
    pixels = resized_image.reshape(-1, 3)

    # Utilisation de KMeans pour trouver la couleur dominante
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Récupérer la couleur centrale (la plus représentée)
    main_color = kmeans.cluster_centers_[0].astype(int)
    return main_color


def rgb_to_hsl(rgb_arr):
    # Extraire les valeurs R, G, B
    R, G, B = rgb_arr
    # Convertir en HSV (OpenCV attend l'image au format BGR)
    main_color_bgr = np.uint8([[rgb_arr[::-1]]])  # Convertir en BGR pour OpenCV
    hsv_color = cv2.cvtColor(main_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    # Extraire les valeurs H, S, V
    H, S, V = hsv_color
    
    # Retourner les valeurs sous forme de dictionnaire
    return [round(R), round(G), round(B), int(H), int(S), int(V)]


def step(rgb_arr, repetitions=1):
    """
    taken from https://www.alanzucconi.com/2015/09/30/colour-sorting/
    """
    r, g, b = rgb_arr
    r_norm, g_norm, b_norm = r / 255, g / 255, b / 255
    l = math.sqrt(.241 * r + .691 * g + .068 * b)
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h2 = int(h * repetitions)
    l2 = int(l * repetitions)
    v2 = int(v * repetitions)
    s2 = int(s * repetitions)
    return [int(r), int(g), int(b), h2, v2, s2, l2]
