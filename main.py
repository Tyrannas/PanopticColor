from pydantic import BaseModel

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

import cv2
import numpy as np
from sklearn.cluster import KMeans

class PluginParams(BaseModel):
    """
    @default_text_prop: the default text prop that will be used for text+image similarity
    @ocr_prop_name: the name of the prop that will be created after an ocr
    """
    default_text_prop: str = "tweet_text"

    ocr_prop_name: str = "ocr"
  
  
class PluginExample(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str):  
        super().__init__(name='PluginExample',project=project, plugin_path=plugin_path)  
        self.params = PluginParams()  
        self.add_action_easy(self.compute_colors, ['execute'])

    async def compute_colors(self, context: ActionContext):
        instances = await self.project.get_instances(ids=context.instance_ids)
        uniques = list({i.sha1: i for i in instances}.values())
        res = {}
        for i in uniques:
            values = get_main_color(i.url)
            res[i.sha1] = values
        await self.save_values(res)

    async def save_values(self, colors):
        commit = DbCommit()
        # first create the props in DB
        properties = []
        for letter in ['R', 'G', 'B', 'H', 'S', 'V']:
            properties.append(self.project.create_property(letter, PropertyType.number, PropertyMode.sha1))
        commit.properties.extend(properties)

        for sha1 in colors:
            for index, prop in enumerate(properties):
                commit.image_values.append(ImageProperty(property_id=prop.id, sha1=sha1, value=colors[sha1][index].item()))
            
        res = await self.project.do(commit)

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
    
    # Extraire les valeurs R, G, B
    R, G, B = main_color
    
    # Convertir en HSV (OpenCV attend l'image au format BGR)
    main_color_bgr = np.uint8([[main_color[::-1]]])  # Convertir en BGR pour OpenCV
    hsv_color = cv2.cvtColor(main_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    # Extraire les valeurs H, S, V
    H, S, V = hsv_color
    
    # Retourner les valeurs sous forme de dictionnaire
    return [R, G, B, H, S, V]