import json

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, TypedDict
from matplotlib.axes import Axes


class ObstacleInfo(TypedDict):
    id_: int
    name: str
    vertices: List[tuple]

class GeometricMap:
    """With boundary and obstacle coordinates.

    The function from_json() should be used to load a map.
    """

    def __init__(self, boundary_coords: List[tuple], 
                 obstacle_info_list: List[ObstacleInfo]):
        """The init should not be used directly. Use from_json() instead.

        Arguments:
            boundary_coords: A list of tuples, each tuple is a pair of coordinates.
            obstacle_info_list: A list of ObstacleInfo including ID and name.
        """
        self._boundary_coords:List[tuple] = []
        self._obstacle_info_dict:Dict[int, dict] = {}

        self.register_boundary(boundary_coords)
        for obs in obstacle_info_list:
            self.register_obstacle(obs)

    @property
    def boundary_coords(self):
        return self._boundary_coords
    
    @property
    def obstacle_info(self, id_:int) -> ObstacleInfo:
        return self._obstacle_info_dict[id_]
    
    @property
    def obstacle_coords_list(self):
        obs_coords_list = []
        for obs in self._obstacle_info_dict.values():
            obs_coords_list.append(obs['vertices'])
        return obs_coords_list

    def __call__(self) -> Tuple[List[tuple], List[List[tuple]]]:
        """Return the boundary and obstacle coordinates."""
        return self.boundary_coords, self.obstacle_coords_list
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load a map from a json file.

        Keys (obstacle_dict or obstacle_list):
            boundary_coords: A list of tuples, each tuple is a pair of coordinates.
            obstacle_dict: A list of ObstacleInfo (dict) including ID (optional) and name (optional).
            obstacle_list: A list of list of tuples, each sublist is an obstacle.
        """
        with open(json_path, 'r') as jf:
            data = json.load(jf)
        boundary_coords = data['boundary_coords']
        if 'obstacle_dict' in data:
            obstacle_dict_list = data['obstacle_dict']
        else:
            obstacle_coords_list = data['obstacle_list']
            obstacle_dict_list = [{'id_': i, 'name': f'obstacle_{i}', 'vertices': obs} for i, obs in enumerate(obstacle_coords_list)]
        return cls(boundary_coords, obstacle_dict_list)
    
    @classmethod
    def from_raw(cls, boundary_coords: List[tuple], obstacle_coords_list: List[List[tuple]]):
        """Load a map from raw data."""
        obstacle_dict_list = [{'id_': i, 'name': f'obstacle_{i}', 'vertices': obs} for i, obs in enumerate(obstacle_coords_list)]
        return cls(boundary_coords, obstacle_dict_list)

    def register_obstacle(self, obstacle: ObstacleInfo) -> None:
        """Check and register an obstacle to the map."""
        if not isinstance(obstacle, dict):
            raise TypeError('An obstacle info must be a dictionary.')
        if 'vertices' not in obstacle:
            raise ValueError('An obstacle info must have a key "vertices".')
        if not isinstance(obstacle['vertices'], list):
            raise TypeError('An obstacle vertices must be a list of tuples.')
        if len(obstacle['vertices'][0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        self._obstacle_info_dict[obstacle['id_']] = obstacle

    def register_boundary(self, vertices: List[tuple]) -> None:
        """Check and register the boundary to the map."""
        if not isinstance(vertices, list):
            raise TypeError('A map boundary must be a list of tuples.')
        if len(vertices[0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        self._boundary_coords = vertices


    def get_boundary_scope(self) -> Tuple[float, float, float, float]:
        """Get the boundary scope."""
        x_min = min([x[0] for x in self._boundary_coords])
        x_max = max([x[0] for x in self._boundary_coords])
        y_min = min([x[1] for x in self._boundary_coords])
        y_max = max([x[1] for x in self._boundary_coords])
        return x_min, x_max, y_min, y_max

    def get_occupancy_map(self, rescale:int=100) -> np.ndarray:
        """
        Arguments:
            rescale: The resolution of the occupancy map (rescale * real size).
        Returns:
            A numpy array of shape (height, width, 3).
        """
        if not isinstance(rescale, int):
            raise TypeError(f'Rescale factor must be int, got {type(rescale)}.')
        assert(0<rescale<2000),(f'Rescale value {rescale} is abnormal.')
        
        boundary_np = np.array(self._boundary_coords)
        width  = max(boundary_np[:,0]) - min(boundary_np[:,0])
        height = max(boundary_np[:,1]) - min(boundary_np[:,1])

        fig, ax = plt.subplots(figsize=(width, height), dpi=rescale)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.plot(np.array(self._boundary_coords)[:,0], 
                np.array(self._boundary_coords)[:,1], 'w-')
        for obs in self.obstacle_coords_list:
            x = np.array(obs)[:,0]
            y = np.array(obs)[:,1]
            plt.fill(x, y, color='k')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        occupancy_map = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        occupancy_map = occupancy_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return occupancy_map

    def plot(self, ax: Axes, original_plot_args:dict={'c':'k'}):
        ax.plot(np.array(self._boundary_coords+[self._boundary_coords[0]])[:,0], 
                np.array(self._boundary_coords+[self._boundary_coords[0]])[:,1], 
                **original_plot_args)
        for obs in self.obstacle_coords_list:
            ax.fill(np.array(obs)[:,0], np.array(obs)[:,1], 
                    fill=True, **original_plot_args)


if __name__ == '__main__':
    boundary = [(0,0), (10,0), (10,10), (0,10)]
    obstacle_list = [{"id_": 1, "vertices": [(1,1), (2,1), (2,2), (1,2)]}, 
                     {"id_": 2, "vertices": [(3,3), (4,3), (4,4), (3,4)]}
                     ]
    map = GeometricMap(boundary, obstacle_list)
    map.get_occupancy_map()
    fig, ax = plt.subplots()
    ax.axis('equal')
    map.plot(ax)
    plt.show()