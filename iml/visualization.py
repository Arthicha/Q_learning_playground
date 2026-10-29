
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import savgol_filter

from iml.dungeon import Dungeon

class DungeonVisualization:
	"""Matplotlib visualization for a Dungeon instance."""

	def __init__(self, dungeon: Dungeon, icon_dir="iml/icon"):
		self.dungeon = dungeon
		self.icon_dir = icon_dir

	def _load_icons(self):
		icon_names = {"o": "dragie.jpg", "s": "knight.jpg", "e": "treasie.jpg", ".": "bananie.jpg", "#": "wall.jpg", "+": "meat.jpg"}
		return {key: mpimg.imread(os.path.join(self.icon_dir, filename)) for key, filename in icon_names.items() if os.path.exists(os.path.join(self.icon_dir, filename))}

	def _draw_cell(self, ax, i, vect, icons, cmap, norm, values, subcell_q_coloring):
		dungeon = self.dungeon
		char = dungeon.tiles[i]
		x, y = i % dungeon.w, i // dungeon.w
		is_wall, is_goal = char == "#", char == "e"

		facecolor = "black" if is_wall else (1, 1, 1, 0.3)
		if vect is not None and vect.dtype.type != np.str_ and not subcell_q_coloring and not is_wall and not is_goal:
			base_color = cmap(norm(vect[i].mean() if vect.ndim == 2 else vect[i]))
			facecolor = (*base_color[:3], 0.6)

		ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=facecolor, edgecolor="grey", zorder=0))

		if char in icons and (char != "." or dungeon.p_slippery > 0):
			ax.imshow(icons[char], extent=(x - 0.5, x + 0.5, y + 0.5, y - 0.5), aspect="auto", zorder=-1, alpha=0.3 if is_wall else 1.0)

		if subcell_q_coloring and vect is not None and not is_wall and vect.ndim == 2:
			colors = []
			for v in vect[:, i]:
				norm_v = norm(v)
				base_color = cmap(norm_v)
				dynamic_alpha = 0.2 + 0.6 * norm_v
				colors.append((*base_color[:3], dynamic_alpha))

			triangles = [[(x, y), (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)], [(x, y), (x - 0.5, y - 0.5), (x + 0.5, y - 0.5)], [(x, y), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)], [(x, y), (x - 0.5, y - 0.5), (x - 0.5, y + 0.5)]]

			if is_goal or char in ("o", "O"):
				color = "blue" if is_goal else "red"
				for points in triangles:
					ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="none", zorder=2, alpha=0.2))
			else:
				for points, color in zip(triangles, colors):
					ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="none", zorder=2))

		if not is_wall:
			ax.plot([x - 0.5, x + 0.5], [y - 0.5, y + 0.5], color="black", linewidth=0.5, alpha=0.4, zorder=3)
			ax.plot([x - 0.5, x + 0.5], [y + 0.5, y - 0.5], color="black", linewidth=0.5, alpha=0.4, zorder=3)

		ax.text(x, y, char, color="white" if is_wall else "black", ha="center", va="center", fontweight="bold", zorder=4)

		if values and vect is not None: #and vect.ndim == 2 and vect.shape[1] == 4:
			offsets = [(x, y + 0.28), (x, y - 0.28), (x + 0.28, y), (x - 0.28, y)]
			for k in range(min(vect.shape[0], len(offsets))):
				px, py = offsets[k]
				text = vect[k][i] if vect.dtype.type == np.str_ else f"{vect[k][i]:.2f}"
				ax.text(px, py, text, color="white" if is_goal else "black", ha="center", va="center", fontsize=8, zorder=5)

	def _draw_trajectory(self, ax, trajectory):
		if trajectory is None or len(trajectory) == 0:
			return
		dungeon = self.dungeon
		coords = np.array([[s % dungeon.w, s // dungeon.w] for s in trajectory], dtype=float)
		ax.plot(coords[:, 0], coords[:, 1], color="crimson", linewidth=2, marker="o", markersize=5, markeredgecolor="black", zorder=10)
		ax.scatter(*coords[0], color="lime", s=100, edgecolors="black", zorder=11, label="Start")
		ax.scatter(*coords[-1], color="gold", s=100, edgecolors="black", zorder=11, label="End")
		ax.legend(loc="upper right", fontsize=8)

	def _plot_trajectory(self, ax, path_states, path_actions):
		dungeon = self.dungeon
		coords = np.array([[s % dungeon.w, s // dungeon.w] for s in path_states], dtype=float)

		for i in range(len(coords) - 1):
			p1, p2 = coords[i], coords[i + 1]
			action = path_actions[i]

			if np.allclose(p1, p2):
				bumps = {0: (0, 0.4), 1: (0, -0.4), 2: (0.4, 0), 3: (-0.4, 0)}
				control = p1 + np.array(bumps.get(action, (0, 0)))
				ax.plot([p1[0], control[0], p2[0]], [p1[1], control[1], p2[1]], color="crimson", linestyle="--", linewidth=1.2, alpha=0.7, zorder=9)
			else:
				mid = (p1 + p2) / 2
				bend_axis = 1 if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]) else 0
				mid[bend_axis] += np.random.choice([-1, 1]) * np.random.uniform(0.12, 0.22)
				path = Path([p1, mid, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
				ax.add_patch(PathPatch(path, facecolor="none", edgecolor="crimson", lw=1.8, alpha=0.85, zorder=9))

		ax.scatter(coords[:, 0], coords[:, 1], color="crimson", s=30, zorder=10)
		ax.scatter(*coords[0], color="lime", s=140, edgecolor="black", zorder=11, label="Start")
		ax.scatter(*coords[-1], color="gold", s=140, edgecolor="black", zorder=11, label="End")
		ax.legend(loc="upper right", fontsize=8)

	def plot_training_reward(self, list_of_total_reward,window=10):

		if len(list_of_total_reward) > 10:

			k = np.clip(min(window,len(list_of_total_reward)//10),2,None)

			plt.plot(list_of_total_reward)
			plt.plot(savgol_filter(np.array(list_of_total_reward),k,2))
			plt.ylabel("reward")
			plt.xlabel("episode")
			plt.show()

		else:

			print("[ERROR] #reward (to plot) < 10.0, plot skipped.")


	def show(self, states=None, actions=None, vect=None, size=1.0, values=True, subcell_q_coloring=True):
		dungeon = self.dungeon
		fig, ax = plt.subplots()
		fig.set_size_inches(dungeon.w * size, dungeon.h * size)
		ax.set_xlim(-0.5, dungeon.w - 0.5)
		ax.set_ylim(dungeon.h - 0.5, -0.5)
		ax.set_aspect(1)
		ax.set_axis_off()

		cmap = plt.get_cmap("coolwarm_r")
		norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

		if vect is not None and vect.dtype.type != np.str_:
			v_min, v_max = vect.min(), vect.max()
			v_low, v_high = (-1, 1) if abs(v_min) < 1 and abs(v_max) < 1 else (min(v_min, -1e-5), max(v_max, 1e-5))
			norm = TwoSlopeNorm(vmin=v_low, vcenter=0, vmax=v_high)

		icons = self._load_icons()
		for i in range(dungeon.n_states):
			self._draw_cell(ax, i, vect, icons, cmap, norm, values, subcell_q_coloring)

		if states is not None and actions is not None:
			self._plot_trajectory(ax, states, actions)
		elif states is not None:
			self._draw_trajectory(ax, states)

		plt.tight_layout()
		plt.show()
