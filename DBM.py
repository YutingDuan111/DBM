import numpy as np
from scipy.ndimage import convolve
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import pickle

class DielectricBreakdown:

    def __init__(self, N=100, eta=1, seed_mode='point', triangle_theta=30, triangle_height=None):
        self.N = N
        self.eta = eta
        self.steps = 0
        self.phi = np.zeros((N, N)) + 0.5
        self.seed_mode = seed_mode
        self.spark_dict = {}
        self.spark = np.zeros((N, N), dtype=bool)
        self.fixed = np.zeros((N, N), dtype=bool)
        self.laplacian_kernel = np.array([[0, 1, 0],
                                          [1, 0, 1],
                                          [0, 1, 0]]) / 4.0

        if seed_mode == 'point':
            # seed point at the bottom center
            self.spark[self.N - 2, self.N // 2] = True
            self.set_electrodes(top=True, bottom=True)
            self.phi[0, :] = 1      # top electrode φ=1
            self.fixed[0, :] = True
            self.phi[-1, :] = 0     # bottom electrode φ=0
            self.fixed[-1, :] = True

        elif seed_mode == 'triangle':
            # set up a triangle seed
            theta = np.deg2rad(triangle_theta)
            h = triangle_height if triangle_height is not None else N // 10
            # caculate the base length according to the triangle's vertex angle and height
            base_len = int(2 * h * np.tan(theta / 2))
            cx = N // 2
            base_y = N - 1
            tip_y = base_y - h + 1
            left_x = cx - base_len // 2
            right_x = cx + base_len // 2

            # fill the triangle area
            for y in range(tip_y, base_y + 1):
                rel_y = y - tip_y
                if h > 1:
                    curr_half = int((base_len / 2) * (rel_y / (h - 1)))
                else:
                    curr_half = 0
                for x in range(cx - curr_half, cx + curr_half + 1):
                    if 0 <= x < N:
                        self.spark[y, x] = True
                        self.phi[y, x] = 0
            # set bottom fixed
            for x in range(left_x, right_x + 1):
                if 0 <= x < N:
                    self.fixed[base_y, x] = True
                    self.phi[base_y, x] = 0
            self.set_electrodes(top=True, bottom=False)
            self.phi[0, :] = 1
            self.fixed[0, :] = True


    def set_electrodes(self, top=True, bottom=True):
        """Set electrode positions"""
        if top:
            self.fixed[0, :] = True
            self.phi[0, :] = 0
        if bottom:
            self.fixed[-1, :] = True
            self.phi[-1, :] = 1



    def solve_phi(self, tol=1e-5):
        while True:
            new_phi = convolve(self.phi, self.laplacian_kernel, mode='nearest')
            new_phi[:, 0] = new_phi[:, 1]  # Neumann left
            new_phi[:, -1] = new_phi[:, -2]  # Neumann right
            new_phi[self.fixed | self.spark] = self.phi[self.fixed | self.spark]
            if np.max(np.abs(new_phi - self.phi)) < tol:
                break
            self.phi = new_phi
        return self.phi


    def get_candidates(self):
        """Get possible growth points"""
        candidates = set()
        for i, j in zip(*np.where(self.spark)):
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < self.N and 0 <= nj < self.N and not self.spark[ni, nj]:
                    candidates.add((ni, nj))
        return list(candidates)


    def grow_spark(self):
        candidates = self.get_candidates()
        if not candidates:
            return None

        probs = np.array([self.phi[pt]**self.eta for pt in candidates])
        probs /= probs.sum()
        
        chosen_idx = np.random.choice(len(candidates), p=probs)
        chosen = candidates[chosen_idx]
        self.spark[chosen] = True
        self.phi[chosen] = 0
        return chosen


    def simulate(self, max_steps=5000, pickle_path='data.pkl'):
        self.solve_phi()  # Initial potential
        self.spark_dict = {}  # clean previous spark records
        for step in range(max_steps):
            chosen = self.grow_spark()
            if chosen is None:
                print("No more growth possible.")
                break

            self.solve_phi()

            if chosen[0] == 0:
                print(f"Breakdown reached top at step {step}.")
                break

            self.spark_dict[step] = chosen  # add current step's spark position

        self.steps = step

        with open(pickle_path, 'wb') as f:
            pickle.dump(self.spark_dict, f)

        return step

    def load_content(self, pickle_path='data.pkl'):
        """Load spark_dict from a pickle file"""
        with open(pickle_path, 'rb') as f:
            self.spark_dict = pickle.load(f)
        for coord in self.spark_dict.values():
            self.spark[coord] = True
        return self.spark_dict

    def plot_spark(self, figsize=(6,6)):
        plt.figure(figsize=figsize)
        plt.imshow(self.spark, cmap='gray')
        plt.title(f"Dielectric Breakdown (η={self.eta})")
        plt.axis('off')
        plt.show()

    def plot_phi(self, figsize=(6,5)):
        eps = 1e-10
        plt.figure(figsize=figsize)
        im = plt.imshow(self.phi + eps, cmap='Reds',
                       norm=LogNorm(vmin=(self.phi + eps).min(), vmax=(self.phi + eps).max()))
        plt.colorbar(im, label='Potential φ (log scale)')
        plt.title('Electric Potential φ')
        plt.axis('off')
        plt.show()

    def plot_phi_linear(self, figsize=(6,5)):
        plt.figure(figsize=figsize)
        im = plt.imshow(self.phi, cmap='Reds',
                        vmin=self.phi.min(), vmax=self.phi.max())
        plt.colorbar(im, label='Potential φ (linear scale)')
        plt.title('Electric Potential φ (Linear)')
        plt.axis('off')
        plt.show()
        
    def plot_overlay(self, figsize=(6, 5)):
        eps = 1e-10
        plt.figure(figsize=figsize)
        
        # Plot potential field with logarithmic scale
        im = plt.imshow(self.phi + eps, cmap='Reds',
                    norm=LogNorm(vmin=(self.phi + eps).min(), 
                                vmax=(self.phi + eps).max()))
        
        # Plot breakdown path
        plt.imshow(np.ma.masked_where(~self.spark, self.spark),
                cmap=plt.cm.colors.ListedColormap(['black']),
                vmin=0, vmax=1, 
                interpolation='none')  # Disable interpolation for sharp edges
        
        plt.colorbar(im, label='Potential φ (log scale)')
        plt.title(f'η={self.eta}, seed_mode={self.seed_mode}, steps={self.steps}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()