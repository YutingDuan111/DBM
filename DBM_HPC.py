import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import KDTree
from scipy.stats import linregress
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import pickle

class DielectricBreakdown:

    def __init__(self, N=100, eta=1, triangle_theta=30, triangle_height=None):
        self.N = N
        self.x_offset = 0
        self.eta = eta
        self.steps = 0
        self.phi = np.zeros((N, N)) + 0.5
        self.spark_dict = {}
        self.spark = np.zeros((N, N), dtype=bool)
        self.fixed = np.zeros((N, N), dtype=bool)
        self.laplacian_kernel = np.array([[0, 1, 0],
                                          [1, 0, 1],
                                          [0, 1, 0]]) / 4.0

        # set up a triangle seed at the top
        theta = np.deg2rad(triangle_theta)
        h = triangle_height if triangle_height is not None else N // 10
        # calculate the base length according to the triangle's vertex angle and height
        base_len = int(2 * h * np.tan(theta / 2))
        cx = N // 2
        base_y = 0  # triangle base at top (connected to top electrode)
        tip_y = base_y + h - 1  # triangle tip pointing downward

        # fill the inverted triangle area
        for y in range(base_y, tip_y + 1):
            rel_y = y - base_y
            if h > 1:
                # For inverted triangle: width decreases as we go down
                curr_half = int((base_len / 2) * (1 - rel_y / (h - 1)))
            else:
                curr_half = 0
            for x in range(cx - curr_half, cx + curr_half + 1):
                if 0 <= x < N:
                    self.spark[y, x] = True
                    self.phi[y, x] = 0
        
        # Set electrodes: top=0V (connected to the needle), bottom=1V
        self.fixed[0, :] = True
        self.fixed[-1, :] = True
        self.phi[0, :] = 0
        self.phi[-1, :] = 1


    def check_boundary_reached(self, coord, boundary_threshold=5):
        """Check if spark is near grid boundaries"""
        y, x = coord
        return (y >= self.N - boundary_threshold or
                x <= boundary_threshold or 
                x >= self.N - boundary_threshold)

    def expand_grid(self, expansion_size=50):
        """Expand grid when spark approaches boundaries"""
        old_N = self.N
        new_N = self.N + expansion_size
        cur_offset = expansion_size // 2
        self.x_offset += cur_offset

        print(f"Grid expanded from {old_N}x{old_N} to {new_N}x{new_N}")

        # Create new grids
        new_phi = np.zeros((new_N, new_N)) + 0.5
        new_spark = np.zeros((new_N, new_N), dtype=bool)
        new_fixed = np.zeros((new_N, new_N), dtype=bool)
        
        # Copy old data to center of new grid
        new_phi[:old_N-2, cur_offset:cur_offset+old_N] = self.phi[:-2,]
        new_spark[:old_N, cur_offset:cur_offset+old_N] = self.spark
        new_fixed[:old_N-2, cur_offset:cur_offset+old_N] = self.fixed[:-2,]

        # Update electrodes
        self.fixed[0, :] = True
        self.fixed[-1, :] = True
        self.phi[0, :] = 0
        self.phi[-1, :] = 1
        
        # Update class attributes
        self.N = new_N
        self.phi = new_phi
        self.spark = new_spark
        self.fixed = new_fixed


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

    def save_final_spark(self, pickle_path):
        """Save the final spark state to a pickle file"""
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.spark, f)
        print(f"Final spark state saved to {pickle_path}")

    def simulate(self, max_steps=1200, pickle_path='data.pkl', dontsave=bool):
        self.solve_phi()  # Initial potential
        self.spark_dict = {}  # clean previous spark records

        for step in range(max_steps):
            chosen = self.grow_spark()
            if chosen is None:
                print("No more growth possible.")
                break
            
            self.solve_phi()
            self.steps = step

            if not dontsave:
                self.spark_dict[step] = (chosen, self.x_offset) # add current step's spark position
                # Save progress every 100 steps
                if step % 100 == 0:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(self.spark_dict, f)

            # Check if expansion is needed
            if step % 10 == 0:
                if self.check_boundary_reached(chosen):
                    self.expand_grid()
                    self.solve_phi()

        if not dontsave: 
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.spark_dict, f)

        if dontsave:
            self.save_final_spark(pickle_path=pickle_path)

        return step


    def load_history(self, pickle_path='data.pkl'):
        """Load spark_dict from a pickle file"""
        with open(pickle_path, 'rb') as f:
            self.spark_dict = pickle.load(f)

        # Rebuild spark from dictionary
        self.steps = max(self.spark_dict.keys())
        self.x_offset = self.spark_dict[self.steps][1]
        self.N = self.x_offset * 2 + 100  # Assuming initial N was 100
        self.spark = np.zeros((self.N, self.N), dtype=bool)
        self.spark[:] = False

        for step_data in self.spark_dict.values():
            coord, cur_offset = step_data
            y, x = coord
            coord = (y, x + (self.x_offset - cur_offset))  # Adjust x with offset
            self.spark[coord] = True
        return self.spark

    def load_spark(self, pickle_path='data.pkl'):
        """Load spark state from a pickle file"""
        with open(pickle_path, 'rb') as f:
            self.spark = pickle.load(f)
            self.N = self.spark.shape[0]
        return self.spark
    
    def plot_spark(self, figsize=(6,6)):
        plt.figure(figsize=figsize)
        plt.imshow(self.spark, cmap='gray')
        plt.title(f"Dielectric Breakdown (η={self.eta}, grid={self.N}x{self.N})")
        plt.axis('off')
        plt.show()

    def plot_phi(self, figsize=(6,5)):
        eps = 1e-10
        plt.figure(figsize=figsize)
        im = plt.imshow(self.phi + eps, cmap='Reds',
                       norm=LogNorm(vmin=(self.phi + eps).min(), vmax=(self.phi + eps).max()))
        plt.colorbar(im, label='Potential φ (log scale)')
        plt.title(f'Electric Potential φ (grid={self.N}x{self.N})')
        plt.axis('off')
        plt.show()

    def plot_phi_linear(self, figsize=(6,5)):
        plt.figure(figsize=figsize)
        im = plt.imshow(self.phi, cmap='Reds',
                        vmin=self.phi.min(), vmax=self.phi.max())
        plt.colorbar(im, label='Potential φ (linear scale)')
        plt.title(f'Electric Potential φ Linear (grid={self.N}x{self.N})')
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
        plt.title(f'η={self.eta}, steps={self.steps}, grid={self.N}x{self.N}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def calculate_fractal_dimension(self, r_min=1, r_max=70, num_r=25):
        points = np.column_stack(np.where(self.spark))
        tree = KDTree(points)
        radii = np.linspace(r_min, r_max, num_r)
        C_r = []
        
        for r in radii:
            counts = tree.query_ball_point(points, r, return_length=True)
            C_r.append(np.mean(counts))
        
        def fit_fractal_dimension(radii, C_r):
            log_r = np.log(radii)
            logC_r = np.log(C_r)
            slope, _, _, _, _ = linregress(log_r, logC_r)
            return slope

        D = fit_fractal_dimension(radii, C_r)
        print(f"Fractal dimension D = {D:.3f}")
        plt.figure(figsize=(8, 5))
        plt.plot(np.log(radii), np.log(C_r), 'bo-', label='Data')
        plt.xlabel('log(r)')
        plt.ylabel('log(C(r))')
        plt.title(f'Fractal Dimension Fit (D = {D:.3f})')
        plt.legend()
        plt.grid()
        plt.show()
        return D
    
    def theoretical_fractal_dimension(self, eta, d_s=2, d_w=2):
        numerator = d_s**2 + eta * (d_w - 1)
        denominator = d_s + eta * (d_w - 1)
        d_f = numerator / denominator
        return d_f
    